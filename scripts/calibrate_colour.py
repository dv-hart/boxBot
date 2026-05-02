"""Solve picamera2 ColourGains + ColourCorrectionMatrix from a reference shot.

Usage:
    python3 scripts/calibrate_colour.py <image.jpg>

The script reads a hard-coded list of (region, true_sRGB) reference patches —
edit ``REFERENCES`` for your shot. It samples each region's mean RGB,
solves white-balance gains from the patch flagged as ``white``, then fits a
3x3 colour-correction matrix via least-squares against the true colours.

Prints YAML you can paste into ``config/config.yaml`` under
``hardware.camera``. Restart boxbot to apply.

Note on iteration: picamera2's ``set_controls`` REPLACES the IPA's default
gains+CCM (it does not compose with them). So the values produced here are
a first-pass approximation derived from the IPA-processed output. After
applying, take a fresh capture and re-run for a tighter fit if needed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Patch:
    label: str
    region: tuple[int, int, int, int]  # x0, y0, x1, y1
    true_rgb: tuple[int, int, int]
    is_white: bool = False


# Reference patches for the Cheerios capture (1280x720, taken 2026-05-01).
# Coordinates were chosen by eyeballing the image; refine if results are off.
REFERENCES: list[Patch] = [
    # Cheerios brand yellow background (Pantone 116, ~#FFCC00)
    Patch("box_yellow_top", (450, 95, 600, 130), (255, 204, 0)),
    # Cheerios dark navy text band (~#001F5F)
    Patch("box_navy_text", (430, 12, 720, 26), (0, 32, 91)),
    # Cheerios brand red heart bowl rim (~#E61E2A)
    Patch("heart_red_left", (300, 230, 360, 290), (230, 30, 42)),
    Patch("heart_red_bottom", (530, 470, 620, 510), (230, 30, 42)),
    # White-ish ingredient panel on the box (top right, near "11")
    Patch("box_white_panel", (810, 60, 870, 95), (245, 245, 240), is_white=True),
    # White wall behind person (right side) — secondary white reference
    Patch("wall_white", (1130, 80, 1190, 130), (235, 235, 230)),
]


def sample_patch(img: np.ndarray, p: Patch) -> np.ndarray:
    x0, y0, x1, y1 = p.region
    region = img[y0:y1, x0:x1]
    return region.reshape(-1, 3).mean(axis=0)


def solve_gains(measured_white: np.ndarray, true_white: np.ndarray) -> tuple[float, float]:
    """Pick (red_gain, blue_gain) so green channel is fixed and R/B match white."""
    # Normalise both vectors to their green channel, then gain = true / measured.
    m = measured_white / measured_white[1]  # ratio R/G, 1.0, B/G
    t = true_white / true_white[1]
    red_gain = float(t[0] / m[0])
    blue_gain = float(t[2] / m[2])
    return red_gain, blue_gain


def apply_gains(rgb: np.ndarray, gains: tuple[float, float]) -> np.ndarray:
    out = rgb.copy().astype(np.float64)
    out[..., 0] *= gains[0]
    out[..., 2] *= gains[1]
    return out


def solve_ccm(measured: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Least-squares fit M (3x3) such that M @ measured.T ≈ true.T.

    Both arrays are (N, 3) in linear-ish RGB. We solve in 0-255 space and
    let picamera2 normalise. Constrain rows to sum near 1.0 by *not*
    normalising — the absolute scale is absorbed into ColourGains.
    """
    # Solve true = measured @ M.T  →  M.T = lstsq(measured, true)
    Mt, *_ = np.linalg.lstsq(measured, true, rcond=None)
    return Mt.T


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    img_path = Path(sys.argv[1])
    img = np.asarray(Image.open(img_path).convert("RGB"))
    print(f"Loaded {img_path} shape={img.shape}")

    # 1. Sample every patch.
    samples: list[tuple[Patch, np.ndarray]] = []
    print("\n--- Sampled vs true ---")
    for p in REFERENCES:
        m = sample_patch(img, p)
        samples.append((p, m))
        delta = m - np.array(p.true_rgb, dtype=float)
        print(
            f"  {p.label:22s} measured={str(m.round(1).tolist()):<24} "
            f"true={str(list(p.true_rgb)):<14} Δ={delta.round(1).tolist()}"
        )

    # 2. Gains from the explicit white reference.
    white_patches = [(p, m) for p, m in samples if p.is_white]
    if not white_patches:
        print("ERROR: at least one patch must have is_white=True")
        return 1
    wp, wm = white_patches[0]
    gains = solve_gains(wm, np.array(wp.true_rgb, dtype=float))
    print(f"\nColourGains (from {wp.label}): red={gains[0]:.4f} blue={gains[1]:.4f}")

    # 3. Apply gains to all measured samples, then fit CCM against true.
    measured_arr = np.array([m for _, m in samples])
    true_arr = np.array([p.true_rgb for p, _ in samples], dtype=float)
    gained = apply_gains(measured_arr, gains)

    M = solve_ccm(gained, true_arr)
    # Normalise so each row sums to 1.0 (picamera2 expects neutral grey to map
    # to neutral grey after gains; gains carry the absolute scale).
    row_sums = M.sum(axis=1, keepdims=True)
    M_norm = M / row_sums
    print("\nColourCorrectionMatrix (row-normalised):")
    for row in M_norm:
        print("  " + "  ".join(f"{v:+.4f}" for v in row))

    # 4. Predict each patch through the full pipeline for a sanity check.
    predicted = (M_norm @ gained.T).T * row_sums.flatten().mean()
    print("\n--- Predicted vs true (after gains + CCM) ---")
    for (p, _), pred in zip(samples, predicted):
        delta = pred - np.array(p.true_rgb, dtype=float)
        print(
            f"  {p.label:22s} predicted={str(pred.round(1).tolist()):<24} "
            f"true={str(list(p.true_rgb)):<14} Δ={delta.round(1).tolist()}"
        )

    # 5. YAML for config.
    print("\n--- Paste into config/config.yaml under hardware.camera ---")
    print(f"colour_gains: [{gains[0]:.4f}, {gains[1]:.4f}]")
    print("colour_correction_matrix:")
    for row in M_norm:
        print(f"  - {row[0]:+.4f}")
        print(f"  - {row[1]:+.4f}")
        print(f"  - {row[2]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
