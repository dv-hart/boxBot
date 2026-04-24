"""Hailo-8L NPU interface via HailoRT.

Provides raw model-level inference.  Callers handle their own pre- and
post-processing (resize, NMS, embedding normalization).  The HAL doesn't
know what YOLO or ReID is — it runs tensors through compiled HEF models.

Hardware: Raspberry Pi AI HAT+ (Hailo-8L, 13 TOPS INT8)
Interface: PCIe via GPIO header
Library: hailo_platform (HailoRT Python API)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import numpy as np

from boxbot.hardware.base import (
    HardwareModule,
    HardwareUnavailableError,
    HealthStatus,
    ModelInfo,
)

logger = logging.getLogger(__name__)

# Default model paths
_DEFAULT_MODELS: dict[str, str] = {
    "yolo": "/usr/share/hailo-models/yolov5s_personface_h8l.hef",
    "reid": "/home/jhart/data/perception/models/repvgg_a0_person_reid_512.hef",
}


class Hailo(HardwareModule):
    """Hailo-8L NPU raw inference interface.

    Loads compiled HEF models at startup and provides ``infer()`` for
    single-shot inference and ``inference_session()`` for multi-step
    operations that should not be interleaved.
    """

    name = "hailo"

    def __init__(self, models: dict[str, str] | None = None) -> None:
        super().__init__()
        self._model_paths: dict[str, str] = models or dict(_DEFAULT_MODELS)

        # Populated by start()
        self._vdevice: Any = None
        self._hefs: dict[str, Any] = {}
        self._network_groups: dict[str, Any] = {}
        self._network_group_params: dict[str, Any] = {}
        self._input_vstream_infos: dict[str, Any] = {}
        self._output_vstream_infos: dict[str, Any] = {}

        # Single lock for all inference.  v1 simplicity — no priority
        # preemption.  Upgrade path: replace with a priority-aware lock
        # where realtime requests can preempt batch by signaling an
        # asyncio.Event that batch holders check between inference calls.
        self._lock = asyncio.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize HailoRT, detect device, load models."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._start_sync)
            self._started = True
            await self._emit_health(HealthStatus.OK)
            logger.info(
                "Hailo started: %d model(s) loaded (%s)",
                len(self._network_groups),
                ", ".join(self._network_groups.keys()),
            )
        except Exception as exc:
            await self._emit_health(HealthStatus.ERROR, str(exc))
            raise HardwareUnavailableError(
                f"Hailo not available: {exc}"
            ) from exc

    def _start_sync(self) -> None:
        """Blocking HailoRT initialization (runs in executor)."""
        from hailo_platform import (  # type: ignore[import-untyped]
            ConfigureParams,
            FormatType,
            HEF,
            HailoStreamInterface,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        self._vdevice = VDevice()

        for name, path in self._model_paths.items():
            logger.debug("Loading Hailo model %s from %s", name, path)
            hef = HEF(path)
            self._hefs[name] = hef

            configure_params = ConfigureParams.create_from_hef(
                hef, interface=HailoStreamInterface.PCIe
            )
            network_group = self._vdevice.configure(hef, configure_params)[0]
            self._network_groups[name] = network_group
            self._network_group_params[name] = network_group.create_params()

            self._input_vstream_infos[name] = hef.get_input_vstream_infos()
            self._output_vstream_infos[name] = hef.get_output_vstream_infos()

    async def stop(self) -> None:
        """Release HailoRT device and resources."""
        if self._vdevice is not None:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self._stop_sync)
            except Exception:
                logger.exception("Error stopping Hailo")
            finally:
                self._vdevice = None
                self._hefs.clear()
                self._network_groups.clear()
                self._network_group_params.clear()
                self._input_vstream_infos.clear()
                self._output_vstream_infos.clear()
        self._started = False
        await self._emit_health(HealthStatus.STOPPED)

    def _stop_sync(self) -> None:
        """Blocking HailoRT cleanup (runs in executor)."""
        if self._vdevice is not None:
            self._vdevice.release()

    # ── Inference ──────────────────────────────────────────────────

    async def infer(
        self,
        model_name: str,
        input_data: np.ndarray,
        priority: str = "realtime",
    ) -> dict[str, np.ndarray]:
        """Run a single inference on a loaded model.

        Args:
            model_name: Name of a loaded model (e.g. "yolo", "reid").
            input_data: Input tensor matching the model's expected shape.
            priority: "realtime" or "batch".  Currently both use the
                same lock (v1 simplicity).

        Returns:
            Dict mapping output layer name to numpy array.

        Raises:
            KeyError: If ``model_name`` is not loaded.
            RuntimeError: If the module is not started.
        """
        if not self._started:
            raise RuntimeError("Hailo is not started")
        if model_name not in self._network_groups:
            raise KeyError(
                f"Model '{model_name}' not loaded. "
                f"Available: {list(self._network_groups.keys())}"
            )

        async with self._lock:
            loop = asyncio.get_event_loop()
            result: dict[str, np.ndarray] = await loop.run_in_executor(
                None, self._infer_sync, model_name, input_data
            )
            return result

    def _infer_sync(
        self, model_name: str, input_data: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Blocking inference (runs in executor)."""
        from hailo_platform import (  # type: ignore[import-untyped]
            FormatType,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
        )

        network_group = self._network_groups[model_name]
        input_infos = self._input_vstream_infos[model_name]
        output_infos = self._output_vstream_infos[model_name]

        input_params = InputVStreamParams.make(
            network_group, format_type=FormatType.FLOAT32
        )
        output_params = OutputVStreamParams.make(
            network_group, format_type=FormatType.FLOAT32
        )

        with InferVStreams(
            network_group, input_params, output_params
        ) as pipeline:
            input_dict = {input_infos[0].name: input_data}
            output = pipeline.infer(input_dict)

        return output

    @asynccontextmanager
    async def inference_session(
        self, priority: str = "realtime"
    ) -> AsyncIterator[InferenceSession]:
        """Hold the inference lock for a multi-step operation.

        Prevents interleaving between logical operations (e.g. YOLO
        detection followed by N x ReID crops).

        Args:
            priority: "realtime" or "batch".  v1 uses a single lock
                for both — the priority parameter is accepted for API
                compatibility with the planned priority upgrade.

        Usage::

            async with hailo.inference_session() as session:
                boxes = await session.infer("yolo", frame)
                for crop in crops:
                    emb = await session.infer("reid", crop)
        """
        async with self._lock:
            yield InferenceSession(self)

    # ── Model info ─────────────────────────────────────────────────

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get metadata for a loaded model.

        Raises:
            KeyError: If ``model_name`` is not loaded.
        """
        if model_name not in self._hefs:
            raise KeyError(
                f"Model '{model_name}' not loaded. "
                f"Available: {list(self._hefs.keys())}"
            )

        hef = self._hefs[model_name]
        input_infos = self._input_vstream_infos[model_name]
        output_infos = self._output_vstream_infos[model_name]

        # Take the first input/output shape (most models have one of each)
        input_shape = tuple(input_infos[0].shape) if input_infos else ()
        output_shape = tuple(output_infos[0].shape) if output_infos else ()

        return ModelInfo(
            name=model_name,
            path=self._model_paths[model_name],
            input_shape=input_shape,
            output_shape=output_shape,
        )

    # ── Health ─────────────────────────────────────────────────────

    @property
    def temperature(self) -> float | None:
        """Hailo die temperature in Celsius, or None if unavailable."""
        if self._vdevice is None:
            return None
        try:
            return self._vdevice.get_chip_temperature().ts0_temperature
        except Exception:
            logger.debug("Could not read Hailo temperature", exc_info=True)
            return None

    @property
    def is_available(self) -> bool:
        """Whether the Hailo device is accessible."""
        if self._vdevice is not None:
            return True
        try:
            from hailo_platform import VDevice  # type: ignore[import-untyped]

            vd = VDevice()
            vd.release()
            return True
        except Exception:
            return False

    async def health_check(self) -> HealthStatus:
        """Check Hailo health via device temperature readability."""
        if not self._started:
            return HealthStatus.STOPPED
        if self._vdevice is None:
            return HealthStatus.ERROR
        # If we can read temperature, device is responsive
        temp = self.temperature
        if temp is None:
            return HealthStatus.DEGRADED
        return HealthStatus.OK


class InferenceSession:
    """Returned by ``Hailo.inference_session()``.

    Holds the inference lock for the duration of the context manager,
    allowing multi-step operations without interleaving.
    """

    def __init__(self, hailo: Hailo) -> None:
        self._hailo = hailo

    async def infer(
        self, model_name: str, input_data: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Run inference within the held session (lock already acquired).

        Same semantics as ``Hailo.infer()`` but does not re-acquire the
        lock.
        """
        if not self._hailo._started:
            raise RuntimeError("Hailo is not started")
        if model_name not in self._hailo._network_groups:
            raise KeyError(
                f"Model '{model_name}' not loaded. "
                f"Available: {list(self._hailo._network_groups.keys())}"
            )

        loop = asyncio.get_event_loop()
        result: dict[str, np.ndarray] = await loop.run_in_executor(
            None, self._hailo._infer_sync, model_name, input_data
        )
        return result
