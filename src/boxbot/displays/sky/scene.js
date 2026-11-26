/* ════════════════════════════════════════════════════════════════════
   Living Sky — scene generator, Seattle edition.
   scene = f(phase, weather, horizon, vista, time, state) — all declarative.
   Vistas: "kerry" (default — Kerry Park looking SSE: Queen Anne firs, the
   Space Needle, downtown, Elliott Bay + ferry, Mount Rainier when it is out)
   and "sound" (looking west over Puget Sound to the Olympics).
   API unchanged from phase one: scene(o) → { svg, P }, palette(phase, weather).
   Full option list, inks and safe zones: notes/scene.md
   ════════════════════════════════════════════════════════════════════ */
let UID = 0;
const rng = seed => { let s = seed >>> 0; return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296; };
const hex2rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const rgb2hex = c => '#' + c.map(v => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')).join('');
function rgb2hsl([r, g, b]) { r /= 255; g /= 255; b /= 255; const mx = Math.max(r, g, b), mn = Math.min(r, g, b); let h = 0, s = 0, l = (mx + mn) / 2;
  if (mx !== mn) { const d = mx - mn; s = l > .5 ? d / (2 - mx - mn) : d / (mx + mn); h = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4; h /= 6; } return [h, s, l]; }
function hsl2rgb([h, s, l]) { const f = (p, q, t) => { t = (t + 1) % 1; return t < 1 / 6 ? p + (q - p) * 6 * t : t < .5 ? q : t < 2 / 3 ? p + (q - p) * (2 / 3 - t) * 6 : p; };
  if (!s) return [l * 255, l * 255, l * 255]; const q = l < .5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q; return [f(p, q, h + 1 / 3) * 255, f(p, q, h) * 255, f(p, q, h - 1 / 3) * 255]; }
const tune = (hex, fs, fl) => { const c = rgb2hsl(hex2rgb(hex)); c[1] *= fs; c[2] = fl(c[2]); return rgb2hex(hsl2rgb(c)); };
const rgba = (hex, a) => { const c = hex2rgb(hex); return `rgba(${c[0]},${c[1]},${c[2]},${a})`; };
const mix = (a, b, t) => { const A = hex2rgb(a), B = hex2rgb(b); return rgb2hex(A.map((v, i) => v + (B[i] - v) * t)); };
const f1 = n => (+n).toFixed(1);

/* ── phase palettes: sky stops (zenith → horizon), 5 depth layers (far → near), ink tokens ──
   ridge[0] far hills · [1] back towers · [2] front towers · [3] Queen Anne slope · [4] foreground firs
   lit = share of windows lit · face = sunlit-facade tint + strength · side = which side the light is on
   hm = default clock time used for ferry position / window density when o.time is omitted */
const PHASE = {
  dawn: { ink: 'dark', lit: .26, side: -1, face: ['#ffd2b0', .4], hm: [7, 24],
    sky: ['#7890cc', '#b09fd6', '#eba4bf', '#fdb596', '#ffd9a8'], ridge: ['#c0a6c8', '#a187b6', '#7c669c', '#54457a', '#2c244c'],
    snow: ['#ffd0bc', '#c79cc6'], haze: '#fff0e2', cloud: ['#fff3ea', '#ffd9c4'], cloudA: .8,
    sun: { x: 236, dy: -52, r: 24, c: '#fff6dc', glow: '#ffd7a0', gr: 420 }, sunW: { x: 1300, dy: 0, glow: '#ffc8b4', gr: 700 },
    t: { text: '#281c44', accent: '#b02a5b', panel: 'rgba(255,250,246,.7)' } },
  noon: { ink: 'dark', lit: 0, side: 1, face: ['#ffffff', .3], hm: [12, 10],
    sky: ['#4a93e2', '#74b1ee', '#a4d0f6', '#cfe7fa', '#eef8fc'], ridge: ['#a3c8e4', '#77a9c9', '#4c8a9c', '#2c6965', '#143b38'],
    snow: ['#ffffff', '#c3daf0'], haze: '#f2fafe', cloud: ['#ffffff', '#eef6fc'], cloudA: .9,
    sun: { x: 842, dy: -292, r: 26, c: '#fffdf2', glow: '#ffffff', gr: 300 },
    t: { text: '#0b2545', accent: '#c4460a', panel: 'rgba(255,255,255,.62)' } },
  afternoon: { ink: 'dark', lit: 0, side: 1, face: ['#fff2d6', .34], hm: [14, 40],
    sky: ['#5596dc', '#82b4e8', '#b3d3f0', '#dbe8f2', '#f6efe2'], ridge: ['#a9c3d8', '#7fa2bb', '#557f8e', '#33605f', '#183936'],
    snow: ['#fffaf0', '#c6d6e6'], haze: '#fbf6ec', cloud: ['#ffffff', '#f3ede2'], cloudA: .9,
    sun: { x: 944, dy: -246, r: 26, c: '#fffbe8', glow: '#fff6dc', gr: 320 }, sunW: { x: 700, dy: -196 },
    t: { text: '#12263a', accent: '#b84d14', panel: 'rgba(255,255,255,.62)' } },
  golden: { ink: 'light', lit: .16, side: 1, face: ['#ffc070', .62], hm: [16, 2],
    sky: ['#27336b', '#5a4687', '#b9587f', '#f38a5e', '#ffcb78'], ridge: ['#b0627f', '#874a76', '#5d3767', '#3a2551', '#1b1230'],
    snow: ['#ffd2a4', '#c9809c'], haze: '#ffc98c', cloud: ['#8e5a8e', '#ffb98a'], cloudA: .82,
    sun: { x: 1090, dy: -34, r: 30, c: '#fff3c4', glow: '#ffd27a', gr: 640 }, sunW: { x: 590, dy: -30, gr: 480 },   /* kerry: sun is off-frame west, only its glow shows; sound: it sets behind the Olympics */
    t: { text: '#fff5e4', accent: '#ffd27a', panel: 'rgba(38,20,58,.44)' } },
  dusk: { ink: 'light', lit: .6, side: 1, face: ['#f0a088', .14], hm: [16, 52], glowOn: 1,
    sky: ['#0e1848', '#253478', '#55489a', '#a8608e', '#f0a070'], ridge: ['#66588e', '#463c78', '#30285e', '#1e1944', '#0e0b26'],
    snow: ['#e2b4c6', '#7868a6'], haze: '#b98aa8', cloud: ['#453c7c', '#c87c98'], cloudA: .8,
    sun: { x: 1120, dy: 30, r: 30, c: '#ffd9a0', glow: '#ffa870', gr: 560 }, sunW: { x: 580, dy: 34, gr: 470 }, stars: .35,
    t: { text: '#f8f0ff', accent: '#ffcf8a', panel: 'rgba(18,13,50,.5)' } },
  night: { ink: 'light', lit: .5, side: 1, face: ['#8f9be0', .1], hm: [21, 36], glowOn: 1,
    sky: ['#060a22', '#0d1740', '#18265e', '#2b3880', '#4b53a0'], ridge: ['#2b3470', '#20275a', '#171c46', '#0f1332', '#080a1e'],
    snow: ['#b6c0f2', '#5a64a8'], haze: '#5a64b8', cloud: ['#232c66', '#39448c'], cloudA: .7,
    moon: { x: 846, y: 96, r: 25 }, stars: 1,
    t: { text: '#eef0ff', accent: '#f6cf85', panel: 'rgba(10,14,46,.52)' } },
};
/* ── weather: cloud cover desaturates the phase palette and adds overlays.
   hood = Mount Rainier visibility ("the mountain is out") · veil = low cloud swallowing tower tops ── */
const WEATHER = {
  clear:   { cover: 0,   clouds: 2,  fog: .10, hood: 1 },
  partly:  { cover: .12, clouds: 6,  fog: .12, hood: .96, cap: 1 },
  fog:     { cover: .1,  clouds: 1,  fog: .2,  hood: 0, marine: 1 },
  overcast:{ cover: .85, clouds: 11, fog: .30, hood: 0, veil: .5 },
  rain:    { cover: .9,  clouds: 11, fog: .40, hood: 0, veil: .82, rain: 1 },
  snow:    { cover: .7,  clouds: 8,  fog: .34, hood: 0, veil: .7, snow: 1 },
};
const PEOPLE = {
  dark:  { p1: '#b8501a', p2: '#7440a6', p3: '#0d7a5e', p4: '#cf2f6c', family: '#24507e' },
  light: { p1: '#ffa96b', p2: '#c79cff', p3: '#74e2b8', p4: '#ff8fb4', family: '#8ccbff' },
};

function palette(phase, weather) {
  const P = PHASE[phase] || PHASE.golden, Wx = WEATHER[weather] || WEATHER.clear, k = Wx.cover, night = !!P.glowOn;
  const sky = P.sky.map(c => tune(c, 1 - .86 * k, l => night ? l + .07 * k : l * (1 - .55 * k) + (.28 + .5 * l) * .55 * k + (k > .5 ? .06 : 0)));
  const ridge = P.ridge.map(c => tune(c, 1 - .62 * k, l => l));
  const light = P.ink === 'light', text = k > .5 && !light ? tune(P.t.text, .5, l => l) : P.t.text;
  return { ...P, wx: Wx, sky, ridge, cloudA: P.cloudA,
    cloud: k > .5 ? (night ? ['#2a3158', '#3d4470'] : ['#f3f5f6', '#8f9ba6']) : P.cloud,
    haze: k > .5 ? tune(P.haze, .2, l => l) : P.haze,
    vars: {
      '--text': text, '--muted': rgba(text, light ? .76 : .7), '--accent': P.t.accent,
      '--panel': P.t.panel, '--panel-stroke': light ? 'rgba(255,235,215,.2)' : 'rgba(255,255,255,.7)',
      '--panel-top': light ? 'rgba(255,255,255,.14)' : 'rgba(255,255,255,.9)', '--panel-shadow': light ? 'rgba(4,2,18,.3)' : 'rgba(30,40,70,.13)',
      '--hair': rgba(text, light ? .18 : .13), '--hair2': rgba(text, .3), '--chip': rgba(text, light ? .12 : .07),
      '--today': light ? 'rgba(255,255,255,.12)' : 'rgba(255,255,255,.72)',
      '--guard': light ? '0 2px 26px rgba(16,8,40,.42)' : '0 1px 22px rgba(255,255,255,.5)',
      '--ground': '#fff6e6', '--ground-muted': 'rgba(255,246,230,.74)',
      ...Object.fromEntries(Object.entries(PEOPLE[P.ink]).map(([n, c]) => ['--' + n, c])),
    } };
}

/* ── shape helpers ── */
function ridgePts(base, amp, seed, sharp) {
  const r = rng(seed), p = [r() * 6.3, r() * 6.3, r() * 6.3, r() * 6.3], f = [.0042 + r() * .002, .0105 + r() * .004, .024 + r() * .008, .055];
  const pts = [];
  for (let x = -8; x <= 1032; x += 8) {
    let n = .5 * Math.sin(x * f[0] + p[0]) + .28 * Math.sin(x * f[1] + p[1]) + .15 * Math.sin(x * f[2] + p[2]) + .07 * Math.sin(x * f[3] + p[3]);
    if (sharp) n = .8 - 1.7 * Math.abs(n);
    pts.push([x, base - amp * n]);
  }
  return pts;
}
const yAt = (pts, x) => { const i = Math.max(0, Math.min(pts.length - 2, Math.floor((x + 8) / 8))); const a = pts[i], b = pts[i + 1]; return a[1] + (b[1] - a[1]) * (x - a[0]) / 8; };
const poly = pts => 'M' + pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join('L') + 'L1032,600L-8,600Z';
function fir(x, y, h, r) {
  const w = h * (.17 + .07 * r()), tiers = h > 70 ? 8 : h > 28 ? 5 : 3, L = [], R = [];
  for (let i = 0; i < tiers; i++) {
    const y0 = y - h + h * .9 * i / tiers, y1 = y - h + h * .9 * (i + 1) / tiers, w1 = w * (.2 + .8 * (i + 1) / tiers), jl = .82 + .36 * r(), jr = .82 + .36 * r();
    if (i) { L.push([x - w1 * .42 * jl, y0]); R.push([x + w1 * .42 * jr, y0]); }
    L.push([x - w1 * jl, y1 + h * .012]); R.push([x + w1 * jr, y1 + h * .012]);
  }
  const t = h < 60 ? 0 : h * .022, yb = y - h * .1;
  return `M${x.toFixed(1)},${(y - h).toFixed(1)}` + L.map(p => `L${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('') +
    `L${(x - t).toFixed(1)},${yb.toFixed(1)}L${(x - t).toFixed(1)},${(y + 6).toFixed(1)}L${(x + t).toFixed(1)},${(y + 6).toFixed(1)}L${(x + t).toFixed(1)},${yb.toFixed(1)}` +
    R.reverse().map(p => `L${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('') + 'Z';
}
function treeline(pts, seed, hmin, hmax, smin, smax) {
  const r = rng(seed); let d = '';
  for (let x = -6; x < 1030; x += smin + r() * (smax - smin)) { const big = r() > .82 ? 1.35 : 1; d += fir(x, yAt(pts, x) + 2, (hmin + r() * (hmax - hmin)) * big, r); }
  return d;
}
function cloudBand(cx, cy, w, h, r) {
  let s = ''; const n = 3 + Math.floor(r() * 3);
  for (let i = 0; i < n; i++) { const ww = w * (.4 + .6 * r()), hh = h * (.45 + .4 * r()), x = cx - w / 2 + (w - ww) * r(), y = cy - h / 2 + (h - hh) * (i / (n - 1));
    s += `<rect x="${x.toFixed(0)}" y="${y.toFixed(0)}" width="${ww.toFixed(0)}" height="${hh.toFixed(1)}" rx="${(hh / 2).toFixed(1)}"/>`; }
  return s;
}
const boxes = (list, w, h) => list.map(p => `M${f1(p[0])},${f1(p[1])}h${w}v${h}h${-w}z`).join('');
const sstep = t => { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); };

/* ── Seattle: the skyline is a place, so it is a fixed table (x, width, height, roof) in "city units":
   x as on a 1024 frame at horizon "high", y = 0 at the waterline, up is negative. Row b = back, f = front. ── */
const TOWERS = {
  b: [[286, 22, 34, 'flat'], [312, 18, 52, 'crown'], [334, 26, 44, 'flat'], [366, 20, 66, 'slant'], [390, 24, 58, 'flat'], [418, 18, 78, 'crown'], [440, 26, 62, 'step'], [470, 22, 90, 'flat'], [496, 18, 72, 'crown'],
      [600, 22, 92, 'step'], [624, 22, 118, 'curve'], [648, 24, 100, 'flat'], [674, 18, 132, 'swoop'], [694, 28, 160, 'columbia'], [726, 22, 124, 'pyr'], [750, 20, 98, 'slant'], [772, 18, 84, 'flat'], [792, 14, 66, 'crown']],
  f: [[352, 14, 40, 'flat'], [404, 16, 50, 'crown'], [456, 14, 62, 'flat'], [484, 16, 46, 'flat'], [512, 13, 74, 'crown'], [572, 15, 86, 'flat'], [590, 13, 64, 'crown'],
      [608, 16, 76, 'crown'], [632, 18, 90, 'flat'], [656, 14, 108, 'spire'], [676, 20, 82, 'step'], [706, 18, 98, 'flat'], [732, 16, 86, 'crown'], [754, 18, 70, 'notch'], [776, 14, 58, 'flat'], [794, 10, 50, 'smith'], [808, 14, 26, 'flat']],
};
/* one tower → { d: outline, cut: how far below the top the full-width body starts } */
function tower([x, w, h, t]) {
  const X = f => f1(x + w * f), Y = v => f1(-(h - v)), up = `M${f1(x)},1V`, dn = `V1Z`;
  switch (t) {
    case 'crown': return { d: `${up}${Y(0)}H${X(1)}${dn}M${X(.28)},${Y(0)}v-4.5h${f1(w * .44)}v4.5zM${X(.46)},${Y(4)}v-11h1.3v11z`, cut: 0 };
    case 'step': return { d: `${up}${Y(13)}H${X(.18)}V${Y(6)}H${X(.4)}V${Y(0)}H${X(.76)}V${Y(8)}H${X(1)}${dn}`, cut: 13 };
    case 'pyr': return { d: `${up}${Y(22)}H${X(.14)}V${Y(15)}L${X(.5)},${Y(0)}L${X(.86)},${Y(15)}V${Y(22)}H${X(1)}${dn}`, cut: 22 };
    case 'slant': return { d: `${up}${Y(0)}L${X(1)},${Y(12)}${dn}`, cut: 12 };
    case 'curve': return { d: `${up}${Y(11)}Q${X(.5)},${Y(-9)} ${X(1)},${Y(11)}${dn}`, cut: 11 };
    case 'swoop': return { d: `${up}${Y(0)}H${X(.36)}Q${X(.5)},${f1(-h * .66)} ${X(1)},${f1(-h * .52)}${dn}`, cut: h * .5 };
    case 'columbia': return { d: `${up}${Y(30)}Q${X(.02)},${Y(34)} ${X(.2)},${Y(34)}V${Y(15)}Q${X(.22)},${Y(18)} ${X(.4)},${Y(18)}V${Y(0)}H${X(.82)}V${Y(22)}H${X(1)}${dn}`, cut: 34 };
    case 'spire': return { d: `${up}${Y(10)}H${X(.22)}V${Y(4)}H${X(.38)}V${Y(0)}H${X(.62)}V${Y(4)}H${X(.78)}V${Y(10)}H${X(1)}${dn}M${X(.46)},${Y(0)}v-15h1.2v15z`, cut: 10 };
    case 'notch': return { d: `${up}${Y(0)}H${X(.38)}V${Y(7)}H${X(.62)}V${Y(0)}H${X(1)}${dn}`, cut: 7 };
    case 'smith': return { d: `${up}${f1(-h * .5)}H${X(.24)}V${Y(11)}L${X(.5)},${Y(0)}L${X(.76)},${Y(11)}V${f1(-h * .5)}H${X(1)}${dn}`, cut: h * .5 };
    default: return { d: `${up}${Y(0)}H${X(1)}${dn}`, cut: 0 };
  }
}
/* row of towers → outline path, sunlit-face path, lit-window lists (warm / pale / cool) */
function towerRow(list, side, lit, seed) {
  const r = rng(seed); let d = '', face = '', lines = ''; const w0 = [], w1 = [], w2 = [];
  list.forEach(b => { const [x, w, h] = b, T = tower(b), top = -(h - T.cut) + 1.5, fw = Math.max(3, w * .3);
    d += T.d; face += `M${f1(side > 0 ? x + w - fw : x)},${f1(top)}h${f1(fw)}V0h${f1(-fw)}z`;
    const mood = .45 + 1.1 * r(), cool = r() < .3, cols = Math.floor((w - 3) / 4.2), rows = Math.floor((-top - 5) / 5.4), ox = x + (w - cols * 4.2) / 2 + 1;
    if (r() < .5) for (let c = 1; c < cols; c++) lines += `M${f1(ox + c * 4.2 - 1)},${f1(top + 2)}V-2`;
    else for (let q = 1; q < rows; q += 2) lines += `M${f1(x + 1.5)},${f1(top + q * 5.4)}h${f1(w - 3)}`;
    for (let q = 0; q < rows; q++) { const band = r() < .07;
      for (let c = 0; c < cols; c++) { const v = r(), t = r(); if (v < lit * mood || (band && lit > 0 && v < .85)) (cool && t < .7 ? w2 : t < .32 ? w1 : w0).push([ox + c * 4.2, top + 3 + q * 5.4]); } } });
  return { d, face, lines, w: [w0, w1, w2] };
}
/* the Space Needle, drawn in metres (184 m tall) so the proportions stay honest */
function needle(cx, by, s) {
  const X = m => f1(cx + m * s), Y = m => f1(by - m * s);
  const ow = y => y <= 113 ? 3.6 + 15.2 * Math.pow(1 - y / 113, 1.75) : 3.6 + 9.6 * Math.pow((y - 113) / 35, 1.9), th = y => 3.1 - 1.6 * y / 148;
  const ys = []; for (let y = 0; y <= 148; y += 6) ys.push(y); ys.push(148);
  const leg = g => 'M' + ys.map(y => X(g * ow(y)) + ',' + Y(y)).join('L') + 'L' + ys.slice().reverse().map(y => X(g * (ow(y) - th(y))) + ',' + Y(y)).join('L') + 'Z';
  const half = [[5.5, 143.5], [10, 147], [16.6, 151], [17.6, 151.5], [17.6, 154.3], [21, 154.8], [21, 156.3], [18, 156.9], [13, 159.6], [6.4, 161.6], [3.2, 162.2], [3.2, 164.6], [1.1, 165.2]];
  const house = 'M' + half.map(p => X(-p[0]) + ',' + Y(p[1])).join('L') + `L${X(-.5)},${Y(184)}L${X(.5)},${Y(184)}L` + half.slice().reverse().map(p => X(p[0]) + ',' + Y(p[1])).join('L') + 'Z';
  return {
    legs: leg(-1) + leg(1) + `M${X(-2.1)},${Y(0)}L${X(-1.5)},${Y(146)}L${X(1.5)},${Y(146)}L${X(2.1)},${Y(0)}Z` +
      `M${X(-10.6)},${Y(29)}h${f1(21.2 * s)}v${f1(-2.6 * s)}h${f1(-21.2 * s)}zM${X(-4.6)},${Y(112)}h${f1(9.2 * s)}v${f1(-1.6 * s)}h${f1(-9.2 * s)}z`,
    house,
    band: `M${X(-17)},${Y(154)}h${f1(34 * s)}v${f1(2.3 * s)}h${f1(-34 * s)}z`,
    bowl: `M${X(-5.5)},${Y(143.5)}L${X(-10)},${Y(147)}L${X(-16.6)},${Y(151)}H${X(16.6)}L${X(10)},${Y(147)}L${X(5.5)},${Y(143.5)}Z`,
    halo: `M${X(-21)},${Y(156.3)}h${f1(42 * s)}v${f1(1.3 * s)}h${f1(-42 * s)}z`,
    roof: g => `M${X(0)},${Y(156.9)}L${X(g * 18)},${Y(156.9)}L${X(g * 13)},${Y(159.6)}L${X(g * 6.4)},${Y(161.6)}L${X(0)},${Y(161.6)}Z`,
    tip: [cx, by - 184 * s], deck: [cx, by - 155 * s],
  };
}
/* Mount Rainier from the north-north-west: broad dome, Little Tahoma on the left shoulder, long right flank */
function rainier(cx, top) {
  const p = (x, y) => f1(cx + x) + ',' + f1(top + y);
  const out = `M${p(-340, 180)}L${p(-262, 160)}Q${p(-215, 150)} ${p(-178, 128)}Q${p(-150, 112)} ${p(-128, 96)}L${p(-112, 88)}L${p(-103, 74)}L${p(-95, 84)}Q${p(-84, 80)} ${p(-72, 62)}Q${p(-58, 38)} ${p(-44, 22)}Q${p(-36, 12)} ${p(-26, 8)}Q${p(-16, 3)} ${p(-6, 1)}Q${p(2, -1)} ${p(9, 3)}Q${p(14, 7)} ${p(20, 6)}Q${p(27, 4)} ${p(33, 9)}Q${p(44, 20)} ${p(58, 36)}Q${p(80, 58)} ${p(104, 74)}Q${p(140, 98)} ${p(184, 120)}Q${p(240, 146)} ${p(300, 162)}L${p(360, 180)}Z`;
  const P = a => 'M' + a.map(q => p(q[0], q[1])).join('L') + 'Z';
  return { out,
    east: P([[2, 1], [-7, 24], [5, 44], [-9, 66], [7, 90], [-12, 114], [3, 138], [-8, 182], [-400, 182], [-400, -20], [2, -20]]),
    west: P([[2, 1], [-7, 24], [5, 44], [-9, 66], [7, 90], [-12, 114], [3, 138], [-8, 182], [400, 182], [400, -20], [2, -20]]),
    ribs: P([[34, 11], [45, 40], [36, 64], [54, 98], [43, 66], [50, 42]]) + P([[72, 50], [86, 78], [77, 100], [99, 124], [86, 98], [93, 80]]) + P([[-42, 22], [-34, 48], [-45, 70], [-32, 100], [-50, 76], [-42, 52]]) + P([[-86, 80], [-75, 100], [-87, 120], [-70, 142], [-93, 124], [-83, 104]]) + P([[130, 94], [146, 116], [138, 132], [160, 148], [146, 130], [152, 118]]) + P([[-150, 114], [-140, 130], [-152, 144], [-136, 158], [-158, 146], [-148, 132]]),
    rock: P([[-22, 10], [-15, 30], [-24, 36]]) + P([[12, 8], [19, 30], [10, 34]]) + P([[-103, 74], [-98, 92], [-110, 98]]) + P([[-60, 48], [-53, 70], [-64, 76]]),
    base: P([[-340, 180], [-262, 160], [-215, 147], [-192, 152], [-170, 130], [-150, 144], [-128, 118], [-104, 140], [-84, 116], [-60, 142], [-36, 120], [-12, 148], [12, 124], [36, 148], [60, 122], [86, 144], [110, 118], [138, 140], [164, 124], [196, 142], [230, 146], [300, 162], [360, 180]]) };
}
/* low roofs with a few taller blocks (Belltown / Interbay) along yf(x) */
function roofline(x0, x1, yf, r, lit, hmax) {
  let d = `M${x0},420L${x0},${f1(yf(x0))}`, x = x0; const w = [];
  while (x < x1) { const bw = 9 + r() * 24, h = r() < .16 ? 0 : 2 + Math.pow(r(), 1.7) * hmax, y = yf(x + bw / 2) - h;
    if (Math.abs(yf(x + bw) - yf(x)) > 2.5) { r(); x += 7; d += `L${f1(x)},${f1(yf(x))}`; continue; }
    d += `L${f1(x)},${f1(y)}L${f1(x + bw)},${f1(y)}`;
    for (let q = 0; q < 3; q++) for (let c = 0; c + 4 < bw; c += 4.4) if (r() < lit * (q ? .5 : .8)) w.push([x + 1.6 + c, y + 2.4 + q * 5]);
    x += bw; }
  return { d: d + `L${x1},${f1(yf(x1))}L${x1},420Z`, w };
}
/* Queen Anne slope: gabled houses, low apartments, round trees and firs along yf(x) */
function slope(x0, x1, yf, r, lit) {
  let d = `M${x0},420L${x0},${f1(yf(x0))}`, x = x0, firs = ''; const w = [];
  while (x < x1) { const t = r(), y = yf(x);
    if (t < .36) { const bw = 18 + r() * 16, wall = 7 + r() * 7, pk = 6 + r() * 6;
      d += `L${f1(x)},${f1(y - wall)}L${f1(x + bw / 2)},${f1(y - wall - pk)}L${f1(x + bw)},${f1(y - wall)}L${f1(x + bw)},${f1(y)}`;
      if (r() < lit * 1.5) w.push([x + bw * .22, y - wall + 2]); if (r() < lit * 1.1) w.push([x + bw * .62, y - wall + 2]); x += bw + 1 + r() * 4;
    } else if (t < .52) { const bw = 22 + r() * 26, h = 11 + r() * 15;
      d += `L${f1(x)},${f1(y - h)}L${f1(x + bw)},${f1(y - h)}L${f1(x + bw)},${f1(y)}`;
      for (let q = 0; q * 5.4 + 6 < h + 3; q++) for (let c = 0; c + 5 < bw; c += 5) if (r() < lit * .9) w.push([x + 2 + c, y - h + 2.6 + q * 5.4]); x += bw + 2;
    } else if (t < .8) { const bw = 13 + r() * 16;
      d += `Q${f1(x + bw * .06)},${f1(y - bw * .92)} ${f1(x + bw / 2)},${f1(y - bw * .86)}Q${f1(x + bw * .94)},${f1(y - bw * .92)} ${f1(x + bw)},${f1(y)}`; x += bw - 1;
    } else { firs += fir(x + 5, y + 2, 20 + r() * 28, r); d += `L${f1(x + 10)},${f1(yf(x + 10))}`; x += 10; } }
  return { d: d + `L${x1},${f1(yf(x1))}L${x1},420Z` + firs, w };
}
/* Washington State ferry: white, green stripe and stack, double-ended. Origin = midships at the waterline. */
function ferry(x, y, s, c, lit) {
  const g = `<g transform="translate(${f1(x)},${f1(y)}) scale(${s.toFixed(3)})">`;
  return g + `<path d="M-35,-1.6L-31.5,-7.4H31.5L35,-1.6L33,.4H-33Z" fill="${c.hull}"/><path d="M-33.8,-3.6H33.8L34.8,-2H-34.8Z" fill="${c.green}"/>
    <path d="M-28,-7.4v-5.4h56v5.4zM-21,-12.8v-3.4h42v3.4zM-25.5,-12.8v-6h7.5v6zM18,-12.8v-6h7.5v6z" fill="${c.white}"/>
    <path d="M-3.6,-16.2v-6h7.2v6z" fill="${c.green}"/><path d="M-3.6,-22.2v-1.5h7.2v1.5z" fill="${c.dark}"/>
    <path d="M-21.7,-18.8v-4.6M21.7,-18.8v-4.6" stroke="${c.dark}" stroke-width=".7" fill="none"/>
    <path d="M-26,-10.4H26" stroke="${lit ? '#ffe2a0' : c.dark}" stroke-width="1.5" stroke-dasharray="2.1 1.5" opacity="${lit ? 1 : .55}" fill="none"/>
    <path d="M-19,-14.6H19" stroke="${lit ? '#ffe2a0' : c.dark}" stroke-width="1.1" stroke-dasharray="1.8 1.6" opacity="${lit ? .9 : .45}" fill="none"/>
    <path d="M-30,-5.6H-24M24,-5.6H30" stroke="${c.dark}" stroke-width="1.6" opacity=".6" fill="none"/>
    ${lit ? `<circle cx="-21.7" cy="-23.8" r="1" fill="#fff"/><circle cx="21.7" cy="-23.8" r="1" fill="#fff"/><circle cx="-33" cy="-4.6" r=".9" fill="#ff6a55"/><circle cx="33" cy="-4.6" r=".9" fill="#7dffb0"/>` : ''}</g>`;
}
function parseTime(o, P) {
  if (typeof o.time === 'string' && /^\d{1,2}:\d{2}/.test(o.time)) { const m = o.time.match(/^(\d{1,2}):(\d{2})/); return [+m[1] % 24, +m[2] % 60]; }
  return [o.hour != null ? +o.hour : P.hm[0], o.minute != null ? +o.minute : P.hm[1]];
}

/* second vista: west over Puget Sound to the Olympics (sunset side). Returns svg; x.add() appends defs. */
function sound(x) {
  const { P, Wx, id, seed, H, k, mn, lit, glowOn, over, night, phase, hood, C, fog } = x, R = P.ridge; let s = '';
  const rdg = (base, amp, a, b, c) => { const q = []; for (let X = -8; X <= 1032; X += 4) { const n = .5 * (1 - Math.abs(Math.sin(X * .0074 + a))) + .32 * (1 - Math.abs(Math.sin(X * .021 + b))) + .18 * (1 - Math.abs(Math.sin(X * .052 + c))); q.push([X, base - amp * Math.pow(n, 1.25) * (.72 + .28 * Math.sin(X * .004 + a))]); } return q; };
  const O0 = rdg(H + 2 * k, 124 * k, 1.9, .6, 2.2), O1 = rdg(H + 2 * k, 54 * k, .4, 2.9, 1.1), K = ridgePts(H - 6 * k, 5 * k, 103);
  if (hood > 0) { const rock = mix(R[0], P.sky[3], .25);
    x.add(`<linearGradient id="${id}ol" gradientUnits="userSpaceOnUse" x1="0" y1="${f1(H - 100 * k)}" x2="0" y2="${H}"><stop offset="0" stop-color="${P.snow[0]}"/><stop offset=".46" stop-color="${P.snow[0]}"/><stop offset=".6" stop-color="${mix(P.snow[1], rock, .45)}"/><stop offset="1" stop-color="${rock}"/></linearGradient>`);
    s += `<g opacity="${(hood * (night ? .7 : 1)).toFixed(2)}"><path d="${poly(O0)}" fill="url(#${id}ol)"/><path d="${poly(O1)}" fill="${mix(R[0], R[1], .3)}"/></g>`; }
  s += fog(H - 12 * k, Wx.fog + .12, 70 * k);
  s += `<path d="${poly(K)}" fill="${mix(R[1], R[2], .55)}"/>`;
  if (Wx.veil) { x.add(`<linearGradient id="${id}vl" gradientUnits="userSpaceOnUse" x1="0" y1="${f1(H - 200 * k)}" x2="0" y2="${f1(H - 4 * k)}"><stop offset="0" stop-color="${P.sky[3]}" stop-opacity="0"/><stop offset=".7" stop-color="${P.sky[3]}" stop-opacity="${Wx.veil}"/><stop offset="1" stop-color="${P.sky[4]}" stop-opacity="${(Wx.veil * .5).toFixed(2)}"/></linearGradient>`);
    s += `<rect x="0" y="${f1(H - 200 * k)}" width="1024" height="${f1(196 * k)}" fill="url(#${id}vl)"/>`; }
  // the Sound
  x.add(`<linearGradient id="${id}w" gradientUnits="userSpaceOnUse" x1="0" y1="${H}" x2="0" y2="${H + 170}"><stop offset="0" stop-color="${mix(P.sky[4], R[1], over ? .4 : .34)}"/><stop offset=".3" stop-color="${C.water0}"/><stop offset=".7" stop-color="${C.water1}"/><stop offset="1" stop-color="${mix(R[3], R[4], .5)}"/></linearGradient>`);
  s += `<rect x="0" y="${H}" width="1024" height="${600 - H}" fill="url(#${id}w)"/>`;
  const wr = rng(seed + 300), u = P.sunW || P.sun, gx = P.moon ? P.moon.x : u ? u.x : -1; let sh = '', gl = '';
  for (let i = 0; i < 46; i++) { const y = 4 + Math.pow(wr(), 1.5) * 130, l = (8 + wr() * 30) * (.5 + y / 50); sh += `M${f1(wr() * 1024)},${f1(H + y)}h${f1(l)}`; }
  s += `<path d="${sh}" stroke="${mix(P.sky[4], '#ffffff', .3)}" stroke-width="1" opacity="${glowOn ? .12 : .3}" fill="none" stroke-linecap="round"/>`;
  if (!over && gx > 0 && gx < 1024 && phase !== 'noon' && phase !== 'dawn') { for (let i = 0; i < 34; i++) { const y = 3 + Math.pow(wr(), .9) * 126, l = (6 + wr() * 22) * (.5 + y / 44); gl += `M${f1(gx + (wr() - .5) * (22 + y * .7) - l / 2)},${f1(H + y)}h${f1(l)}`; }
    s += `<path d="${gl}" stroke="${P.moon ? '#dfe4ff' : phase === 'afternoon' ? '#ffffff' : '#ffe2a8'}" stroke-width="1.5" opacity="${P.moon ? .45 : .7}" fill="none" stroke-linecap="round"/>`; }
  if (lit > .1) { let dots = ''; const lr = rng(seed + 5); for (let i = 0; i < 46; i++) { const X = lr() * 1024; dots += `<circle cx="${f1(X)}" cy="${f1(yAt(K, X) + 3 + lr() * 3)}" r="${(.7 + lr() * .6).toFixed(1)}" opacity="${(.4 + lr() * .6).toFixed(2)}"/>`; } s += `<g fill="#ffd98a">${dots}</g>`; }
  // West Point: low spit and lighthouse
  { const y = H + 56 * k, lx = 792, c = mix(R[2], R[3], .6), w = glowOn ? '#aeb6e6' : mix('#ffffff', P.sky[3], .1);
    s += `<path d="M1030,${f1(y - 8)}C950,${f1(y - 10)} 860,${f1(y - 3)} 752,${f1(y + 3)}L752,${f1(y + 6)}C870,${f1(y + 9)} 960,${f1(y + 16)} 1030,${f1(y + 20)}Z" fill="${c}"/><path d="${fir(900, y - 4, 20, wr)}${fir(930, y - 5, 28, wr)}${fir(952, y - 5, 22, wr)}${fir(985, y - 6, 34, wr)}${fir(1010, y - 6, 26, wr)}" fill="${c}"/>`;
    s += `<path d="M${lx - 3},${f1(y)}L${lx - 2.2},${f1(y - 17)}h4.4L${lx + 3},${f1(y)}ZM${lx + 3},${f1(y)}v-7h13v7z" fill="${w}"/><path d="M${lx - 3.4},${f1(y - 17)}L${lx},${f1(y - 22)}L${lx + 3.4},${f1(y - 17)}ZM${lx + 2.4},${f1(y - 7)}l7.1,-4l7.1,4z" fill="${glowOn ? '#8a4a5a' : '#c2503c'}"/>`;
    if (lit > .1) s += `<circle cx="${lx}" cy="${f1(y - 15)}" r="2" fill="#fff6c8"/><circle cx="${lx}" cy="${f1(y - 15)}" r="11" fill="#fff0b0" opacity=".28"/><rect x="${lx - 1.5}" y="${f1(y + 7)}" width="3" height="26" fill="#fff0b0" opacity=".3"/>`; }
  // ferry crossing, position from the minute
  { const t = (mn % 30) / 30, back = Math.floor(mn / 30) % 2, fx = back ? 860 - 700 * t : 160 + 700 * t, fy = H + 30 * k, fs = 1.5 * (.6 + .4 * k);
    const c = glowOn ? { hull: '#8f97c8', white: '#aab2de', green: '#2f6a58', dark: '#141a3c' } : { hull: mix('#ffffff', P.sky[3], .14), white: mix('#ffffff', P.sky[3], .06), green: mix('#1f7a55', R[2], over ? .35 : .15), dark: mix('#1b2b33', R[3], .4) };
    s += `<path d="M${f1(fx + (back ? 34 : -34) * fs)},${f1(fy + .5)}h${f1((back ? 70 : -70) * fs)}" stroke="#fff" stroke-width="1.8" opacity="${glowOn ? .2 : .45}" stroke-linecap="round"/>`;
    if (lit > .1) { x.add(`<linearGradient id="${id}rw" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffd98a" stop-opacity=".9"/><stop offset="1" stop-color="#ffd98a" stop-opacity="0"/></linearGradient>`); s += `<rect x="${f1(fx - 26 * fs)}" y="${f1(fy + 1)}" width="${f1(52 * fs)}" height="30" fill="url(#${id}rw)" opacity="${glowOn ? .5 : .25}"/>`; }
    else s += `<rect x="${f1(fx - 30 * fs)}" y="${f1(fy + 1)}" width="${f1(60 * fs)}" height="10" fill="#fff" opacity=".16"/>`;
    s += ferry(fx, fy, fs, c, lit > .1); }
  if (!glowOn && !Wx.rain && !Wx.snow) { // a sailboat and gulls
    const w = mix('#ffffff', P.sky[3], .08), X = 318 + (mn * 5) % 120, Y = H + 62 * k; s += `<path d="M${X},${f1(Y)}v-21l10,18zM${X - 1.6},${f1(Y - 1)}v-15l-7,13z" fill="${w}"/><path d="M${X - 10},${f1(Y + 1)}h22l-3,3.4h-16z" fill="${mix('#22343c', R[3], .4)}"/>`;
    let gd = ''; [[640, -112, 1.2], [672, -124, 1], [612, -100, .9], [700, 22, 1]].forEach(([gx2, gy, z]) => { const q = 5 * z; gd += `M${f1(gx2 - q)},${f1(H + gy * k)}q${f1(q / 2)},${f1(-q * .7)} ${f1(q)},0q${f1(q / 2)},${f1(-q * .7)} ${f1(q)},0`; });
    s += `<path d="${gd}" stroke="${P.ink === 'light' ? '#2a1c40' : mix(R[3], R[4], .5)}" stroke-width="1.4" fill="none" stroke-linecap="round" opacity=".8"/>`; }
  if (Wx.marine) { x.add(`<linearGradient id="${id}ml" gradientUnits="userSpaceOnUse" x1="0" y1="${f1(H - 70 * k)}" x2="0" y2="${H + 130}"><stop offset="0" stop-color="${P.haze}" stop-opacity="0"/><stop offset=".3" stop-color="${P.haze}" stop-opacity=".9"/><stop offset=".6" stop-color="${mix(P.haze, R[1], .38)}" stop-opacity=".85"/><stop offset="1" stop-color="${mix(P.haze, R[2], .7)}" stop-opacity=".2"/></linearGradient>`);
    s += `<rect x="0" y="${f1(H - 70 * k)}" width="1024" height="${f1(130 + 70 * k)}" fill="url(#${id}ml)"/>`; }
  else s += fog(H + 20, Wx.fog * .9, 90) + fog(H + 110, Wx.fog * .7, 110);
  return s;
}

function scene(o) {
  o = o || {};
  const phase = PHASE[o.phase] ? o.phase : 'golden', weather = WEATHER[o.weather] ? o.weather : 'clear';
  const P = palette(phase, weather), Wx = P.wx, id = 'g' + (UID++), seed = o.seed || 7;
  const vista = o.vista === 'sound' ? 'sound' : 'kerry';
  const hz = o.horizon === 'low' || o.horizon === 'mid' ? o.horizon : 'high', hn = typeof o.horizon === 'number' || /^\d+$/.test(o.horizon || '') ? Math.max(120, Math.min(480, +o.horizon)) : null;
  const H = hn != null ? hn : hz === 'low' ? 446 : hz === 'mid' ? 404 : 372;   // horizon may also be a pixel y (waterline)
  const k0 = hn != null ? Math.max(.6, Math.min(1, 1 - (H - 372) * .004)) : hz === 'low' ? .7 : hz === 'mid' ? .88 : 1;
  const k = o.scale != null && +o.scale > 0 ? Math.max(.4, Math.min(1.2, +o.scale)) : k0, sc = hn != null ? Math.min(1, k0 * .95) : hz === 'low' ? .62 : hz === 'mid' ? .9 : 1;
  const [hr, mn] = parseTime(o, P), R = P.ridge, over = Wx.cover > .5, glowOn = !!P.glowOn, night = phase === 'night';
  const late = night && (hr >= 23 || hr < 5);
  const lit = Math.min(.8, (late ? .2 : P.lit) + (!glowOn && over ? .13 : 0) + (Wx.marine && !glowOn ? .05 : 0));
  const AX = 640, G = inner => `<g transform="translate(${f1(AX * (1 - k))},${H}) scale(${k})">${inner}</g>`;
  const px = x => AX + (x - AX) * k, py = y => H + y * k;
  const snowy = !!Wx.snow, white = glowOn ? '#aeb6e6' : mix('#ffffff', P.sky[3], .1);
  const frost = c => snowy ? mix(c, glowOn ? '#8f98cc' : '#eef2f6', glowOn ? .2 : .3) : c;
  const C = { far: mix(R[0], P.sky[4], .42), hill: mix(R[0], R[1], .5), back: R[1], front: mix(R[1], R[2], .78), needle: mix(R[2], R[3], .5),
    midA: frost(mix(R[2], R[3], .5)), midA2: frost(mix(R[2], R[3], .8)), midB: frost(mix(R[3], R[4], .25)), fg: R[4], water0: mix(P.sky[3], R[2], over ? .5 : .6), water1: mix(R[2], R[3], .66) };
  const warm = ['#ffd98a', '#fff0c8', '#a9d4ff'];
  const fa = P.face[1] * (1 - .85 * Wx.cover), fc = P.face[0];

  let d = `<defs>
   <linearGradient id="${id}s" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="${H + 24}">${P.sky.map((c, i) => `<stop offset="${[0, .3, .58, .82, 1][i]}" stop-color="${c}"/>`).join('')}</linearGradient>
   <linearGradient id="${id}f" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${P.haze}" stop-opacity="0"/><stop offset=".55" stop-color="${P.haze}" stop-opacity="1"/><stop offset="1" stop-color="${P.haze}" stop-opacity="0"/></linearGradient>
   <linearGradient id="${id}c" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${P.cloud[0]}"/><stop offset="1" stop-color="${P.cloud[1]}"/></linearGradient>
   <linearGradient id="${id}w" gradientUnits="userSpaceOnUse" x1="0" y1="${H}" x2="0" y2="${H + 104 * k}"><stop offset="0" stop-color="${C.water0}"/><stop offset=".5" stop-color="${mix(C.water0, C.water1, .6)}"/><stop offset="1" stop-color="${C.water1}"/></linearGradient>
   <linearGradient id="${id}t0" gradientUnits="userSpaceOnUse" x1="0" y1="-150" x2="0" y2="0"><stop offset="0" stop-color="${C.back}"/><stop offset="1" stop-color="${mix(C.back, P.sky[4], over ? .34 : .22)}"/></linearGradient>
   <linearGradient id="${id}t1" gradientUnits="userSpaceOnUse" x1="0" y1="-120" x2="0" y2="0"><stop offset="0" stop-color="${C.front}"/><stop offset="1" stop-color="${mix(C.front, R[3], .3)}"/></linearGradient>
   <linearGradient id="${id}ma" gradientUnits="userSpaceOnUse" x1="0" y1="-30" x2="0" y2="40"><stop offset="0" stop-color="${C.midA}"/><stop offset="1" stop-color="${mix(C.midA, P.sky[4], .12)}"/></linearGradient>
   <linearGradient id="${id}ma2" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="80"><stop offset="0" stop-color="${C.midA2}"/><stop offset="1" stop-color="${mix(C.midA2, P.sky[4], .1)}"/></linearGradient>
   <linearGradient id="${id}mb" gradientUnits="userSpaceOnUse" x1="0" y1="50" x2="0" y2="210"><stop offset="0" stop-color="${C.midB}"/><stop offset="1" stop-color="${mix(R[3], R[4], .8)}"/></linearGradient>
   <linearGradient id="${id}r4" gradientUnits="userSpaceOnUse" x1="0" y1="${H + 76}" x2="0" y2="${H + 270}"><stop offset="0" stop-color="${R[4]}"/><stop offset="1" stop-color="${tune(R[4], 1, l => l * .7)}"/></linearGradient>
   <linearGradient id="${id}v" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${R[4]}" stop-opacity="0"/><stop offset="1" stop-color="${tune(R[4], 1, l => l * .55)}" stop-opacity=".9"/></linearGradient>
   <clipPath id="${id}wc"><rect x="0" y="${H}" width="1024" height="${f1(116 * k)}"/></clipPath>
   <filter id="${id}b" x="-20%" y="-60%" width="140%" height="220%"><feGaussianBlur stdDeviation="13"/></filter>
   <filter id="${id}b2" x="-10%" y="-30%" width="120%" height="160%"><feGaussianBlur stdDeviation="2.4"/></filter>
   <filter id="${id}n" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency=".85" numOctaves="2" seed="4"/><feColorMatrix values="0 0 0 0 .5  0 0 0 0 .5  0 0 0 0 .5  0 0 0 .55 -.16"/></filter>`;
  let s = `<rect width="1024" height="600" fill="url(#${id}s)"/>`;
  const fog = (y, a, h = 90) => a > .01 ? `<rect x="0" y="${f1(y - h * .55)}" width="1024" height="${f1(h)}" fill="url(#${id}f)" opacity="${Math.min(1, a).toFixed(2)}"/>` : '';

  // ── stars
  if (P.stars && !over) { const sr = rng(91); let st = '';
    for (let i = 0; i < 190; i++) { const x = sr() * 1024, y = Math.pow(sr(), 1.35) * (H - 30), rr = .5 + Math.pow(sr(), 3) * 1.5, a = .3 + sr() * .7; if (P.stars < 1 && y > (H - 30) * .55) continue; st += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${rr.toFixed(2)}" opacity="${a.toFixed(2)}"/>`; }
    if (P.stars === 1) [[148, 78], [392, 44], [960, 210], [70, 236], [702, 62]].forEach(([x, y]) => st += `<path d="M${x},${y - 7}L${x + 1.3},${y - 1.3}L${x + 7},${y}L${x + 1.3},${y + 1.3}L${x},${y + 7}L${x - 1.3},${y + 1.3}L${x - 7},${y}L${x - 1.3},${y - 1.3}Z" opacity=".95"/>`);
    s += `<g fill="#f4f1ff" opacity="${P.stars}">${st}</g>`; }
  // ── sun / moon
  if (P.sun) { const u = vista === 'sound' && P.sunW ? { ...P.sun, ...P.sunW } : P.sun, sy = H + u.dy * (u.dy < -100 ? 1 : k), a = 1 - .8 * Wx.cover;
    d += `<radialGradient id="${id}g" gradientUnits="userSpaceOnUse" cx="${u.x}" cy="${f1(sy)}" r="${u.gr}"><stop offset="0" stop-color="${u.glow}" stop-opacity="${(.95 * a).toFixed(2)}"/><stop offset=".3" stop-color="${u.glow}" stop-opacity="${(.42 * a).toFixed(2)}"/><stop offset="1" stop-color="${u.glow}" stop-opacity="0"/></radialGradient>`;
    s += `<rect width="1024" height="600" fill="url(#${id}g)"/>`;
    if (u.x < 1060 && u.dy < 20) { if (!over) s += `<circle cx="${u.x}" cy="${f1(sy)}" r="${u.r * 1.7}" fill="${u.c}" opacity=".2" filter="url(#${id}b2)"/><circle cx="${u.x}" cy="${f1(sy)}" r="${u.r}" fill="${u.c}"/>`;
      else s += `<circle cx="${u.x}" cy="${f1(sy)}" r="${u.r * 1.1}" fill="#fff" opacity=".5" filter="url(#${id}b)"/>`; } }
  if (P.moon) { const m = P.moon, a = 1 - .75 * Wx.cover;
    d += `<radialGradient id="${id}g" gradientUnits="userSpaceOnUse" cx="${m.x}" cy="${m.y}" r="230"><stop offset="0" stop-color="#aab6ff" stop-opacity="${.5 * a}"/><stop offset="1" stop-color="#aab6ff" stop-opacity="0"/></radialGradient>
      <mask id="${id}m"><rect width="1024" height="600" fill="#fff"/><circle cx="${m.x - 13}" cy="${m.y - 7}" r="${m.r * .92}" fill="#000"/></mask>`;
    s += `<rect width="1024" height="600" fill="url(#${id}g)"/>` + (over ? `<circle cx="${m.x}" cy="${m.y}" r="${m.r * 1.5}" fill="#dfe4ff" opacity=".2" filter="url(#${id}b)"/>` : `<circle cx="${m.x}" cy="${m.y}" r="${m.r}" fill="#fdf6dc" opacity="${a}" mask="url(#${id}m)"/><circle cx="${m.x}" cy="${m.y}" r="${m.r}" fill="#fdf6dc" opacity="${.07 * a}"/>`); }
  // ── conversation state (legacy layer; the moment screen draws its own presence)
  if (o.state) {
    const cy = x => H - 92 + 30 * Math.sin(x * .0062 + 1.1) + 16 * Math.sin(x * .017 + .4);
    const col = o.state === 'speaking' ? ['#ffb86a', '#ffe9a8'] : ['#8a6cff', '#6dffc9'];
    if (o.state === 'thinking') {
      const pts = [[318, 236], [392, 196], [468, 222], [540, 178], [622, 204], [694, 164], [742, 226]];
      s += `<polyline points="${pts.map(p => p.join(',')).join(' ')}" fill="none" stroke="#f6cf85" stroke-width="1.6" opacity=".75"/>` + pts.map((p, i) => `<circle cx="${p[0]}" cy="${p[1]}" r="${i % 2 ? 3.4 : 4.6}" fill="#fff3cf"/><circle cx="${p[0]}" cy="${p[1]}" r="11" fill="#f6cf85" opacity=".22"/>`).join('');
    } else if (o.state === 'listening' || o.state === 'speaking') {
      d += `<linearGradient id="${id}a" gradientUnits="userSpaceOnUse" x1="0" y1="${H - 250}" x2="0" y2="${H - 40}"><stop offset="0" stop-color="${col[0]}" stop-opacity="0"/><stop offset=".45" stop-color="${col[0]}" stop-opacity=".5"/><stop offset=".86" stop-color="${col[1]}" stop-opacity=".95"/><stop offset="1" stop-color="${col[1]}" stop-opacity="0"/></linearGradient>`;
      let top = [], bot = [], rays = ''; const ar = rng(5);
      for (let x = -20; x <= 1044; x += 14) { const y = cy(x), hgt = 96 + 70 * Math.pow(Math.sin(x * .011 + 2) * .5 + .5, 1.5) + 30 * ar(); top.push([x, y - hgt]); bot.push([x, y + 12]);
        rays += `<line x1="${x}" y1="${(y - hgt * (.7 + .5 * ar())).toFixed(0)}" x2="${x}" y2="${y.toFixed(0)}" stroke-width="${(2 + 5 * ar()).toFixed(1)}" opacity="${(.12 + .3 * ar()).toFixed(2)}"/>`; }
      const shape = 'M' + top.map(p => p.join(',')).join('L') + 'L' + bot.reverse().map(p => p.join(',')).join('L') + 'Z';
      s += `<path d="${shape}" fill="url(#${id}a)" filter="url(#${id}b)" opacity=".95"/><g stroke="url(#${id}a)" filter="url(#${id}b2)">${rays}</g>`;
    }
  }
  // ── clouds: flat pill bands when fair, one blurred deck plus crisp stratus when grey
  { const cr = rng(seed * 31 + 5), n = Wx.clouds; let c = '', soft = '';
    for (let i = 0; i < n; i++) {
      if (over) { const w = 380 + cr() * 420, h = 30 + cr() * 34, cx = cr() * 1024, cy = 20 + (i / n) * (H - 70) + cr() * 30;
        soft += `<g opacity="${(.32 + cr() * .3).toFixed(2)}" fill="${i % 2 ? P.cloud[1] : P.cloud[0]}">${cloudBand(cx, cy, w, h, cr)}</g>`;
      } else { const w = 150 + cr() * 210, h = 13 + cr() * 17, cx = 300 + ((i * 397 + cr() * 180) % 700), cy = cx < 590 ? H - (44 + cr() * 50) * k : H - (236 - cr() * 150) * k;
        c += `<g opacity="${(P.cloudA * (.55 + .45 * cr())).toFixed(2)}" fill="url(#${id}c)">${cloudBand(cx, cy, w, h, cr)}</g>`; } }
    if (over) s += `<g filter="url(#${id}b)">${soft}</g>`;
    s += c; }
  // ── low cloud catching the city's light
  if (glowOn && over) { d += `<radialGradient id="${id}cg" gradientUnits="userSpaceOnUse" cx="${f1(px(690))}" cy="${f1(py(-70))}" r="${f1(430 * k)}" gradientTransform="translate(0,${f1(py(-70) * .45)}) scale(1,.55)"><stop offset="0" stop-color="#e0a078" stop-opacity=".55"/><stop offset=".5" stop-color="#b87a90" stop-opacity=".22"/><stop offset="1" stop-color="#b87a90" stop-opacity="0"/></radialGradient>`;
    s += `<rect width="1024" height="${H}" fill="url(#${id}cg)"/>`; }
  // ── seaplane on the Lake Union approach (fair daylight only)
  if (!glowOn && !Wx.rain && !Wx.snow && !Wx.marine && phase !== 'dawn') { const x = 600 + (mn * 37) % 190, y = H - (206 + (mn * 13) % 30) * k, c = mix(R[3], R[2], .3);
    s += `<g transform="translate(${x},${f1(y)}) rotate(-4) scale(${(.9 * (.6 + .4 * k)).toFixed(2)})" fill="${c}" stroke="${c}" stroke-linecap="round"><path d="M-13,0Q-12,-3.2 -4,-3.4L9,-2.6Q13,-1.6 13.5,.4Q8,2.4 -2,2.2L-13,1Z" stroke="none"/><path d="M11,-1.5L15.5,-6.5L17,-6.2L15,-.2Z" stroke="none"/><path d="M-10,-4.4H7" stroke-width="2.2"/><path d="M-9,6.4H6" stroke-width="1.7"/><path d="M-5,2L-6,6M2,2L3,6" stroke-width=".8" fill="none"/></g>`; }

  const hood = o.mountain === true ? 1 : o.mountain === false ? 0 : Wx.hood, rim = snowy ? (glowOn ? '#c9d0f4' : '#ffffff') : null;
  if (vista === 'sound') s += sound({ P, Wx, id, seed, H, k, mn, lit, glowOn, over, night, phase, hood, C, rim, fog, add: x => d += x });
  else {
  // ── the city, back to front (city units, see G) ──
  let g = '';
  if (hood > 0) { const M = rainier(0, 0), lightE = P.side < 0, base = mix(R[0], P.sky[3], .35);
    d += `<clipPath id="${id}h"><path d="${M.out}"/></clipPath>
     <linearGradient id="${id}mt" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="176"><stop offset="0" stop-color="${P.snow[0]}"/><stop offset=".5" stop-color="${mix(P.snow[0], P.snow[1], .25)}"/><stop offset=".8" stop-color="${mix(P.snow[1], base, .5)}" stop-opacity=".8"/><stop offset="1" stop-color="${base}" stop-opacity="0"/></linearGradient>
     <linearGradient id="${id}mb2" gradientUnits="userSpaceOnUse" x1="0" y1="112" x2="0" y2="174"><stop offset="0" stop-color="${base}" stop-opacity=".9"/><stop offset="1" stop-color="${base}" stop-opacity="0"/></linearGradient>`;
    g += `<g opacity="${(hood * (night ? .6 : 1)).toFixed(2)}" transform="translate(856,-198) scale(1.1)"><path d="${M.out}" fill="url(#${id}mt)"/><g clip-path="url(#${id}h)">
      <path d="${lightE ? M.west : M.east}" fill="${P.snow[1]}" opacity="${night ? .4 : .55}"/><path d="${M.ribs}" fill="${P.snow[1]}" opacity="${night ? .3 : .5}"/>
      <path d="${M.rock}" fill="${mix(R[1], P.snow[1], .35)}" opacity="${night ? .3 : .42}"/><path d="${M.base}" fill="url(#${id}mb2)"/></g></g>`;
    if (Wx.cap) { const cr = rng(seed + 61); g += `<g fill="url(#${id}c)" opacity="${(P.cloudA * .92).toFixed(2)}">${cloudBand(760, -66, 250, 15, cr)}${cloudBand(972, -98, 190, 12, cr)}</g>`; } }
  // far hills: Cascade foothills, Capitol Hill with its radio masts (left), West Seattle (right)
  g += `<path d="M-700,6V-18C-300,-30 60,-30 300,-22S620,-9 760,-11S900,-15 1000,-18S1400,-17 1800,-13V6Z" fill="${C.far}"/>`;
  g += `<path d="M-700,6V-34C-380,-46 -40,-48 140,-42S390,-24 520,-13L610,-5V6Z" fill="${C.hill}"/><path d="M118,-41l2,-46l2,46zM150,-40l2.2,-60l2.2,60zM196,-37l1.8,-40l1.8,40z" fill="${C.hill}"/>`;
  g += `<path d="M872,4V-3C910,-14 962,-21 1040,-23S1320,-27 1800,-20V6Z" fill="${mix(C.hill, C.back, .35)}"/>`;
  if (glowOn || phase === 'dawn') g += `<g fill="#ff5a4a"><circle cx="120" cy="-87" r="1.5"/><circle cx="152.2" cy="-100" r="1.6"/><circle cx="152.2" cy="-72" r="1.1"/><circle cx="197.8" cy="-77" r="1.4"/></g>`;
  s += G(g) + fog(py(-22), Wx.fog * 1.0, 120 * k); g = '';
  // back row, stadium roofs, Harbor Island cranes
  const B = towerRow(TOWERS.b, P.side, lit * .8, seed * 7 + 3), F = towerRow(TOWERS.f, P.side, lit, seed * 7 + 4);
  const crane = x => `M${x},0V-25M${x + 9},0V-25M${x - 7},-25H${x + 13}M${x + 9},-25L${x + 17},-47M${x + 4.5},-25V-35L${x + 17},-47M${x + 4.5},-35L${x - 7},-25`;
  const craneC = glowOn ? mix(C.back, '#c06a48', .25) : mix(C.back, '#d2643a', over ? .3 : .5);
  g += `<path d="${B.d}" fill="url(#${id}t0)"/><path d="${B.face}" fill="${fc}" opacity="${(fa * .7).toFixed(2)}"/><path d="${B.lines}" stroke="${R[3]}" stroke-width=".8" opacity=".13" fill="none"/>`;
  g += `<path d="M812,1V-9H846V1ZM848,1V-13Q860,-21 873,-13V1Z" fill="url(#${id}t0)"/><path d="M813,-9Q829,-27 845,-9M817,-9Q829,-21 841,-9" stroke="${C.back}" stroke-width="1.5" fill="none"/>`;
  g += `<path d="${[866, 882, 902, 918].map(crane).join('')}" stroke="${craneC}" stroke-width="1.7" fill="none" stroke-linejoin="round"/>`;
  s += G(g) + fog(py(-30), Wx.fog * .9, 150 * k); g = '';
  // front row, great wheel, waterfront
  g += `<path d="${F.d}" fill="url(#${id}t1)"/><path d="${F.face}" fill="${fc}" opacity="${fa.toFixed(2)}"/><path d="${F.lines}" stroke="${R[4]}" stroke-width=".8" opacity=".14" fill="none"/>`;
  g += `<path d="M548,1V-3H884V1Z" fill="${mix(C.front, R[3], .4)}"/>`;
  const wheelC = glowOn ? '#cfd4ff' : mix(C.front, R[3], .3);
  g += `<g stroke="${wheelC}" fill="none" stroke-width="${glowOn ? 1.5 : 1.3}"><circle cx="832" cy="-15" r="11"/><path d="M832,-26V-4M821,-15H843M824.2,-22.8L839.8,-7.2M824.2,-7.2L839.8,-22.8" stroke-width=".6"/><path d="M832,-15L827,-1M832,-15L837,-1" stroke-width="1.2"/></g>`;
  if (lit > 0) { const a = glowOn ? 1 : phase === 'golden' ? .8 : .7;
    g += [0, 1, 2].map(i => `<path d="${boxes(B.w[i], 1.9, 2.5)}" fill="${warm[i]}" opacity="${(a * .8).toFixed(2)}"/><path d="${boxes(F.w[i], 2.1, 2.7)}" fill="${warm[i]}" opacity="${a}"/>`).join('');
    if (glowOn) g += `<g fill="#ff5a4a"><circle cx="716" cy="-161.5" r="1.5"/><circle cx="663.6" cy="-124" r="1.3"/><circle cx="505.3" cy="-91" r="1.2"/><circle cx="426.6" cy="-94" r="1.2"/></g>`; }
  s += `<g id="${id}city">${G(g)}</g>`; g = '';
  // low cloud swallowing the tops
  if (Wx.veil) { d += `<linearGradient id="${id}vl" gradientUnits="userSpaceOnUse" x1="0" y1="${f1(py(-370))}" x2="0" y2="${f1(py(-84))}"><stop offset="0" stop-color="${P.sky[2]}" stop-opacity="0"/><stop offset=".42" stop-color="${P.sky[2]}" stop-opacity="${Wx.veil}"/><stop offset=".68" stop-color="${P.sky[3]}" stop-opacity="${(Wx.veil * .8).toFixed(2)}"/><stop offset="1" stop-color="${P.sky[3]}" stop-opacity="0"/></linearGradient>`;
    s += `<rect x="0" y="${f1(py(-370))}" width="1024" height="${f1(290 * k)}" fill="url(#${id}vl)"/>`; }

  // ── Elliott Bay ──
  { let w = `<rect x="0" y="${H}" width="1024" height="${f1(116 * k)}" fill="url(#${id}w)"/>`;
    const rf = over || glowOn ? .5 : .4;
    w += `<use href="#${id}city" transform="translate(0,${f1(H * (1 + rf))}) scale(1,${-rf})" opacity="${Wx.rain ? .42 : glowOn ? .5 : .3}"/>`;
    const wr = rng(seed + 300); let sh = '', st = '';
    for (let i = 0; i < 34; i++) { const y = 5 + Math.pow(wr(), 1.4) * 88, x = 580 + wr() * 460, l = (8 + wr() * 30) * (.5 + y / 60); sh += `M${f1(px(x))},${f1(py(y))}h${f1(l * k)}`; }
    w += `<path d="${sh}" stroke="${mix(P.sky[4], '#ffffff', .3)}" stroke-width="1" opacity="${glowOn ? .12 : over ? .3 : .34}" fill="none" stroke-linecap="round"/>`;
    if (lit > .1) { d += `<linearGradient id="${id}rw" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffd98a" stop-opacity=".9"/><stop offset="1" stop-color="#ffd98a" stop-opacity="0"/></linearGradient><linearGradient id="${id}rc" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#bcd8ff" stop-opacity=".8"/><stop offset="1" stop-color="#bcd8ff" stop-opacity="0"/></linearGradient>`;
      let a = '', b = ''; const nn = Math.round(46 * Math.min(1, lit * 1.8));
      for (let i = 0; i < nn; i++) { const x = px(560 + wr() * 330), l = (10 + wr() * 44) * k, r = `<rect x="${f1(x)}" y="${H + 1}" width="${f1(1.4 + wr() * 2.2)}" height="${f1(l)}" opacity="${(.3 + wr() * .6).toFixed(2)}"/>`; wr() < .72 ? a += r : b += r; }
      st = `<g fill="url(#${id}rw)" opacity="${glowOn ? .9 : .5}">${a}</g><g fill="url(#${id}rc)" opacity="${glowOn ? .8 : .4}">${b}</g>`; w += st; }
    if (P.moon && !over) { let gl = ''; for (let i = 0; i < 16; i++) { const y = 8 + Math.pow(wr(), .8) * 84, l = (5 + wr() * 16) * (.5 + y / 70); gl += `M${f1(px(900) + (wr() - .5) * (16 + y * .5) - l / 2)},${f1(py(y))}h${f1(l)}`; }
      w += `<path d="${gl}" stroke="#dfe4ff" stroke-width="1.2" opacity=".42" fill="none" stroke-linecap="round"/>`; }
    if (phase === 'golden' && !over) { let gl = ''; for (let i = 0; i < 22; i++) { const x = 820 + wr() * 220, y = 6 + wr() * 68, l = 14 + wr() * 40; gl += `M${f1(px(x))},${f1(py(y))}h${f1(l)}`; }
      w += `<path d="${gl}" stroke="#ffd9a0" stroke-width="1.4" opacity=".55" fill="none" stroke-linecap="round"/>`; }
    // ferry: Seattle ⇄ Bainbridge, position from the minute
    { const t = (mn % 30) / 30, back = Math.floor(mn / 30) % 2, fx = px(back ? 1004 - 310 * t : 694 + 310 * t), fy = py(46), fs = 1.16 * (.55 + .45 * k);
      const fcx = glowOn ? { hull: '#8f97c8', white: '#aab2de', green: '#2f6a58', dark: '#141a3c' } : { hull: mix('#ffffff', P.sky[3], .14), white: mix('#ffffff', P.sky[3], .06), green: mix('#1f7a55', R[2], over ? .35 : .15), dark: mix('#1b2b33', R[3], .4) };
      w += `<path d="M${f1(fx + (back ? 34 : -34) * fs)},${f1(fy + .5)}h${f1((back ? 46 : -46) * fs)}" stroke="#fff" stroke-width="1.6" opacity="${glowOn ? .22 : .5}" stroke-linecap="round"/>`;
      if (lit > .1) w += `<rect x="${f1(fx - 26 * fs)}" y="${f1(fy + 1)}" width="${f1(52 * fs)}" height="${f1(22 * k)}" fill="url(#${id}rw)" opacity="${glowOn ? .5 : .25}"/>`;
      else w += `<rect x="${f1(fx - 30 * fs)}" y="${f1(fy + 1)}" width="${f1(60 * fs)}" height="${f1(9 * k)}" fill="#fff" opacity=".16"/>`;
      w += ferry(fx, fy, fs, fcx, lit > .1); }
    // gulls over the bay
    if (!glowOn && !Wx.snow) { const gc = P.ink === 'dark' && !over ? '#ffffff' : '#fff6e6'; let gd = '';
      [[742, 22, 1], [768, 13, .8], [716, 34, .7]].forEach(([x, y, z]) => { const X = px(x + (mn * 3) % 40), Y = py(y), q = 5 * z * (.6 + .4 * k); gd += `M${f1(X - q)},${f1(Y)}q${f1(q / 2)},${f1(-q * .7)} ${f1(q)},0q${f1(q / 2)},${f1(-q * .7)} ${f1(q)},0`; });
      w += `<path d="${gd}" stroke="${gc}" stroke-width="1.3" fill="none" stroke-linecap="round" opacity=".9"/>`; }
    s += `<g clip-path="url(#${id}wc)">${w}</g>` + fog(py(4), Wx.fog * 1.1, 70 * k); }

  // ── the Needle ──
  { const N = needle(545, 34, 1.353), top = glowOn ? '#eef0ff' : mix(C.needle, P.sky[4], over ? .12 : .2), lg = glowOn ? `url(#${id}nl)` : C.needle;
    if (glowOn) { d += `<linearGradient id="${id}nl" gradientUnits="userSpaceOnUse" x1="0" y1="-190" x2="0" y2="20"><stop offset="0" stop-color="#d9defa"/><stop offset=".45" stop-color="${mix('#aab2e6', C.needle, .35)}"/><stop offset="1" stop-color="${C.needle}"/></linearGradient>
       <radialGradient id="${id}ng" gradientUnits="userSpaceOnUse" cx="${N.deck[0]}" cy="${f1(N.deck[1])}" r="62"><stop offset="0" stop-color="#fff2c8" stop-opacity="${over ? .6 : .46}"/><stop offset=".4" stop-color="#ffe0a0" stop-opacity=".16"/><stop offset="1" stop-color="#ffe0a0" stop-opacity="0"/></radialGradient>`;
      g += `<circle cx="${N.deck[0]}" cy="${f1(N.deck[1])}" r="62" fill="url(#${id}ng)"/>`; }
    g += `<path d="${N.legs}" fill="${lg}"/><path d="${N.house}" fill="${top}"/><path d="${N.bowl}" fill="${glowOn ? '#8f98d0' : R[3]}" opacity="${glowOn ? .5 : .3}"/>`;
    g += `<path d="${N.roof(P.side)}" fill="${glowOn ? '#ffffff' : fc}" opacity="${glowOn ? .35 : (fa * .9).toFixed(2)}"/>`;
    g += `<path d="${N.band}" fill="${lit > .1 ? '#ffd98a' : R[4]}" opacity="${lit > .1 ? 1 : .42}"/><path d="${N.halo}" fill="${glowOn ? '#ffffff' : mix(top, '#ffffff', .4)}" opacity="${glowOn ? .95 : .6}"/>`;
    if (glowOn || phase === 'dawn' || over) g += `<circle cx="${N.tip[0]}" cy="${f1(N.tip[1] - 1)}" r="2" fill="#ff5a4a"/><circle cx="${N.tip[0]}" cy="${f1(N.tip[1] - 1)}" r="6" fill="#ff5a4a" opacity=".25"/>`;
    s += G(g); g = '';
    if (Wx.veil) s += `<rect x="${f1(px(545) - 60)}" y="${f1(py(-250))}" width="120" height="${f1(60 * k)}" fill="url(#${id}vl)" opacity=".55"/>`; }

  // ── Belltown rooftops, then the Queen Anne slope ──
  const yB = x => 56 + 42 * sstep((x - 250) / 640) + 5 * Math.sin(x * .013 + 1);
  const yA = x => { const t = -9 + 2.5 * Math.sin(x * .021); return t + (yB(x) + 10 - t) * sstep((x - 528) / 118); };
  const yA2 = x => { const t = 17 + 3 * Math.sin(x * .017 + 2); return t + (yB(x) + 10 - t) * sstep((x - 566) / 96); };
  const A = roofline(-700, 700, yA, rng(seed * 13 + 1), lit * .7, 26), A2 = roofline(-700, 700, yA2, rng(seed * 13 + 5), lit * .7, 17), Bq = slope(-700, 1800, yB, rng(seed * 13 + 2), glowOn ? lit * .8 : lit * .5);
  if (rim) g += `<path d="${A.d}" fill="${rim}" transform="translate(0,-1.8)" opacity=".85"/>`;
  g += `<path d="${A.d}" fill="url(#${id}ma)"/>`;
  if (rim) g += `<path d="${A2.d}" fill="${rim}" transform="translate(0,-1.8)" opacity=".85"/>`;
  g += `<path d="${A2.d}" fill="url(#${id}ma2)"/>`;
  if (lit > 0) { g += `<path d="${boxes(A.w, 2, 2.4)}${boxes(A2.w, 2.2, 2.6)}" fill="${warm[0]}" opacity="${glowOn ? .9 : .6}"/>`;
    if (glowOn) { const lr = rng(seed + 77); let dots = ''; for (let i = 0; i < 80; i++) { const x = -200 + lr() * 800, y = yA(x) + 8 + lr() * Math.max(6, yB(x) - yA(x) - 12); dots += `<circle cx="${f1(x)}" cy="${f1(y)}" r="${(.7 + lr() * .7).toFixed(1)}" opacity="${(.4 + lr() * .6).toFixed(2)}"/>`; }
      g += `<g fill="#ffcf8a">${dots}</g>`; } }
  s += G(g); g = '';
  if (Wx.marine) { // the marine layer: everything low drowns, tops poke through
    d += `<linearGradient id="${id}ml" gradientUnits="userSpaceOnUse" x1="0" y1="${f1(py(-124))}" x2="0" y2="${f1(py(120))}"><stop offset="0" stop-color="${P.haze}" stop-opacity="0"/><stop offset=".24" stop-color="${P.haze}" stop-opacity=".5"/><stop offset=".42" stop-color="${P.haze}" stop-opacity=".92"/><stop offset=".62" stop-color="${mix(P.haze, R[1], .38)}" stop-opacity=".92"/><stop offset=".82" stop-color="${mix(P.haze, R[2], .6)}" stop-opacity=".78"/><stop offset="1" stop-color="${mix(P.haze, R[2], .7)}" stop-opacity=".3"/></linearGradient>`;
    s += `<rect x="0" y="${f1(py(-124))}" width="1024" height="${f1(244 * k)}" fill="url(#${id}ml)"/>`;
    const fr = rng(seed + 9); let wisps = ''; for (let i = 0; i < 4; i++) wisps += cloudBand(120 + fr() * 860, py(-70 + fr() * 30), 260 + fr() * 300, 10 + fr() * 7, fr);
    s += `<g fill="${P.haze}" opacity=".42">${wisps}</g>`;
  } else s += fog(py(34), Wx.fog * .9, 100 * k);
  if (rim) g += `<path d="${Bq.d}" fill="${rim}" transform="translate(-.6,-2.2)" opacity=".9"/>`;
  g += `<path d="${Bq.d}" fill="url(#${id}mb)"/>`;
  if (lit > 0) g += `<path d="${boxes(Bq.w, 2.6, 3)}" fill="${warm[0]}" opacity="${glowOn ? .95 : .7}"/>`;
  s += G(g); g = '';
  s += fog(py(96), (Wx.marine ? .5 : Wx.fog * .8), 110 * k);
  }

  // ── foreground: Kerry Park firs framing the view ──
  { const L4 = ridgePts(H + 142, 20, seed + 5), tr = rng(seed + 44); let big = '';
    [[-6, 176], [32, 214], [76, 142], [110, 182], [152, 104], [188, 72], [946, 86], [982, 150], [1016, 214]].forEach(([x, h]) => big += fir(x, yAt(L4, x) + 6, h * sc * (.9 + .2 * tr()), tr));
    const fd = `${poly(L4)}${treeline(L4, seed + 22, 20 * sc, 46 * sc, 16, 40)}${big}`;
    if (rim) s += `<path d="${fd}" fill="${rim}" transform="translate(-1.2,-2.6)" opacity=".8"/>`;
    s += `<path d="${fd}" fill="url(#${id}r4)"/>`; }
  s += `<rect y="${H + 100}" width="1024" height="${500 - H}" fill="url(#${id}v)"/>`;
  // ── precipitation
  if (Wx.rain) { const rr = rng(3); let l = ''; for (let i = 0; i < 170; i++) { const x = rr() * 1100 - 20, y = rr() * 600, len = 16 + rr() * 26; l += `<line x1="${x.toFixed(0)}" y1="${y.toFixed(0)}" x2="${(x - len * .2).toFixed(0)}" y2="${(y + len).toFixed(0)}" opacity="${(.1 + rr() * .26).toFixed(2)}"/>`; }
    s += `<g stroke="#fff" stroke-width="1.1" stroke-linecap="round">${l}</g>`; }
  if (Wx.snow) { const rr = rng(8); let l = ''; for (let i = 0; i < 210; i++) l += `<circle cx="${(rr() * 1024).toFixed(0)}" cy="${(rr() * 600).toFixed(0)}" r="${(.8 + Math.pow(rr(), 2.4) * 2.8).toFixed(1)}" opacity="${(.3 + rr() * .6).toFixed(2)}"/>`;
    s += `<g fill="#fff">${l}</g>`; }
  s += `<rect width="1024" height="600" filter="url(#${id}n)" opacity=".5" style="mix-blend-mode:soft-light"/>`;
  return { svg: `<svg class="scene" viewBox="0 0 1024 600" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">${d}</defs>${s}</svg>`, P };
}
