/* Living Sky — the original block vocabulary (docs/display-system.md), drawn in the new look.

   Every display written for the PIL renderer keeps working: containers map onto flexbox, content blocks
   take the atmosphere tokens, `card` becomes a slab-tinted surface. Params and defaults: displays/blocks.py.

   Layout rules carried over from the old engine:
   - row: children at natural width; `align: spread` = equal cells; a spacer with no size is flexible.
   - column / card: children stacked, stretched unless `item_align`; `grow: true` absorbs leftover height.
   - columns: cells by ratio, every cell full height.
   - gap is applied between real children only, so a spacer never doubles it (`--g`, see base.css).

   Text placed straight on the scene is sky ink; after mount each element's ink is checked against what the
   scene actually draws behind it (zoneInk), because a legacy layout does not know where the skyline is. */
(function () {
  'use strict';
  const esc = BB.esc;
  const has = v => v != null && v !== '';
  const str = v => (v == null ? '' : String(v));
  const num = (v, d) => { const n = typeof v === 'number' ? v : parseFloat(v); return isFinite(n) ? n : d; };
  const px = v => Math.max(0, Math.round(num(v, 0))) + 'px';
  const css = o => Object.entries(o).filter(([, v]) => has(v)).map(([k, v]) => k + ':' + v).join(';');
  const truthy = v => v === true || v === 'true' || v === 1;
  const falsy = v => v === false || v === 'false' || v === 0;

  /* ── tokens ─────────────────────────────────────────────────── */
  const ROOT_PAD = '34px 44px';
  const GROUND_Y = 404;                     // at horizon "high": below this the scene is mid-to-dark
  const TOKENS = {
    default: 'var(--text)', text: 'var(--text)', muted: 'var(--muted)', dim: 'var(--bb-dim)',
    accent: 'var(--accent)', secondary: 'var(--bb-sec)', success: 'var(--bb-ok)', warning: 'var(--bb-warn)',
    error: 'var(--bb-err)', accent_soft: 'var(--chip)', surface: 'var(--bb-line)', surface_alt: 'var(--bb-line)',
    background: 'var(--bb-line)',
  };
  const HEX = /^#([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$/i;
  /* Colour token or hex → a CSS colour. Anything else falls back; nothing from data reaches a style raw. */
  function color(tok, fallback) {
    if (!has(tok)) return fallback || '';
    tok = String(tok).trim();
    if (HEX.test(tok)) return tok;
    return TOKENS[tok.toLowerCase()] || fallback || '';
  }
  const STRUCT = { surface: 1, surface_alt: 1, background: 1, dim: 1, default: 1 };
  const lineColor = tok => (has(tok) && !STRUCT[String(tok).toLowerCase()] ? color(tok, '') : '');

  function pad(p) {   // 24 | [v, h] | [t, r, b, l]
    if (Array.isArray(p)) return p.length === 2 || p.length === 4 ? p.map(px).join(' ') : p.length ? px(p[0]) : '';
    return has(p) ? px(p) : '';
  }
  const isRoot = (b, ctx) => !!(ctx && ctx.state && ctx.state.layout === b);
  /* A root with no padding gets the default edge, unless its children bring their own (the old engine's idiom). */
  const padOf = (b, ctx, dflt) => (b.padding != null ? pad(b.padding)
    : isRoot(b, ctx) && !(b.children || []).some(c => c && c.padding != null) ? ROOT_PAD : dflt || '');
  const gapOf = (b, dflt) => { const g = px(b.gap != null ? b.gap : dflt || 0); return { '--g': g, gap: g }; };
  const sub = (ctx, kind) => ({ ...ctx, bbIn: kind });
  const kids = (b, ctx, kind) => (b.children || []).map(c => BB.render(c, sub(ctx, kind))).join('');

  function box(b, cls, style, inner, attrs) {
    return `<div class="bb ${cls}${truthy(b.grow) ? ' bb-grow' : ''}"${style ? ` style="${esc(style)}"` : ''}${attrs || ''}>${inner}</div>`;
  }
  /* The spec's root block fills the screen and carries the sky ink. */
  const def = (type, fn) => BB.define(type, (b, ctx) => {
    const html = fn(b, ctx || BB.ctx || {});
    return isRoot(b, ctx) ? `<div class="bb-root sky-ink">${html}</div>` : html;
  });

  /* ── containers ─────────────────────────────────────────────── */
  const MAIN = { start: 'flex-start', center: 'safe center', end: 'flex-end' };
  const CROSS = { start: 'flex-start', top: 'flex-start', center: 'center', end: 'flex-end', bottom: 'flex-end' };

  def('row', (b, ctx) => {
    const spread = b.align === 'spread';
    return box(b, 'bb-row' + (spread ? ' bb-spread' : ''), css({
      ...gapOf(b), padding: padOf(b, ctx), 'justify-content': spread ? '' : MAIN[b.align],
      'align-items': b.valign && b.valign !== 'center' ? CROSS[b.valign] : '',
    }), kids(b, ctx, 'row'));
  });

  /* A centred root lifts its content into the quiet sky (base.css .bb-rise) instead of onto the skyline. */
  const flexes = c => c && (truthy(c.grow) || (c.type === 'spacer' && !(num(c.size, 0) > 0)));
  const flexSpacer = c => c && c.type === 'spacer' && !(num(c.size, 0) > 0);
  function column(b, ctx) {
    const c = b.children || [], root = isRoot(b, ctx), sp = root ? c.filter(flexSpacer) : [];
    const rise = root && b.align === 'center' && !sp.length && !c.some(k => k && truthy(k.grow));
    const inner = sp.length > 1 ? { ...sub(ctx, 'column'), bbLift: sp[0] } : sub(ctx, 'column');   // spacer-centred: lift the same way
    return box(b, 'bb-column' + (rise ? ' bb-rise' : ''), css({
      ...gapOf(b), padding: padOf(b, ctx), 'justify-content': MAIN[b.align],
      'align-items': b.item_align && b.item_align !== 'stretch' ? CROSS[b.item_align] : '',
    }), c.map(k => BB.render(k, inner)).join(''));
  }
  def('column', column);
  def('stack', column);

  def('columns', (b, ctx) => {
    const r = Array.isArray(b.ratios) && b.ratios.length ? b.ratios : [1, 1];
    const cells = (b.children || []).slice(0, r.length).map((c, i) =>
      `<div class="bb-col" style="flex:${Math.max(0, num(r[i], 1))} 1 0">${BB.render(c, sub(ctx, 'col'))}</div>`).join('');
    return box(b, 'bb-columns', css({ ...gapOf(b), padding: padOf(b, ctx) }), cells);
  });

  /* Relative luminance of a hex colour, 0..1 — picks the ink for a card given a literal colour. */
  function lum(hex) {
    let h = hex.slice(1); if (h.length < 6) h = [...h].map(c => c + c).join('');
    const [r, g, bl] = [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16) / 255).map(v => (v <= .03928 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4));
    return .2126 * r + .7152 * g + .0722 * bl;
  }
  function cardSurface(tok) {
    if (!has(tok)) return ['', ''];
    const t = String(tok).trim().toLowerCase();
    if (t === 'surface' || t === 'background' || t === 'default') return ['', ''];
    if (t === 'surface_alt') return [' bb-card-alt', ''];
    if (HEX.test(t)) return [lum(t) > .4 ? ' bb-card-paper' : ' bb-card-lit', t];
    const c = color(t, '');
    return c ? ['', `color-mix(in srgb, ${c} 24%, var(--slab-solid))`] : ['', ''];
  }
  def('card', (b, ctx) => {
    const [cls, bg] = cardSurface(b.color);
    return box(b, 'bb-card' + cls, css({
      '--g': '8px', padding: padOf(b, ctx, ''), 'border-radius': b.radius != null ? px(b.radius) : '',
      'justify-content': MAIN[b.align], background: bg,
    }), kids(b, ctx, 'card'));
  });

  def('spacer', (b, ctx) => (num(b.size, 0) > 0
    ? `<i class="bb bb-spacer" style="flex:0 0 ${px(b.size)}"></i>` : `<i class="bb bb-spacer bb-flex${ctx.bbLift === b ? ' bb-lift' : ''}"></i>`));

  def('divider', b => {
    const v = b.orientation === 'vertical', t = Math.max(1, Math.min(12, num(b.thickness, 1)));
    return `<i class="bb bb-divider${v ? ' bb-v' : ''}" style="${esc(css({ [v ? 'width' : 'height']: t + 'px', background: lineColor(b.color) }))}"></i>`;
  });

  /* Arrives expanded: one child per item. In a row the items join the row; anywhere else they stack. */
  def('repeat', (b, ctx) => {
    if (!(b.children || []).length) return '';
    return ctx.bbIn === 'row' ? `<div class="bb-flow">${kids(b, ctx, 'row')}</div>`
      : box(b, 'bb-column bb-repeat', '--g:8px;gap:8px', kids(b, ctx, 'column'));
  });

  /* ── text ───────────────────────────────────────────────────── */
  const SIZES = { title: 1, heading: 1, subtitle: 1, body: 1, caption: 1, small: 1 };
  const SANS_W = { normal: 400, regular: 400, medium: 500, semibold: 600, bold: 700 };
  const SERIF_W = { normal: 320, regular: 320, medium: 350, semibold: 380, bold: 400 };
  const ALIGN = { center: 'center', right: 'right' };

  def('text', (b, ctx) => {
    const content = str(b.content);
    if (!content.trim()) return '';
    const st = {}; let cls = 'bb-text', serif = false;
    if (typeof b.size === 'number' || /^\d+(\.\d+)?$/.test(str(b.size))) {
      const fs = Math.max(16, Math.min(220, num(b.size, 20)));
      st['font-size'] = fs + 'px'; serif = fs >= 34; cls += ' t-px';
      if (fs >= 34) st['line-height'] = '1.04';
    } else {
      const s = SIZES[b.size] ? b.size : 'body';
      cls += ' t-' + s; serif = s === 'title' || s === 'heading';
      if ((s === 'small' || s === 'caption') && /^[A-Z][A-Z &·\/-]{2,27}$/.test(content.trim())) cls += ' t-label';   // "NEXT UP": a label
    }
    if (serif) cls += ' d';
    const w = str(b.weight).toLowerCase();
    if (has(b.weight)) st['font-weight'] = (serif ? SERIF_W : SANS_W)[w] || (/^[1-9]00$/.test(w) ? (serif ? Math.min(+w, 600) : w) : '');
    st.color = color(b.color, '');
    st['text-align'] = ALIGN[b.align] || '';
    const ml = Math.floor(num(b.max_lines, 0));
    if (ctx.bbIn === 'row' && !(ml > 0) && (content.length <= 28 || num(b.min_width, 0) > 0)) cls += ' bb-keep';   // short words in a row never wrap
    if (ml === 1) cls += ' bb-clip1';
    else if (ml > 1) { cls += ' bb-clip'; st['-webkit-line-clamp'] = ml; }
    if (num(b.min_width, 0) > 0) st['min-width'] = px(b.min_width);
    return `<div class="bb ${cls}"${css(st) ? ` style="${esc(css(st))}"` : ''}>${esc(content)}</div>`;
  });

  /* ── icons ──────────────────────────────────────────────────── */
  const ISZ = { sm: 22, md: 28, lg: 38, xl: 60 };
  function icon(name, size, col, extra) {
    if (!has(name)) return '';
    const s = ISZ[size] ? size : 'md';
    return `<svg class="ico bb bb-icon bb-i-${s}${extra || ''}"${col ? ` style="color:${esc(col)}"` : ''}><use href="#i-${icoName(name)}"/></svg>`;
  }
  def('icon', b => icon(b.name, b.size, color(b.color, '')));

  /* No emoji font is bundled. A name the icon set knows is drawn as its icon; a literal emoji character is
     passed through; any other name draws nothing rather than a word. */
  def('emoji', b => {
    const n = str(b.name).trim(); if (!n) return '';
    const key = n.toLowerCase().replace(/_/g, '-');
    if (ICONS[key] || ICON_ALIAS[key]) return icon(key, b.size, 'var(--accent)');
    return /^[\x20-\x7e]+$/.test(n) ? '' : `<span class="bb bb-emoji bb-i-${ISZ[b.size] ? b.size : 'md'}">${esc(n)}</span>`;
  });

  /* ── metric, badge, list, table, key_value, progress ────────── */
  function splitUnit(v) {   // "3.1 kW" → ["3.1", "kW"]; "72°" and "87%" stay whole
    const m = /^(\D{0,2}?[-+−]?\d[\d.,:]*[°%]?)\s*([A-Za-zµ\/][^\d]{0,11})$/.exec(v);
    return m ? [m[1], m[2]] : [v, ''];
  }
  function changeColor(change, tok) {
    if (has(tok) && tok !== 'auto') return color(tok, 'var(--muted)');
    return /^\s*(\+|\d)/.test(change) ? 'var(--bb-ok)' : /^\s*[-−]/.test(change) ? 'var(--bb-err)' : 'var(--muted)';
  }
  def('metric', b => {
    const value = str(b.value).trim(), label = str(b.label).trim(), change = str(b.change).trim();
    if (!value && !label) return '';
    const [n, unit] = splitUnit(value), tint = color(b.color, '');
    const head = label || has(b.icon) ? `<div class="bb-m-label label">${icon(b.icon, 'sm', tint)}${label ? `<span>${esc(label)}</span>` : ''}</div>` : '';
    return box(b, 'bb-metric', '', `${head}<div class="bb-m-row"><span class="bb-m-val d num">${esc(n)}</span>${
      unit ? `<span class="bb-m-unit">${esc(unit)}</span>` : ''}${
      change ? `<span class="bb-m-chg num" style="color:${esc(changeColor(change, b.change_color))}">${esc(change)}</span>` : ''}</div>`);
  });

  def('badge', b => {
    const t = str(b.text).trim(); if (!t) return '';
    return `<span class="bb bb-badge" style="background:${esc(color(b.color, 'var(--accent)'))}">${esc(t)}</span>`;
  });

  const itemText = it => (it != null && typeof it === 'object' ? str(it.text || it.description || it.title || it.name) : str(it));
  def('list', b => {
    let items = Array.isArray(b.items) ? b.items : has(b.items) ? [b.items] : [];
    items = items.map(itemText).filter(s => s.trim());
    if (!items.length) return '';
    const max = Math.floor(num(b.max_items, 0)), shown = max > 0 ? items.slice(0, max) : items, style = str(b.style) || 'bullet';
    const mark = i => (has(b.icon) ? icon(b.icon, 'sm', 'var(--muted)')
      : style === 'number' ? `<span class="bb-l-n num">${i + 1}</span>`
      : style === 'check' ? '<i class="bb-l-check"></i>' : style === 'none' ? '' : '<i class="bb-l-dot"></i>');
    return box(b, 'bb-list', '', shown.map((t, i) => `<div class="bb-l-item">${mark(i)}<span>${esc(t)}</span></div>`).join('') +
      (items.length > shown.length ? `<div class="bb-l-more">+${items.length - shown.length} more</div>` : ''));
  });

  def('table', b => {
    const headers = Array.isArray(b.headers) ? b.headers : [];
    let rows = Array.isArray(b.rows) ? b.rows.filter(Array.isArray) : [];
    const n = headers.length || Math.max(0, ...rows.map(r => r.length));
    if (!n) return '';
    const max = Math.floor(num(b.max_rows, 0)); if (max > 0) rows = rows.slice(0, max);
    const cell = (v, cls) => `<div class="${cls}">${esc(v)}</div>`;
    return box(b, 'bb-table' + (truthy(b.striped) ? ' bb-striped' : ''), `grid-template-columns:repeat(${Math.min(n, 12)},minmax(0,auto))`,
      (headers.length ? headers.slice(0, 12).map(h => cell(h, 'bb-th')).join('') : '') +
      rows.map((r, i) => Array.from({ length: Math.min(n, 12) }, (_, c) => cell(r[c], 'bb-td' + (i % 2 ? ' bb-odd' : ''))).join('')).join(''));
  });

  def('key_value', b => {
    const d = b.data && typeof b.data === 'object' && !Array.isArray(b.data) ? Object.entries(b.data) : [];
    if (!d.length) return '';
    return box(b, 'bb-kv', '', d.map(([k, v]) => `<div class="bb-k">${esc(k)}</div><div class="bb-v num">${esc(v)}</div>`).join(''));
  });

  def('progress', b => {
    let v = num(b.value, 0); if (v > 1 && v <= 100) v /= 100;
    v = Math.max(0, Math.min(1, v));
    const auto = !has(b.color) || b.color === 'auto';
    const c = auto ? (v < .6 ? 'var(--bb-ok)' : v < .85 ? 'var(--bb-warn)' : 'var(--bb-err)') : color(b.color, 'var(--accent)');
    return box(b, 'bb-progress', '', (has(b.label) ? `<div class="bb-p-label">${esc(b.label)}</div>` : '') +
      `<div class="bb-p-track">${v > 0 ? `<i style="width:${(v * 100).toFixed(1)}%;background:${esc(c)}"></i>` : ''}</div>`);
  });

  /* ── image ──────────────────────────────────────────────────── */
  const NOIMG = '<svg class="ico bb-noimg-ico"><use href="#i-image"/></svg>';
  function imageUrl(src) {
    src = str(src).trim();
    if (/^data:image\//i.test(src) || /^file:\/\//i.test(src)) return src;
    if (src.startsWith('/')) return 'file://' + src.split('/').map(encodeURIComponent).join('/');
    return '';                                   // photo:…, url:…, asset:… never resolved → placeholder
  }
  def('image', (b, ctx) => {
    const url = imageUrl(b.source), fit = { contain: 'contain', fill: 'fill' }[b.fit] || 'cover';
    const fill = isRoot(b, ctx) || ctx.bbIn === 'col';
    const r = b.radius != null ? px(b.radius) : isRoot(b, ctx) && url ? '0px' : '';
    return box(b, `bb-image bb-fit-${fit}${fill ? ' bb-fill' : ''}${url ? '' : ' bb-noimg'}`, r ? `--r:${r}` : '',
      (url ? `<img src="${esc(url)}" alt="">` : '') + NOIMG);
  });

  /* ── chart ──────────────────────────────────────────────────── */
  const SERIES_DEFAULT = ['var(--accent)', 'var(--bb-sec)', 'var(--bb-ok)', 'var(--bb-warn)', 'var(--bb-err)'];
  const points = a => (Array.isArray(a) ? a : []).map(v => (v != null && typeof v === 'object' ? num(v.y, NaN) : num(v, NaN))).filter(v => !isNaN(v));
  const f2 = v => (+v).toFixed(2);

  function chart(b) {
    const kind = { bar: 'bar', area: 'area' }[b.chart_type || b.kind || (b.type !== 'chart' ? b.type : '')] || 'line';
    let series = [];
    if (Array.isArray(b.series) && b.series.length) {
      series = b.series.filter(s => s && typeof s === 'object').map((s, i) => ({
        v: points(s.data), c: color(s.color, SERIES_DEFAULT[i % SERIES_DEFAULT.length]), label: str(s.label) }));
    } else series = [{ v: points(b.data), c: color(b.color, 'var(--accent)'), label: '' }];
    series = series.filter(s => s.v.length);
    const h = Math.max(90, Math.min(560, num(b.height, 200)));
    if (!series.length) return box(b, 'bb-chart bb-chart-empty', `height:${h}px`, '<span>No data yet</span>');

    const all = series.flatMap(s => s.v); let lo = Math.min(...all), hi = Math.max(...all);
    if (kind === 'bar') { lo = Math.min(0, lo); hi = Math.max(0, hi); }
    else { const p = (hi - lo || Math.abs(hi) || 1) * .1; lo -= p; hi += p; }
    const span = hi - lo || 1, Y = v => 100 - (v - lo) / span * 100;
    const fo = Math.max(0, Math.min(1, num(b.fill_opacity, .15)));
    let plot = '', dots = '';

    if (!falsy(b.show_grid)) plot += [25, 50, 75].map(y => `<line class="bb-c-grid" x1="0" x2="100" y1="${y}" y2="${y}"/>`).join('');
    if (kind === 'bar') {
      const n = Math.max(...series.map(s => s.v.length)), k = series.length, slot = 100 / n, bw = slot * .68 / k, y0 = Y(0);
      series.forEach((s, si) => s.v.forEach((v, i) => {
        const x = i * slot + slot * .16 + si * bw, y = Y(v);
        plot += `<rect x="${f2(x)}" y="${f2(Math.min(y, y0))}" width="${f2(bw * .92)}" height="${f2(Math.max(.6, Math.abs(y - y0)))}" style="fill:${esc(s.c)}"/>`;
      }));
    } else {
      series.forEach((s, si) => {
        const n = s.v.length, pts = s.v.map((v, i) => [n > 1 ? i / (n - 1) * 100 : 50, Y(v)]);
        const line = pts.map(p => f2(p[0]) + ',' + f2(p[1])).join(' L');
        if (kind === 'area' && n > 1) {
          const id = 'bbc' + (chart.n = (chart.n || 0) + 1) + si;
          plot += `<defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" style="stop-color:${esc(s.c)};stop-opacity:${f2(Math.min(1, fo * 3.2))}"/><stop offset="1" style="stop-color:${esc(s.c)};stop-opacity:${f2(fo * .35)}"/></linearGradient></defs>
            <path d="M${line} L100,100 L0,100Z" fill="url(#${id})"/>`;
        }
        if (n > 1) plot += `<path class="bb-c-line" d="M${line}" style="stroke:${esc(s.c)}"/>`;
        if (truthy(b.show_dots) || n === 1) dots += pts.map(p => `<i style="left:${f2(p[0])}%;top:${f2(p[1])}%;background:${esc(s.c)}"></i>`).join('');
      });
    }
    let labels = '';
    if (Array.isArray(b.x_labels) && b.x_labels.some(l => str(l).trim())) {
      const L = b.x_labels, n = L.length;
      labels = `<div class="bb-c-x${kind === 'bar' ? ' bb-c-xbar' : ''}">${L.map((l, i) => {
        if (!str(l).trim()) return '';
        const x = kind === 'bar' ? (i + .5) / n * 100 : n > 1 ? i / (n - 1) * 100 : 50;
        const edge = kind === 'bar' ? '' : i === 0 ? ' bb-c-first' : i === n - 1 ? ' bb-c-last' : '';
        return `<span class="${edge.trim()}" style="left:${f2(x)}%">${esc(l)}</span>`;
      }).join('')}</div>`;
    }
    const legend = truthy(b.show_legend) && series.some(s => s.label)
      ? `<div class="bb-c-legend">${series.filter(s => s.label).map(s => `<span><i style="background:${esc(s.c)}"></i>${esc(s.label)}</span>`).join('')}</div>` : '';
    return box(b, 'bb-chart', `height:${h}px`,
      `<div class="bb-c-plot"><svg viewBox="0 0 100 100" preserveAspectRatio="none">${plot}</svg>${dots}</div>${labels}${legend}`);
  }
  def('chart', chart);
  /* blocks.py serialises a chart's `type` param over the block type, so a chart can arrive as its kind. */
  ['line', 'bar', 'area'].forEach(t => def(t, chart));

  /* ── clock, countdown (tick in the page) ────────────────────── */
  const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  const p2 = n => String(n).padStart(2, '0');
  function clockParts(d, h24, secs) {
    const h = d.getHours();
    return { t: `${h24 ? h : h % 12 || 12}:${p2(d.getMinutes())}`, s: secs ? ':' + p2(d.getSeconds()) : '',
      m: h24 ? '' : h < 12 ? 'AM' : 'PM', date: `${DAYS[d.getDay()]} · ${MONTHS[d.getMonth()]} ${d.getDate()}` };
  }
  def('clock', b => {
    const h24 = b.format === '24h', secs = truthy(b.show_seconds), size = { md: 'md', xl: 'xl' }[b.size] || 'lg';
    const p = clockParts(BB.now(), h24, secs);
    return box(b, `bb-clock bb-clock-${size}`, '',
      (falsy(b.show_date) ? '' : `<div class="bb-ck-date label">${esc(p.date)}</div>`) +
      `<div class="bb-ck-time"><span class="bb-ck-t d num">${p.t}</span>${secs ? `<span class="bb-ck-s d num">${p.s}</span>` : ''}${h24 ? '' : `<span class="bb-ck-m">${p.m}</span>`}</div>`,
      ` data-h24="${h24 ? 1 : 0}" data-secs="${secs ? 1 : 0}"`);
  });

  function remaining(target, now) {
    const t = new Date(target);
    if (!has(target) || isNaN(t)) return null;
    let s = Math.max(0, Math.floor((t - now) / 1000));
    const d = Math.floor(s / 86400); s -= d * 86400;
    const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), r = s % 60;
    return (d ? `${d}d ` : '') + (d || h ? `${h}:${p2(m)}:${p2(r)}` : `${m}:${p2(r)}`);
  }
  def('countdown', b => {
    const left = remaining(b.target, BB.now());
    return box(b, 'bb-countdown', '', (has(b.label) ? `<div class="label">${esc(b.label)}</div>` : '') +
      `<div class="bb-cd-t d num">${left == null ? '–:––' : left}</div>`, ` data-target="${esc(b.target)}"`);
  });

  /* ── composites ─────────────────────────────────────────────── */
  const source = (b, ctx, dflt) => { const d = (ctx.data || {})[str(b.data_source) || dflt]; return d && typeof d === 'object' ? d : {}; };
  const deg = v => (has(v) ? esc(v) + '°' : '');

  def('weather_widget', (b, ctx) => {
    const w = source(b, ctx, 'weather'), days = (Array.isArray(w.forecast) ? w.forecast : []).filter(d => d && typeof d === 'object').slice(0, 5);
    if (!has(w.temp) && !has(w.condition) && !days.length) return box(b, 'bb-card bb-widget bb-empty', '', '<div class="label">Weather</div><div class="bb-none d">No forecast right now</div>');
    const facts = [has(w.humidity) ? `Humidity ${esc(w.humidity)}%` : '', has(w.wind) ? `Wind ${esc(w.wind)}` : ''].filter(Boolean).join('<i class="bb-sep"></i>');
    return box(b, 'bb-card bb-widget bb-wx', '',
      `<div class="bb-wx-now">${icon(w.icon || 'cloud', 'xl', 'var(--accent)')}<span class="bb-wx-t d num">${deg(w.temp)}</span>
        <div class="bb-wx-c"><b>${esc(w.condition)}</b>${facts ? `<span>${facts}</span>` : ''}</div></div>` +
      (days.length ? `<div class="bb-wx-days">${days.map(d => `<div><span class="label">${esc(d.day)}</span>${icon(d.icon || 'cloud', 'md', '')}<b class="num">${deg(d.high)}</b><span class="num">${deg(d.low)}</span></div>`).join('')}</div>` : ''));
  });

  def('calendar_widget', (b, ctx) => {
    const c = source(b, ctx, 'calendar'), ev = (Array.isArray(c.events) ? c.events : []).filter(e => e && typeof e === 'object').slice(0, 6);
    return box(b, 'bb-card bb-widget bb-cal', '', `<div class="label">Today</div>` + (ev.length
      ? ev.map(e => `<div class="bb-cal-row"><span class="bb-cal-t num">${esc(e.time)}</span><div><b>${esc(e.title)}</b>${has(e.location) ? `<span>${esc(e.location)}</span>` : ''}</div></div>`).join('')
      : '<div class="bb-none d">Nothing on the calendar</div>'));
  });

  /* ── rotate, page_dots ──────────────────────────────────────── */
  def('rotate', (b, ctx) => {
    const c = b.children || [];
    if (!c.length) return '';
    return box(b, 'bb-rotate', '', c.map((k, i) => `<div class="bb-page${i ? '' : ' on'}">${BB.render(k, sub(ctx, 'col'))}</div>`).join(''),
      ` data-interval="${Math.max(2, num(b.interval, 20))}"`);
  });
  def('page_dots', b => `<div class="bb bb-dots" style="${esc(css({ '--dot': color(b.color, '') }))}"></div>`);

  function syncDots(root) {
    const rot = root.querySelector('.bb-rotate');
    root.querySelectorAll('.bb-dots').forEach(el => {
      const n = rot ? rot.children.length : 0, on = rot ? [...rot.children].findIndex(c => c.classList.contains('on')) : -1;
      if (el.children.length !== n) el.replaceChildren(...Array.from({ length: n > 1 ? n : 0 }, () => document.createElement('i')));
      [...el.children].forEach((d, i) => d.classList.toggle('on', i === on));
    });
  }

  /* ── after mount: images, dots, ink ─────────────────────────── */
  /* A legacy layout does not know where the skyline is, so its ink is chosen after layout, per element:
     1. at once, from the scene's known zones (ground is dark, the skyline band is busy);
     2. then from the scene itself — the SVG is drawn once per atmosphere into a 128×75 canvas and each
        element reads the luminance behind it: scene ink, or the opposite ink where that would not hold,
        plus a halo of the opposite tone where the backdrop is busy or the contrast is thin. */
  const INKED = '.bb-text,.bb-icon,.bb-emoji,.bb-clock,.bb-countdown,.bb-metric,.bb-list,.bb-progress,.bb-divider,.bb-dots,.bb-badge,.bb-noimg';
  const SW = 128, SH = 75, LIN = Array.from({ length: 256 }, (_, i) => { const c = i / 255; return c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4; });
  let shot = { key: '', wait: null };

  function sceneShot() {   // → Promise<pixel data | null>, cached per atmosphere (the clock minute only moves the ferry)
    const st = BB.state || {}, a = st.atmosphere || {}, key = [a.phase, a.weather, a.vista, JSON.stringify(st.scene || {})].join('|');
    if (shot.key === key && shot.wait) return shot.wait;
    const all = document.querySelectorAll('#scene svg'), svg = all[all.length - 1];
    shot = { key, wait: new Promise(done => {
      try {
        const xml = new XMLSerializer().serializeToString(svg).replace(/^<svg /, '<svg width="1024" height="600" ');
        const img = new Image();
        img.onload = () => {
          try {
            const c = document.createElement('canvas'); c.width = SW; c.height = SH;
            const g = c.getContext('2d', { willReadFrequently: true });
            g.drawImage(img, 0, 0, SW, SH); done(g.getImageData(0, 0, SW, SH).data);
          } catch (e) { done(null); }
        };
        img.onerror = () => done(null);
        img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(xml);
      } catch (e) { done(null); }
    }) };
    return shot.wait;
  }
  function backdrop(px, x0, y0, x1, y1) {   // stage px → { mean, sd } of relative luminance
    const cx0 = Math.max(0, Math.floor(x0 / 8)), cx1 = Math.min(SW, Math.ceil(x1 / 8)), cy0 = Math.max(0, Math.floor(y0 / 8)), cy1 = Math.min(SH, Math.ceil(y1 / 8));
    let n = 0, sum = 0, sq = 0;
    for (let y = cy0; y < cy1; y++) for (let x = cx0; x < cx1; x++) {
      const i = (y * SW + x) * 4, L = .2126 * LIN[px[i]] + .7152 * LIN[px[i + 1]] + .0722 * LIN[px[i + 2]];
      n++; sum += L; sq += L * L;
    }
    if (!n) return null;
    const mean = sum / n;
    return { mean, sd: Math.sqrt(Math.max(0, sq / n - mean * mean)) };
  }
  function ink(root, px) {
    const stage = document.getElementById('stage'), sc = (BB.state && BB.state.scene) || {};
    if (!stage || !root.isConnected) return;
    const off = stage.classList.contains('no-scene'), dim = stage.classList.contains('dim'), light = stage.classList.contains('ink-light');
    const box0 = stage.getBoundingClientRect(), k = box0.width / 1024 || 1, zones = sc.horizon == null || sc.horizon === 'high';
    root.querySelectorAll('.bb-table,.bb-kv,.bb-chart').forEach(el => el.classList.toggle('bb-plate', !el.closest('.bb-card,.slab')));   // dense blocks never sit bare on the scene
    const range = document.createRange();
    root.querySelectorAll(INKED).forEach(el => {
      if (el.closest('.bb-card,.slab,.bb-plate')) return;
      let r = el.getBoundingClientRect(); if (!r.height) return;
      if (el.firstChild && !(el instanceof SVGElement)) {            // the words, not the stretched box they sit in
        range.selectNodeContents(el); const t = range.getBoundingClientRect(); if (t.width && t.height) r = t;
      }
      const x0 = (r.left - box0.left) / k, x1 = (r.right - box0.left) / k, y0 = (r.top - box0.top) / k, y1 = (r.bottom - box0.top) / k;
      let lite = false, dark = false, halo = false;
      const bg = px && !off && !dim ? backdrop(px, x0, y0, x1, y1) : null;
      if (bg) {
        const L = bg.mean, useLite = light ? L <= .45 : L < .17;          // lean towards the scene's own ink
        const contrast = useLite ? .97 / (L + .05) : (L + .05) / .07;
        lite = useLite && !light; dark = !useLite && light; halo = contrast < 7 || bg.sd > .07;
      } else if (!off && zones) {
        lite = (y0 + y1) / 2 >= GROUND_Y && y0 >= GROUND_Y - 90 && !light;     // ground; a tall hero that merely reaches down keeps sky ink
        halo = lite || y1 > 292 || (y1 > 150 && x1 > 290);                       // skyline band, left firs
      }
      el.classList.toggle('bb-lite', lite); el.classList.toggle('bb-dark', dark); el.classList.toggle('bb-halo', halo);
    });
  }
  function zoneInk(root) {
    ink(root, null);
    const stage = document.getElementById('stage');
    if (stage && !stage.classList.contains('no-scene')) sceneShot().then(px => { if (px) ink(root, px); });
  }
  BB.onMount(pane => {
    const root = pane.querySelector('.bb-root');
    pane.querySelectorAll('.bb-image > img').forEach(img => {
      const fail = () => { img.parentNode.classList.add('bb-noimg'); img.remove(); };
      if (img.complete && !img.naturalWidth) fail(); else img.addEventListener('error', fail, { once: true });
    });
    pane.querySelectorAll('.bb-rotate').forEach(r => { r.dataset.t0 = String(BB.now().getTime()); });
    syncDots(pane);
    if (root) zoneInk(root);
  });

  BB.onTick((d, view) => {
    view.querySelectorAll('.bb-clock').forEach(el => {
      const p = clockParts(d, el.dataset.h24 === '1', el.dataset.secs === '1');
      const set = (sel, v) => { const n = el.querySelector(sel); if (n && n.textContent !== v) n.textContent = v; };
      set('.bb-ck-t', p.t); set('.bb-ck-s', p.s); set('.bb-ck-m', p.m); set('.bb-ck-date', p.date);
    });
    view.querySelectorAll('.bb-countdown').forEach(el => {
      const left = remaining(el.dataset.target, d), n = el.querySelector('.bb-cd-t');
      if (n && left != null && n.textContent !== left) n.textContent = left;
    });
    view.querySelectorAll('.bb-rotate').forEach(el => {
      const n = el.children.length, t0 = +el.dataset.t0 || d.getTime();
      const i = Math.floor(Math.max(0, d - t0) / 1000 / (+el.dataset.interval || 20)) % n;
      [...el.children].forEach((c, j) => c.classList.toggle('on', j === i));
      const pane = el.closest('.pane'); if (pane) syncDots(pane);
    });
  });

  /* ── icons the legacy displays name that the sprite lacks ───── */
  /* Lucide (ISC, displays/assets/lucide/LICENSE.txt): the names the PIL renderer could draw. Generated; one line. */
  const LUCIDE = { activity: '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>', apple: '<path d="M12 6.528V3a1 1 0 0 1 1-1h0"/><path d="M18.237 21A15 15 0 0 0 22 11a6 6 0 0 0-10-4.472A6 6 0 0 0 2 11a15.1 15.1 0 0 0 3.763 10 3 3 0 0 0 3.648.648 5.5 5.5 0 0 1 5.178 0A3 3 0 0 0 18.237 21"/>', 'arrow-down': '<path d="M12 5v14"/><path d="m19 12-7 7-7-7"/>', 'arrow-left': '<path d="m12 19-7-7 7-7"/><path d="M19 12H5"/>', 'arrow-right': '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>', 'arrow-up': '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>', award: '<path d="m15.477 12.89 1.515 8.526a.5.5 0 0 1-.81.47l-3.58-2.687a1 1 0 0 0-1.197 0l-3.586 2.686a.5.5 0 0 1-.81-.469l1.514-8.526"/><circle cx="12" cy="8" r="6"/>', bath: '<path d="M10 4 8 6"/><path d="M17 19v2"/><path d="M2 12h20"/><path d="M7 19v2"/><path d="M9 5 7.621 3.621A2.121 2.121 0 0 0 4 5v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5"/>', 'battery-full': '<path d="M10 10v4"/><path d="M14 10v4"/><path d="M22 14v-4"/><path d="M6 10v4"/><rect x="2" y="6" width="16" height="12" rx="2"/>', 'battery-low': '<path d="M22 14v-4"/><path d="M6 14v-4"/><rect x="2" y="6" width="16" height="12" rx="2"/>', 'battery-medium': '<path d="M10 14v-4"/><path d="M22 14v-4"/><path d="M6 14v-4"/><rect x="2" y="6" width="16" height="12" rx="2"/>', 'battery-warning': '<path d="M10 17h.01"/><path d="M10 7v6"/><path d="M14 6h2a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2"/><path d="M22 14v-4"/><path d="M6 18H4a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h2"/>', battery: '<path d="M 22 14 L 22 10"/><rect x="2" y="6" width="16" height="12" rx="2"/>', bed: '<path d="M2 4v16"/><path d="M2 8h18a2 2 0 0 1 2 2v10"/><path d="M2 17h20"/><path d="M6 8v9"/>', 'bell-off': '<path d="M10.268 21a2 2 0 0 0 3.464 0"/><path d="M17 17H4a1 1 0 0 1-.74-1.673C4.59 13.956 6 12.499 6 8a6 6 0 0 1 .258-1.742"/><path d="m2 2 20 20"/><path d="M8.668 3.01A6 6 0 0 1 18 8c0 2.687.77 4.653 1.707 6.05"/>', 'bell-ring': '<path d="M10.268 21a2 2 0 0 0 3.464 0"/><path d="M22 8c0-2.3-.8-4.3-2-6"/><path d="M3.262 15.326A1 1 0 0 0 4 17h16a1 1 0 0 0 .74-1.673C19.41 13.956 18 12.499 18 8A6 6 0 0 0 6 8c0 4.499-1.411 5.956-2.738 7.326"/><path d="M4 2C2.8 3.7 2 5.7 2 8"/>', bell: '<path d="M10.268 21a2 2 0 0 0 3.464 0"/><path d="M3.262 15.326A1 1 0 0 0 4 17h16a1 1 0 0 0 .74-1.673C19.41 13.956 18 12.499 18 8A6 6 0 0 0 6 8c0 4.499-1.411 5.956-2.738 7.326"/>', bike: '<circle cx="18.5" cy="17.5" r="3.5"/><circle cx="5.5" cy="17.5" r="3.5"/><circle cx="15" cy="5" r="1"/><path d="M12 17.5V14l-3-3 4-3 2 3h2"/>', 'book-open': '<path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/>', book: '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H19a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H6.5a1 1 0 0 1 0-5H20"/>', 'calendar-check': '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/><path d="m9 16 2 2 4-4"/>', 'calendar-clock': '<path d="M16 14v2.2l1.6 1"/><path d="M16 2v4"/><path d="M21 7.5V6a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h3.5"/><path d="M3 10h5"/><path d="M8 2v4"/><circle cx="16" cy="16" r="6"/>', 'calendar-days': '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/><path d="M8 14h.01"/><path d="M12 14h.01"/><path d="M16 14h.01"/><path d="M8 18h.01"/><path d="M12 18h.01"/><path d="M16 18h.01"/>', camera: '<path d="M13.997 4a2 2 0 0 1 1.76 1.05l.486.9A2 2 0 0 0 18.003 7H20a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1.997a2 2 0 0 0 1.759-1.048l.489-.904A2 2 0 0 1 10.004 4z"/><circle cx="12" cy="13" r="3"/>', 'chevron-down': '<path d="m6 9 6 6 6-6"/>', 'chevron-left': '<path d="m15 18-6-6 6-6"/>', 'chevron-right': '<path d="m9 18 6-6-6-6"/>', 'chevron-up': '<path d="m18 15-6-6-6 6"/>', 'clock-2': '<path d="M12 6v6l4-2"/><circle cx="12" cy="12" r="10"/>', clock: '<path d="M12 6v6l4 2"/><circle cx="12" cy="12" r="10"/>', 'cloud-hail': '<path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M16 14v2"/><path d="M8 14v2"/><path d="M16 20h.01"/><path d="M8 20h.01"/><path d="M12 16v2"/><path d="M12 22h.01"/>', coffee: '<path d="M10 2v2"/><path d="M14 2v2"/><path d="M16 8a1 1 0 0 1 1 1v8a4 4 0 0 1-4 4H7a4 4 0 0 1-4-4V9a1 1 0 0 1 1-1h14a4 4 0 1 1 0 8h-1"/><path d="M6 2v2"/>', crown: '<path d="M11.562 3.266a.5.5 0 0 1 .876 0L15.39 8.87a1 1 0 0 0 1.516.294L21.183 5.5a.5.5 0 0 1 .798.519l-2.834 10.246a1 1 0 0 1-.956.734H5.81a1 1 0 0 1-.957-.734L2.02 6.02a.5.5 0 0 1 .798-.519l4.276 3.664a1 1 0 0 0 1.516-.294z"/><path d="M5 21h14"/>', 'disc-2': '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/><path d="M12 12h.01"/>', disc: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="2"/>', 'door-closed': '<path d="M10 12h.01"/><path d="M18 20V6a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v14"/><path d="M2 20h20"/>', 'door-open': '<path d="M11 20H2"/><path d="M11 4.562v16.157a1 1 0 0 0 1.242.97L19 20V5.562a2 2 0 0 0-1.515-1.94l-4-1A2 2 0 0 0 11 4.561z"/><path d="M11 4H8a2 2 0 0 0-2 2v14"/><path d="M14 12h.01"/><path d="M22 20h-3"/>', dot: '<circle cx="12.1" cy="12.1" r="1"/>', droplets: '<path d="M7 16.3c2.2 0 4-1.83 4-4.05 0-1.16-.57-2.26-1.71-3.19S7.29 6.75 7 5.3c-.29 1.45-1.14 2.84-2.29 3.76S3 11.1 3 12.25c0 2.22 1.8 4.05 4 4.05z"/><path d="M12.56 6.6A10.97 10.97 0 0 0 14 3.02c.5 2.5 2 4.9 4 6.5s3 3.5 3 5.5a6.98 6.98 0 0 1-11.91 4.97"/>', 'ear-off': '<path d="M6 18.5a3.5 3.5 0 1 0 7 0c0-1.57.92-2.52 2.04-3.46"/><path d="M6 8.5c0-.75.13-1.47.36-2.14"/><path d="M8.8 3.15A6.5 6.5 0 0 1 19 8.5c0 1.63-.44 2.81-1.09 3.76"/><path d="M12.5 6A2.5 2.5 0 0 1 15 8.5M10 13a2 2 0 0 0 1.82-1.18"/><line x1="2" x2="22" y1="2" y2="22"/>', 'eye-off': '<path d="M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49"/><path d="M14.084 14.158a3 3 0 0 1-4.242-4.242"/><path d="M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143"/><path d="m2 2 20 20"/>', eye: '<path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/>', 'fast-forward': '<path d="M12 6a2 2 0 0 1 3.414-1.414l6 6a2 2 0 0 1 0 2.828l-6 6A2 2 0 0 1 12 18z"/><path d="M2 6a2 2 0 0 1 3.414-1.414l6 6a2 2 0 0 1 0 2.828l-6 6A2 2 0 0 1 2 18z"/>', file: '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/>', flag: '<path d="M4 22V4a1 1 0 0 1 .4-.8A6 6 0 0 1 8 2c3 0 5 2 7.333 2q2 0 3.067-.8A1 1 0 0 1 20 4v10a1 1 0 0 1-.4.8A6 6 0 0 1 16 16c-3 0-5-2-8-2a6 6 0 0 0-4 1.528"/>', flame: '<path d="M12 3q1 4 4 6.5t3 5.5a1 1 0 0 1-14 0 5 5 0 0 1 1-3 1 1 0 0 0 5 0c0-2-1.5-3-1.5-5q0-2 2.5-4"/>', folder: '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>', frown: '<circle cx="12" cy="12" r="10"/><path d="M16 16s-1.5-2-4-2-4 2-4 2"/><line x1="9" x2="9.01" y1="9" y2="9"/><line x1="15" x2="15.01" y1="9" y2="9"/>', gauge: '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>', gift: '<rect x="3" y="8" width="18" height="4" rx="1"/><path d="M12 8v13"/><path d="M19 12v7a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2v-7"/><path d="M7.5 8a2.5 2.5 0 0 1 0-5A4.8 8 0 0 1 12 8a4.8 8 0 0 1 4.5-5 2.5 2.5 0 0 1 0 5"/>', headphones: '<path d="M3 14h3a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-7a9 9 0 0 1 18 0v7a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3"/>', heart: '<path d="M2 9.5a5.5 5.5 0 0 1 9.591-3.676.56.56 0 0 0 .818 0A5.49 5.49 0 0 1 22 9.5c0 2.29-1.5 4-3 5.5l-5.492 5.313a2 2 0 0 1-3 .019L5 15c-1.5-1.5-3-3.2-3-5.5"/>', hourglass: '<path d="M5 22h14"/><path d="M5 2h14"/><path d="M17 22v-4.172a2 2 0 0 0-.586-1.414L12 12l-4.414 4.414A2 2 0 0 0 7 17.828V22"/><path d="M7 2v4.172a2 2 0 0 0 .586 1.414L12 12l4.414-4.414A2 2 0 0 0 17 6.172V2"/>', image: '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>', images: '<path d="m22 11-1.296-1.296a2.4 2.4 0 0 0-3.408 0L11 16"/><path d="M4 8a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2"/><circle cx="13" cy="7" r="1" fill="currentColor"/><rect x="8" y="2" width="14" height="14" rx="2"/>', info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>', key: '<path d="m15.5 7.5 2.3 2.3a1 1 0 0 0 1.4 0l2.1-2.1a1 1 0 0 0 0-1.4L19 4"/><path d="m21 2-9.6 9.6"/><circle cx="7.5" cy="15.5" r="5.5"/>', lamp: '<path d="M12 12v6"/><path d="M4.077 10.615A1 1 0 0 0 5 12h14a1 1 0 0 0 .923-1.385l-3.077-7.384A2 2 0 0 0 15 2H9a2 2 0 0 0-1.846 1.23Z"/><path d="M8 20a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v1a1 1 0 0 1-1 1H9a1 1 0 0 1-1-1z"/>', leaf: '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>', 'lightbulb-off': '<path d="M16.8 11.2c.8-.9 1.2-2 1.2-3.2a6 6 0 0 0-9.3-5"/><path d="m2 2 20 20"/><path d="M6.3 6.3a4.67 4.67 0 0 0 1.2 5.2c.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>', lightbulb: '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>', 'list-checks': '<path d="M13 5h8"/><path d="M13 12h8"/><path d="M13 19h8"/><path d="m3 17 2 2 4-4"/><path d="m3 7 2 2 4-4"/>', 'loader-circle': '<path d="M21 12a9 9 0 1 1-6.219-8.56"/>', loader: '<path d="M12 2v4"/><path d="m16.2 7.8 2.9-2.9"/><path d="M18 12h4"/><path d="m16.2 16.2 2.9 2.9"/><path d="M12 18v4"/><path d="m4.9 19.1 2.9-2.9"/><path d="M2 12h4"/><path d="m4.9 4.9 2.9 2.9"/>', lock: '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>', 'mail-open': '<path d="M21.2 8.4c.5.38.8.97.8 1.6v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V10a2 2 0 0 1 .8-1.6l8-6a2 2 0 0 1 2.4 0l8 6Z"/><path d="m22 10-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 10"/>', 'map-pin': '<path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0"/><circle cx="12" cy="10" r="3"/>', map: '<path d="M14.106 5.553a2 2 0 0 0 1.788 0l3.659-1.83A1 1 0 0 1 21 4.619v12.764a1 1 0 0 1-.553.894l-4.553 2.277a2 2 0 0 1-1.788 0l-4.212-2.106a2 2 0 0 0-1.788 0l-3.659 1.83A1 1 0 0 1 3 19.381V6.618a1 1 0 0 1 .553-.894l4.553-2.277a2 2 0 0 1 1.788 0z"/><path d="M15 5.764v15"/><path d="M9 3.236v15"/>', meh: '<circle cx="12" cy="12" r="10"/><line x1="8" x2="16" y1="15" y2="15"/><line x1="9" x2="9.01" y1="9" y2="9"/><line x1="15" x2="15.01" y1="9" y2="9"/>', menu: '<path d="M4 5h16"/><path d="M4 12h16"/><path d="M4 19h16"/>', 'message-circle': '<path d="M2.992 16.342a2 2 0 0 1 .094 1.167l-1.065 3.29a1 1 0 0 0 1.236 1.168l3.413-.998a2 2 0 0 1 1.099.092 10 10 0 1 0-4.777-4.719"/>', 'mic-off': '<path d="M12 19v3"/><path d="M15 9.34V5a3 3 0 0 0-5.68-1.33"/><path d="M16.95 16.95A7 7 0 0 1 5 12v-2"/><path d="M18.89 13.23A7 7 0 0 0 19 12v-2"/><path d="m2 2 20 20"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12"/>', minus: '<path d="M5 12h14"/>', music: '<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>', newspaper: '<path d="M15 18h-5"/><path d="M18 14h-8"/><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-4 0v-9a2 2 0 0 1 2-2h2"/><rect width="8" height="4" x="10" y="6" rx="1"/>', package: '<path d="M11 21.73a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73z"/><path d="M12 22V12"/><polyline points="3.29 7 12 12 20.71 7"/><path d="m7.5 4.27 9 5.15"/>', 'party-popper': '<path d="M5.8 11.3 2 22l10.7-3.79"/><path d="M4 3h.01"/><path d="M22 8h.01"/><path d="M15 2h.01"/><path d="M22 20h.01"/><path d="m22 2-2.24.75a2.9 2.9 0 0 0-1.96 3.12c.1.86-.57 1.63-1.45 1.63h-.38c-.86 0-1.6.6-1.76 1.44L14 10"/><path d="m22 13-.82-.33c-.86-.34-1.82.2-1.98 1.11c-.11.7-.72 1.22-1.43 1.22H17"/><path d="m11 2 .33.82c.34.86-.2 1.82-1.11 1.98C9.52 4.9 9 5.52 9 6.23V7"/><path d="M11 13c1.93 1.93 2.83 4.17 2 5-.83.83-3.07-.07-5-2-1.93-1.93-2.83-4.17-2-5 .83-.83 3.07.07 5 2Z"/>', pause: '<rect x="14" y="3" width="5" height="18" rx="1"/><rect x="5" y="3" width="5" height="18" rx="1"/>', percent: '<line x1="19" x2="5" y1="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>', phone: '<path d="M13.832 16.568a1 1 0 0 0 1.213-.303l.355-.465A2 2 0 0 1 17 15h3a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2A18 18 0 0 1 2 4a2 2 0 0 1 2-2h3a2 2 0 0 1 2 2v3a2 2 0 0 1-.8 1.6l-.468.351a1 1 0 0 0-.292 1.233 14 14 0 0 0 6.392 6.384"/>', pizza: '<path d="m12 14-1 1"/><path d="m13.75 18.25-1.25 1.42"/><path d="M17.775 5.654a15.68 15.68 0 0 0-12.121 12.12"/><path d="M18.8 9.3a1 1 0 0 0 2.1 7.7"/><path d="M21.964 20.732a1 1 0 0 1-1.232 1.232l-18-5a1 1 0 0 1-.695-1.232A19.68 19.68 0 0 1 15.732 2.037a1 1 0 0 1 1.232.695z"/>', plane: '<path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z"/>', play: '<path d="M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z"/>', plus: '<path d="M5 12h14"/><path d="M12 5v14"/>', radio: '<path d="M16.247 7.761a6 6 0 0 1 0 8.478"/><path d="M19.075 4.933a10 10 0 0 1 0 14.134"/><path d="M4.925 19.067a10 10 0 0 1 0-14.134"/><path d="M7.753 16.239a6 6 0 0 1 0-8.478"/><circle cx="12" cy="12" r="2"/>', repeat: '<path d="m17 2 4 4-4 4"/><path d="M3 11v-1a4 4 0 0 1 4-4h14"/><path d="m7 22-4-4 4-4"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/>', rewind: '<path d="M12 6a2 2 0 0 0-3.414-1.414l-6 6a2 2 0 0 0 0 2.828l6 6A2 2 0 0 0 12 18z"/><path d="M22 6a2 2 0 0 0-3.414-1.414l-6 6a2 2 0 0 0 0 2.828l6 6A2 2 0 0 0 22 18z"/>', rocket: '<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/>', search: '<path d="m21 21-4.34-4.34"/><circle cx="11" cy="11" r="8"/>', settings: '<path d="M9.671 4.136a2.34 2.34 0 0 1 4.659 0 2.34 2.34 0 0 0 3.319 1.915 2.34 2.34 0 0 1 2.33 4.033 2.34 2.34 0 0 0 0 3.831 2.34 2.34 0 0 1-2.33 4.033 2.34 2.34 0 0 0-3.319 1.915 2.34 2.34 0 0 1-4.659 0 2.34 2.34 0 0 0-3.32-1.915 2.34 2.34 0 0 1-2.33-4.033 2.34 2.34 0 0 0 0-3.831A2.34 2.34 0 0 1 6.35 6.051a2.34 2.34 0 0 0 3.319-1.915"/><circle cx="12" cy="12" r="3"/>', 'shield-alert': '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="M12 8v4"/><path d="M12 16h.01"/>', 'shield-check': '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>', shield: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>', 'signal-high': '<path d="M2 20h.01"/><path d="M7 20v-4"/><path d="M12 20v-8"/><path d="M17 20V8"/>', 'signal-low': '<path d="M2 20h.01"/><path d="M7 20v-4"/>', signal: '<path d="M2 20h.01"/><path d="M7 20v-4"/><path d="M12 20v-8"/><path d="M17 20V8"/><path d="M22 4v16"/>', 'skip-back': '<path d="M17.971 4.285A2 2 0 0 1 21 6v12a2 2 0 0 1-3.029 1.715l-9.997-5.998a2 2 0 0 1-.003-3.432z"/><path d="M3 20V4"/>', 'skip-forward': '<path d="M21 4v16"/><path d="M6.029 4.285A2 2 0 0 0 3 6v12a2 2 0 0 0 3.029 1.715l9.997-5.998a2 2 0 0 0 .003-3.432z"/>', smile: '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" x2="9.01" y1="9" y2="9"/><line x1="15" x2="15.01" y1="9" y2="9"/>', sparkles: '<path d="M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z"/><path d="M20 2v4"/><path d="M22 4h-4"/><circle cx="4" cy="20" r="2"/>', square: '<rect width="18" height="18" x="3" y="3" rx="2"/>', star: '<path d="M11.525 2.295a.53.53 0 0 1 .95 0l2.31 4.679a2.123 2.123 0 0 0 1.595 1.16l5.166.756a.53.53 0 0 1 .294.904l-3.736 3.638a2.123 2.123 0 0 0-.611 1.878l.882 5.14a.53.53 0 0 1-.771.56l-4.618-2.428a2.122 2.122 0 0 0-1.973 0L6.396 21.01a.53.53 0 0 1-.77-.56l.881-5.139a2.122 2.122 0 0 0-.611-1.879L2.16 9.795a.53.53 0 0 1 .294-.906l5.165-.755a2.122 2.122 0 0 0 1.597-1.16z"/>', 'thermometer-snowflake': '<path d="m10 20-1.25-2.5L6 18"/><path d="M10 4 8.75 6.5 6 6"/><path d="M10.585 15H10"/><path d="M2 12h6.5L10 9"/><path d="M20 14.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0z"/><path d="m4 10 1.5 2L4 14"/><path d="m7 21 3-6-1.5-3"/><path d="m7 3 3 6h2"/>', 'thermometer-sun': '<path d="M12 9a4 4 0 0 0-2 7.5"/><path d="M12 3v2"/><path d="m6.6 18.4-1.4 1.4"/><path d="M20 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/><path d="M4 13H2"/><path d="M6.34 7.34 4.93 5.93"/>', thermometer: '<path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/>', ticket: '<path d="M2 9a3 3 0 0 1 0 6v2a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-2a3 3 0 0 1 0-6V7a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2Z"/><path d="M13 5v2"/><path d="M13 17v2"/><path d="M13 11v2"/>', timer: '<line x1="10" x2="14" y1="2" y2="2"/><line x1="12" x2="15" y1="14" y2="11"/><circle cx="12" cy="14" r="8"/>', 'trash-2': '<path d="M10 11v6"/><path d="M14 11v6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>', trash: '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>', trophy: '<path d="M10 14.66v1.626a2 2 0 0 1-.976 1.696A5 5 0 0 0 7 21.978"/><path d="M14 14.66v1.626a2 2 0 0 0 .976 1.696A5 5 0 0 1 17 21.978"/><path d="M18 9h1.5a1 1 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M6 9a6 6 0 0 0 12 0V3a1 1 0 0 0-1-1H7a1 1 0 0 0-1 1z"/><path d="M6 9H4.5a1 1 0 0 1 0-5H6"/>', truck: '<path d="M14 18V6a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v11a1 1 0 0 0 1 1h2"/><path d="M15 18H9"/><path d="M19 18h2a1 1 0 0 0 1-1v-3.65a1 1 0 0 0-.22-.624l-3.48-4.35A1 1 0 0 0 17.52 8H14"/><circle cx="17" cy="18" r="2"/><circle cx="7" cy="18" r="2"/>', tv: '<path d="m17 2-5 5-5-5"/><rect width="20" height="15" x="2" y="7" rx="2"/>', umbrella: '<path d="M12 13v7a2 2 0 0 0 4 0"/><path d="M12 2v2"/><path d="M20.992 13a1 1 0 0 0 .97-1.274 10.284 10.284 0 0 0-19.923 0A1 1 0 0 0 3 13z"/>', 'user-check': '<path d="m16 11 2 2 4-4"/><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>', 'user-minus': '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="22" x2="16" y1="11" y2="11"/>', 'user-plus': '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="19" x2="19" y1="8" y2="14"/><line x1="22" x2="16" y1="11" y2="11"/>', user: '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>', users: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><path d="M16 3.128a4 4 0 0 1 0 7.744"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><circle cx="9" cy="7" r="4"/>', 'volume-2': '<path d="M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z"/><path d="M16 9a5 5 0 0 1 0 6"/><path d="M19.364 18.364a9 9 0 0 0 0-12.728"/>', 'volume-x': '<path d="M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z"/><line x1="22" x2="16" y1="9" y2="15"/><line x1="16" x2="22" y1="9" y2="15"/>', 'wifi-off': '<path d="M12 20h.01"/><path d="M8.5 16.429a5 5 0 0 1 7 0"/><path d="M5 12.859a10 10 0 0 1 5.17-2.69"/><path d="M19 12.859a10 10 0 0 0-2.007-1.523"/><path d="M2 8.82a15 15 0 0 1 4.177-2.643"/><path d="M22 8.82a15 15 0 0 0-11.288-3.764"/><path d="m2 2 20 20"/>', wifi: '<path d="M12 20h.01"/><path d="M2 8.82a15 15 0 0 1 20 0"/><path d="M5 12.859a10 10 0 0 1 14 0"/><path d="M8.5 16.429a5 5 0 0 1 7 0"/>', x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>', zap: '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>' };
  const EXTRA = {
    home: '<path d="M4 11.2L12 4.5l8 6.7M6.3 9.8V19.5h11.4V9.8M10.3 19.5v-5h3.4v5"/>',
    'trending-up': '<path d="M3 17l6-6 4 4 8-8M15 7h6v6"/>', 'trending-down': '<path d="M3 7l6 6 4-4 8 8M15 17h6v-6"/>',
    'arrow-up-right': '<path d="M7 17L17 7M8 7h9v9"/>', 'arrow-down-right': '<path d="M7 7l10 10M17 8v9H8"/>',
    'alert-triangle': '<path d="M12 4.5l8.5 14.5h-17zM12 10v4.5M12 17h.01"/>', 'check-circle': '<circle cx="12" cy="12" r="8.5"/><path d="M8.5 12.3l2.5 2.5 4.6-5"/>',
  };
  const ALIAS = { house: 'home', 'triangle-alert': 'alert-triangle', 'circle-check': 'check-circle', 'circle-alert': 'info', 'alarm-clock': 'clock' };
  Object.entries({ ...LUCIDE, ...EXTRA }).forEach(([k, v]) => { if (!ICONS[k] && !ICON_ALIAS[k]) ICONS[k] = v; });
  Object.entries(ALIAS).forEach(([k, v]) => { if (!ICONS[k] && !ICON_ALIAS[k]) ICON_ALIAS[k] = v; });
})();
