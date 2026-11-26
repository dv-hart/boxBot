/* Living Sky — page runtime. One global: BB.

   Python (displays/sky_surface.py) calls BB.apply(state) over the DevTools pipe every time what should be
   on screen changes. This file owns the layers, the atmosphere tokens and the block registry; everything
   drawn inside the view layer comes from a block registered with BB.define(). State contract: README.md.

   Block renderers return HTML strings. EVERY value that came from data goes through BB.esc() — calendar
   titles, email subjects and photo captions are written by strangers. The page's CSP blocks inline script
   as a second line, not as the first. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = v => (v == null ? '' : String(v)).replace(/[&<>"']/g, c => ESC[c]);
  const slug = v => String(v == null ? '' : v).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  const styleOf = vars => Object.entries(vars).map(([k, v]) => k + ':' + v).join(';');

  const blocks = {};      // type -> (block, ctx) => html
  const tickers = [];     // (Date, root) => void, once a second
  const mounts = [];      // (root, ctx) => void, after a view pane lands in the DOM
  let state = null, ctx = null, lastView = '', lastScene = '', lastDisplay = null, clockOffset = 0;

  /* ── atmosphere ─────────────────────────────────────────────── */
  function slabVars(P, phase) {
    const base = (P.ridge && P.ridge[4]) || '#142430', night = phase === 'night';
    const c = tune(base, night ? .8 : .9, l => Math.max(.07, Math.min(.125, l * .68)));
    const hi = tune(base, .75, l => Math.max(.2, Math.min(.27, l * 1.25)));
    const txt = night ? '#ddd5c6' : '#fff6e6', acc = night ? '#e9c27c' : '#ffcf7d';
    const L = PEOPLE.light, dimmed = col => (night ? tune(col, .9, l => l * .93) : col);
    return {
      '--slab': rgba(c, .9), '--slab-lip': rgba(c, .8), '--slab-solid': c, '--slab-fade': rgba(c, 0),
      '--raise': rgba(hi, .62), '--st': txt, '--sm': rgba(txt, .72), '--sh': rgba(txt, .16), '--sh2': rgba(txt, .34),
      '--sa': acc, '--paper': night ? '#d8cfc0' : '#fbf1de', '--on-accent': c,
      '--l1': dimmed(L.p1), '--l2': dimmed(L.p2), '--l3': dimmed(L.p3), '--l4': dimmed(L.p4), '--lf': dimmed(L.family),
    };
  }

  function paintScene(atmo, opts) {
    const stage = $('stage'), o = {
      phase: atmo.phase, weather: atmo.weather, time: atmo.time, vista: opts.vista || atmo.vista || 'kerry',
      horizon: opts.horizon != null ? opts.horizon : 'high', seed: opts.seed || 7,
    };
    if (opts.scale != null) o.scale = opts.scale;
    if (opts.mountain != null) o.mountain = opts.mountain;
    const key = JSON.stringify(o);
    stage.classList.toggle('no-scene', opts.hidden === true);
    const P = palette(o.phase, o.weather);
    if (key !== lastScene) {
      lastScene = key;
      stage.setAttribute('style', styleOf({ ...P.vars, ...slabVars(P, o.phase) }) + ';' + stageTransform());
      stage.className = stage.className.replace(/\b(ph|wx|ink)-\S+/g, '').trim() + ` ph-${o.phase} wx-${o.weather} ink-${P.ink}`;
      swap($('scene'), scene(o).svg);
    }
    stage.classList.toggle('dim', !!atmo.dim);
    return P;
  }

  /* Cross-fade a layer's content. Old node fades out and is removed; never more than two alive. */
  function swap(layer, html, cls, instant) {
    const next = document.createElement('div');
    if (cls) next.className = cls;
    next.innerHTML = html;
    const olds = [...layer.children];
    if (instant || !olds.length || BB.still) { layer.replaceChildren(next); return next; }
    next.classList.add('out');
    layer.appendChild(next);
    void next.offsetWidth;                       // commit the starting opacity before the transition
    next.classList.remove('out');
    olds.forEach((o, i) => { if (i < olds.length - 1) o.remove(); else { o.classList.add('out'); setTimeout(() => o.remove(), 750); } });
    return next;
  }

  const stageTransform = () => {
    const k = Math.min(innerWidth / 1024, innerHeight / 600) || 1;
    return k === 1 ? '' : `transform:scale(${k})`;
  };

  /* ── blocks ─────────────────────────────────────────────────── */
  function render(block, c) {
    if (block == null) return '';
    if (typeof block === 'string') return esc(block);
    const fn = blocks[block.type];
    if (!fn) return `<span class="bb-unknown">${esc(block.type || 'block')}?</span>`;
    try { return fn(block, c || ctx) || ''; }
    catch (e) { console.error('block', block.type, e); return `<span class="bb-unknown">${esc(block.type)} failed</span>`; }
  }
  const children = (block, c) => (block.children || []).map(b => render(b, c)).join('');

  /* Household member → colour class. Python sends people in a stable order; slot = position. */
  function person(name) {
    const key = slug(name);
    if (!key) return '';
    if (key === 'family' || key === 'everyone' || key === 'all') return 'p-family';
    const i = ((state && state.people) || []).findIndex(p => slug(p) === key);
    return i >= 0 && i < 4 ? 'p-p' + (i + 1) : 'p-family';
  }

  /* ── time ───────────────────────────────────────────────────── */
  const now = () => new Date(Date.now() + clockOffset);
  function fmtTime(d, o = {}) {
    const h = d.getHours(), m = String(d.getMinutes()).padStart(2, '0');
    if (o.h24) return `${String(h).padStart(2, '0')}:${m}`;
    return `${h % 12 || 12}:${m}` + (o.meridiem === false ? '' : (o.sep == null ? ' ' : o.sep) + (h < 12 ? 'AM' : 'PM'));
  }
  /* "in 1 h 28 min", "in 29 min", "now"; past → '' */
  function until(date, from) {
    const min = Math.round((new Date(date) - (from || now())) / 60000);
    if (isNaN(min) || min < 0) return '';
    if (min < 1) return 'now';
    if (min < 60) return `in ${min} min`;
    const h = Math.floor(min / 60), r = min % 60;
    return h < 24 ? `in ${h} h${r ? ` ${r} min` : ''}` : `in ${Math.round(h / 24)} d`;
  }

  /* ── state in ───────────────────────────────────────────────── */
  function apply(next) {
    if (typeof next === 'string') next = JSON.parse(next);
    state = next;
    const atmo = next.atmosphere || { phase: 'noon', weather: 'clear' };
    clockOffset = next.now ? new Date(next.now) - Date.now() : 0;
    const P = paintScene(atmo, next.scene || {});
    ctx = { state: next, data: next.data || {}, args: next.args || {}, atmosphere: atmo, P, now: now(), display: next.display };

    $('stage').classList.toggle('raw', !!next.frame);
    if (next.frame) $('frame').src = next.frame;

    const key = JSON.stringify([next.display, next.layout, next.args, next.people, lastScene]);
    if (key !== lastView) {
      lastView = key;
      const sameDisplay = next.display === lastDisplay && next.transition !== 'always';
      lastDisplay = next.display;
      const pane = swap($('view'), render(next.layout, ctx), 'pane d-' + slug(next.display),
        sameDisplay || next.transition === 'none');
      mounts.forEach(fn => { try { fn(pane, ctx); } catch (e) { console.error(e); } });
      tick();
    }
    overlays(next);
    return true;
  }

  function overlays(s) {
    const st = blocks.status_pill, pr = blocks.presence;
    $('status').innerHTML = s.status && st ? st({ type: 'status_pill', text: s.status }, ctx) : '';
    const key = s.presence ? JSON.stringify(s.presence) : '';
    const el = $('presence');
    if (el.dataset.key !== key) {
      el.dataset.key = key;
      if (BB.presence) BB.presence(el, s.presence || null, ctx);
      else el.innerHTML = s.presence && pr ? pr({ type: 'presence', ...s.presence }, ctx) : '';
    }
  }

  function tick() {
    const d = now(), root = $('view');
    tickers.forEach(fn => { try { fn(d, root); } catch (e) { console.error(e); } });
  }

  function boot() {
    mountIcons();
    addEventListener('resize', () => { lastScene = ''; if (state) apply(state); });
    setInterval(tick, 1000);
    const q = new URLSearchParams(location.search);
    BB.still = q.has('still') || q.has('fixture');        // no cross-fades: screenshots land on the final frame
    if (q.get('fixture')) {                                // dev: index.html?fixture=home&phase=night&weather=rain
      const s = document.createElement('script');
      s.src = `fixtures/${q.get('fixture').replace(/[^\w-]/g, '')}.js`;
      s.onload = () => {
        const f = BB.fixture; if (!f) return;
        f.atmosphere = { ...(f.atmosphere || {}) };
        ['phase', 'weather', 'time', 'vista'].forEach(k => q.get(k) && (f.atmosphere[k] = q.get(k)));
        if (q.has('dim')) f.atmosphere.dim = q.get('dim') !== '0';
        if (q.get('now')) f.now = q.get('now');
        if (q.get('status')) f.status = q.get('status');
        if (q.get('presence')) f.presence = { state: q.get('presence'), who: q.get('who') || undefined, text: q.get('text') || undefined,
          detail: q.get('detail') || undefined, progress: q.get('progress') ? +q.get('progress') : undefined };
        apply(f);
      };
      document.head.appendChild(s);
    }
    BB.ready = true;
  }

  window.BB = {
    define: (type, fn) => { blocks[type] = fn; },
    alias: (type, of) => { blocks[type] = (b, c) => blocks[of](b, c); },
    onTick: fn => tickers.push(fn),
    onMount: fn => mounts.push(fn),
    render, children, esc, slug, person, now, fmtTime, until, styleOf, apply, blocks,
    get state() { return state; }, get ctx() { return ctx; },
    ready: false, still: false, presence: null, fixture: null,
  };
  document.readyState === 'loading' ? addEventListener('DOMContentLoaded', boot) : boot();
})();
