/* picture, photo_caption, sky_chip, capture.

   picture — one photo (or a pair) with its caption. photos[0] is the one on screen; a slideshow is Python
   rotating that list and pushing again, and this block cross-fades from the previous photo.
   Layout rule (layout:"auto"), all from the photo record:
     pair && 2+ portraits            → diptych  two portraits at full height
     aspect < 1.2                    → side     uncropped at full height, flush left; the living sky beside it
     landscape, crop loses <= 25 %   → fill     cover; crop steered by the person boxes (none: keep the top third)
       and no person is cut
     otherwise                       → fit      whole photo over its own stretched thumbnail
   side sits over the scene layer; fill / fit / centred diptych paint an opaque frame, so they look the same
   whether or not the spec hides the scene.

   Photo record: {id, src, thumb?, width?, height?, title?, description?, people?:[{name, box:[x,y,w,h] 0..1}],
   taken?, source?}. src / thumb are absolute paths, file: or data: URLs (or a path relative to the page, for
   fixtures); any other scheme is dropped. Missing or wrong width/height are measured when the image loads.

   Every string from a record goes through BB.esc(): captions, names and paths are written outside the house. */
(function () {
  'use strict';
  ICONS.knob = ICONS.knob || '<circle cx="12" cy="12" r="8.5"/><path d="M12 3.5v5"/>';
  ICONS.phone = '<rect x="7" y="3" width="10" height="18" rx="2.6"/><path d="M11 17.8h2"/>';
  ICONS.lens = '<path d="M4 8.5h3l1.6-2.5h6.8L17 8.5h3v10.5H4z"/><circle cx="12" cy="13.4" r="3.4"/>';
  ICONS.spark = '<path d="M12 3.5l1.9 5.6 5.6 1.9-5.6 1.9-1.9 5.6-1.9-5.6L4.5 11l5.6-1.9z"/>';

  const FW = 1024, FH = 600, GAP = 6, esc = BB.esc;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const arr = v => (Array.isArray(v) ? v : []);
  const px = v => v.toFixed(1) + 'px';
  const DIMS = {};        // image URL → {w, h} measured on load (wins over the record), or {bad: true}
  const LIVE = {};        // data-pid → block JSON, so a block can redraw itself once an image has been measured
  let uid = 0, last = null;   // last = {key, html}: the photo stage drawn most recently, for the cross-fade

  /* ── data in ── */
  /* Path or URL → something an <img> may load under the page CSP (file: and data: only). '' = refuse. */
  function url(s) {
    s = s == null ? '' : String(s).trim();
    if (!s) return '';
    if (/^data:/i.test(s)) return /^data:image\/(png|jpe?g|webp|gif|avif);base64,[a-z0-9+/=\s]+$/i.test(s) ? s : '';
    const enc = p => p.split('/').map(seg => { try { return encodeURIComponent(decodeURIComponent(seg)); } catch (e) { return encodeURIComponent(seg); } }).join('/');
    if (/^file:\/\//i.test(s)) return 'file://' + enc(s.slice(7).replace(/[?#].*$/, ''));
    if (/^[a-z][a-z0-9+.-]*:/i.test(s)) return '';
    return s[0] === '/' ? 'file://' + s.split('/').map(encodeURIComponent).join('/') : s.split('/').map(encodeURIComponent).join('/');
  }
  function box(b, w, h) {
    if (!Array.isArray(b) || b.length !== 4) return null;
    let v = b.map(Number);
    if (!v.every(isFinite)) return null;
    if (v.some(n => n > 1.5) && w && h) v = [v[0] / w, v[1] / h, v[2] / w, v[3] / h];   // pixels → fractions
    return v[2] > 0 && v[3] > 0 ? v.map(n => clamp(n, 0, 1)) : null;
  }
  function people(list, w, h) {
    return arr(list).map(q => (typeof q === 'string' ? { name: q } : q)).filter(q => q && q.name != null && q.name !== '')
      .map(q => ({ name: String(q.name), box: box(q.box || q.bbox, w, h) }));
  }
  function photo(p) {
    if (typeof p === 'string') p = { src: p };
    if (!p || typeof p !== 'object') return null;
    const src = url(p.src || p.path);
    if (!src) return null;
    const m = DIMS[src] || {}, w = m.w || +p.width || 0, h = m.h || +p.height || 0;
    return { src, thumb: url(p.thumb), w, h, bad: !!m.bad, known: w > 0 && h > 0,
      title: str(p.title), desc: str(p.description), taken: p.taken, source: str(p.source), people: people(p.people, w, h) };
  }
  const str = v => (v == null ? '' : String(v).trim());

  /* ── geometry: where a w×h photo sits in a bw×bh box ──
     fit = whole photo. fill = cover; the union of the person boxes (padded 35 % for hair and chins) is centred
     horizontally and put at 42 % of the box height, then clamped to the photo. safe = that union is not cut.
     No boxes: centre horizontally, take a third of the loss off the top and two thirds off the bottom. */
  function place(ph, bw, bh, mode) {
    const s = mode === 'fit' ? Math.min(bw / ph.w, bh / ph.h) : Math.max(bw / ph.w, bh / ph.h), w = ph.w * s, h = ph.h * s;
    let ox = (bw - w) / 2, oy = (bh - h) / 2, safe = true, ax = 50, ay = 42;
    const bb = ph.people.map(p => p.box).filter(Boolean);
    if (mode !== 'fit') {
      if (bb.length) {
        const x0 = Math.min(...bb.map(b => b[0])), y0 = Math.min(...bb.map(b => b[1]));
        const x1 = Math.max(...bb.map(b => b[0] + b[2])), y1 = Math.max(...bb.map(b => b[1] + b[3]));
        const mx = (x1 - x0) * .35, my = (y1 - y0) * .35, u = [clamp(x0 - mx, 0, 1), clamp(y0 - my, 0, 1), clamp(x1 + mx, 0, 1), clamp(y1 + my, 0, 1)];
        ox = clamp(bw / 2 - (u[0] + u[2]) / 2 * w, bw - w, 0); oy = clamp(bh * .42 - (u[1] + u[3]) / 2 * h, bh - h, 0);
        safe = u[0] * w + ox >= -.5 && u[2] * w + ox <= bw + .5 && u[1] * h + oy >= -.5 && u[3] * h + oy <= bh + .5;
        ax = (u[0] + u[2]) * 50; ay = (u[1] + u[3]) * 50;
      } else oy = (bh - h) / 3;
    }
    return { w, h, ox, oy, safe, ax, ay, loss: 1 - (Math.min(bw, w) * Math.min(bh, h)) / (w * h) };
  }
  function pick(list, b) {
    const a = list[0].w / list[0].h, want = b.layout;
    const two = list.length >= 2 && list[1].known && !list[1].bad;
    if (want === 'diptych' && two) return 'diptych';
    if (want === 'side' || want === 'fill' || want === 'fit') return want;
    if (two && truthy(b.pair) && a < 1 && list[1].w / list[1].h < 1) return 'diptych';
    if (a < 1.2) return 'side';
    const g = place(list[0], FW, FH, 'fill');
    return g.loss <= .25 && g.safe ? 'fill' : 'fit';
  }
  const truthy = v => v === true || v === 1 || v === 'true' || v === '1';

  function pic(ph, x, y, bw, bh, mode, cls, kb) {
    const g = place(ph, bw, bh, mode);
    return `<div class="pc-pic ${cls || ''}" style="left:${px(x)};top:${px(y)};width:${px(bw)};height:${px(bh)}">` +
      `<img src="${esc(ph.src)}" data-a="${(ph.w / ph.h).toFixed(4)}" alt="" style="left:${px(g.ox)};top:${px(g.oy)};width:${px(g.w)};height:${px(g.h)}` +
      `${kb ? `;transform-origin:${g.ax.toFixed(1)}% ${g.ay.toFixed(1)}%;--dwell:${kb}s` : ''}"></div>`;
  }
  const backdrop = ph => `<div class="pc-pic pc-bk${ph.thumb ? '' : ' pc-raw'}" style="left:0;top:0;width:${FW}px;height:${FH}px">` +
    `<img src="${esc(ph.thumb || ph.src)}" alt=""></div><div class="pc-shade"></div>`;

  /* ── small parts ── */
  const WX_ICON = { clear: 'sun', partly: 'partly', fog: 'fog', overcast: 'cloud', rain: 'rain', snow: 'snow' };
  const wxIcon = (b, ctx) => b.icon || (ctx.atmosphere.phase === 'night' && ctx.atmosphere.weather === 'clear' ? 'moon' : WX_ICON[ctx.atmosphere.weather] || '');
  const deg = t => (t == null || t === '' ? '' : (isFinite(+t) ? Math.round(+t) : String(t).replace(/\s*°.*$/, '')) + '°');
  const clock = () => `<span class="pc-clock">${esc(BB.fmtTime(BB.now(), { meridiem: false }))}</span>`;
  const who = name => `<span class="who ${BB.person(name)}"><i class="dot"></i>${esc(name)}</span>`;
  const whos = ph => { const seen = {}; return ph.people.filter(p => !seen[p.name] && (seen[p.name] = 1)).slice(0, 5).map(p => who(p.name)).join(''); };

  /* The sky as a pill, for layouts where the photo covers it: fill = the current sky gradient. */
  function chip(b, ctx, cls) {
    const P = ctx.P, sky = P.sky || [], ic = wxIcon(b, ctx), t = deg(b.temp);
    return `<div class="pc-chip ${cls || ''}" style="background:linear-gradient(180deg,${sky[1] || '#555'} 0%,${sky[2] || '#666'} 55%,${sky[3] || '#777'} 100%);color:${P.vars['--text']}">` +
      `${ic ? ico(ic) : ''}${clock()}${t ? `<span class="t">${esc(t)}</span>` : ''}</div>`;
  }
  const nowCol = (b, ctx, slim) => {
    const t = deg(b.temp), c = slim ? '' : str(b.condition), ic = t || c ? wxIcon(b, ctx) : '';   // no weather data: the sky beside it says it
    return `<div class="pc-now sky-ink"><div class="t d">${clock()}</div>` +
      (t || c ? `<div class="w">${ic ? ico(ic) : ''}${t ? `<b>${esc(t)}</b>` : ''}${c ? `<span>${esc(c)}</span>` : ''}</div>` : '') + '</div>';
  };

  const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  /* "Saturday, November 14" this year, "November 14, 2024" otherwise; short drops the weekday. Not a date → as given. */
  function when(v, short) {
    if (v == null || v === '') return '';
    const d = new Date(v);
    if (isNaN(d)) return typeof v === 'string' && v.length <= 28 ? v : '';
    const md = `${MONTHS[d.getMonth()]} ${d.getDate()}`;
    if (d.getFullYear() !== BB.now().getFullYear()) return `${md}, ${d.getFullYear()}`;
    return short ? md : `${DAYS[d.getDay()]}, ${md}`;
  }
  /* The DB stores the channel, not the sender. Anything else is shown as "from <source>". */
  const SOURCES = { signal: ['phone', 'via Signal'], whatsapp: ['phone', 'via WhatsApp'], camera: ['lens', 'Box camera'], box: ['lens', 'Box camera'] };
  function source(s) {
    if (!s) return '';
    const k = SOURCES[s.toLowerCase()];
    return `<span class="pc-src">${ico(k ? k[0] : 'phone')}${k ? k[1] : 'from ' + esc(s)}</span>`;
  }
  function position(v) {
    if (typeof v === 'string') { const m = v.match(/^\s*(\d+)\s*(?:\/|of)\s*(\d+)\s*$/); v = m ? { index: +m[1], count: +m[2] } : null; }
    const of = v ? Math.floor(+v.count) : 0, n = v ? clamp(Math.floor(+v.index) || 1, 1, of) : 0;
    if (!(of >= 2)) return '';
    const marks = of <= 12 ? `<span class="pc-pips">${Array.from({ length: of }, (_, i) => `<i${i + 1 === n ? ' class="on"' : ''}></i>`).join('')}</span>`
      : `<span class="pc-bar"><i style="width:${(100 * n / of).toFixed(1)}%"></i></span>`;
    return `<div class="pc-pos">${marks}<span class="pc-num">${ico('knob')}<span><b>${n}</b> of ${of}</span></span></div>`;
  }
  /* reason / title (or the tagger's sentence, clamped) / people, date, source */
  function caption(ph, reason, cls) {
    const meta = whos(ph) + (when(ph.taken) ? `<span>${esc(when(ph.taken))}</span>` : '') + source(ph.source);
    const head = ph.title ? `<h2 class="d">${esc(ph.title)}</h2>` : ph.desc ? `<p class="pc-raw-t">${esc(ph.desc)}</p>` : '';
    if (!reason && !head && !meta) return '';
    return `<div class="pc-cap ${cls || ''}">${reason ? `<div class="pc-why">${ico('spark')}<span>${esc(reason)}</span></div>` : ''}${head}` +
      `${meta ? `<div class="pc-meta">${meta}</div>` : ''}</div>`;
  }

  /* ── shell ── */
  const lightPeople = Object.entries(PEOPLE.light).map(([k, c]) => `--${k}:${c}`).join(';');
  const minimalAfter = (b, dflt) => (b.ui === 'minimal' ? 0 : b.minimal_after_s == null || b.minimal_after_s === '' ? dflt : Math.max(0, +b.minimal_after_s || 0));
  function shell(b, cls, html, after) {
    const id = ++uid; LIVE[id] = b;
    return `<div class="picture ${cls}${b.ui === 'minimal' ? ' min' : ''}" data-pid="${id}" data-min="${after || 0}" style="${lightPeople}">${html}</div>`;
  }
  /* The stage is the photo layer. If a different photo was on screen a moment ago, it stays on top and fades out. */
  function stage(key, html, hold) {
    const old = last && last.key !== key && !BB.still && document.querySelector('#view .picture .pc-stage') ? last.html : '';
    if (!hold) last = { key, html };
    return `<div class="pc-stage">${html}</div>${old ? `<div class="pc-prev"${hold ? ' style="animation:none"' : ''}>${old}</div>` : ''}`;
  }
  const none = (b, ctx, msg) => shell(b, 'pc-side pc-empty' + (hidden(ctx) ? ' pc-nosky' : ''),
    `<div class="pc-col" style="left:0"><div class="pc-gscrim"></div>${nowCol(b, ctx)}<div class="pc-none">${ico('lens')}<span>${esc(msg)}</span></div></div>`);
  const hidden = ctx => !!(ctx.state && ctx.state.scene && ctx.state.scene.hidden);
  /* Size unknown: load the image off-screen, keep whatever was showing, redraw when it has been measured. */
  const pending = (b, list) => shell(b, 'pc-pending', stage('', '', true) + list.map(p => `<img class="pc-probe" src="${esc(p.src)}" alt="">`).join(''));

  /* ── picture ── */
  BB.define('picture', (b, ctx) => {
    const list = arr(typeof b.photos === 'string' || (b.photos && !Array.isArray(b.photos)) ? [b.photos] : b.photos).map(photo).filter(Boolean);
    if (!list.length) return none(b, ctx, 'No photo to show');
    const a = list[0], pairing = list.length >= 2 && (b.layout === 'diptych' || truthy(b.pair));
    if (a.bad) return none(b, ctx, 'This photo can’t be shown');
    const need = [a].concat(pairing ? [list[1]] : []).filter(p => !p.known && !p.bad);
    if (need.length) return pending(b, need);

    const layout = pick(list, b), reason = str(b.reason), pos = position(b.position), after = minimalAfter(b, 6);
    const kb = !BB.still && +b.dwell_s > 0 ? +b.dwell_s : 0;

    if (layout === 'diptych') {
      const c = list[1];
      let wa = FH * a.w / a.h, wb = FH * c.w / c.h;
      if (wa + wb + GAP > FW) { const k = (FW - GAP) / (wa + wb); wa *= k; wb *= k; }
      wa = Math.round(wa); wb = Math.round(wb);
      const rest = FW - wa - wb - GAP, dock = rest >= 200, x0 = dock ? 0 : Math.max(0, Math.round(rest / 2));
      const half = (ph, x, w) => {
        const head = ph.title ? `<h3 class="d">${esc(ph.title)}</h3>` : ph.desc ? `<p class="pc-raw-t">${esc(ph.desc)}</p>` : '';
        const meta = whos(ph) + (when(ph.taken, true) ? `<span>${esc(when(ph.taken, true))}</span>` : '');
        return head || meta ? `<div class="pc-half" style="left:${x}px;width:${w}px"><div class="pc-hs"></div><div class="pc-hc">${head}${meta ? `<div class="pc-meta">${meta}</div>` : ''}</div></div>` : '';
      };
      const why = reason ? `<div class="pc-why">${ico('spark')}<span>${esc(reason)}</span></div>` : '';
      const st = stage(`dip|${a.src}|${c.src}`, pic(a, x0, 0, wa, FH, 'fill', 'pc-edge') + pic(c, x0 + wa + GAP, 0, wb, FH, 'fill', 'pc-edge'));
      return shell(b, `pc-dip ${dock ? 'pc-dock' : 'pc-ctr'}${dock && hidden(ctx) ? ' pc-nosky' : ''}`, st + half(a, x0, wa) + half(c, x0 + wa + GAP, wb) +
        (dock ? `<div class="pc-col slim${/^(noon|afternoon|night)$/.test(ctx.atmosphere.phase) ? ' low' : ''}" style="left:${wa + wb + GAP}px"><div class="pc-gscrim"></div>${nowCol(b, ctx, true)}<div class="pc-ground">${why}${pos}</div></div>`
          : `<div class="pc-top l"></div>${chip(b, ctx)}${pos.replace('class="pc-pos"', `class="pc-pos" style="left:${x0 + 26}px"`)}`), after);
    }

    if (layout === 'side') {
      const pw = Math.min(600, Math.round(FH * a.w / a.h));
      const cap = caption(a, reason, pos ? 'has-pos' : '');
      return shell(b, `pc-side${hidden(ctx) ? ' pc-nosky' : ''}`, stage(`side|${a.src}`, pic(a, 0, 0, pw, FH, 'fill', 'pc-edge')) +
        `<div class="pc-col${FW - pw < 480 ? ' narrow' : ''}" style="left:${pw}px"><div class="pc-gscrim"></div>${nowCol(b, ctx)}` +
        `<div class="pc-ground">${cap}${pos}</div></div>`, after);
    }

    let st;
    if (layout === 'fill') st = stage(`fill|${a.src}`, pic(a, 0, 0, FW, FH, 'fill', kb ? 'pc-kb' : '', kb));
    else { const g = place(a, FW, FH, 'fit'); st = stage(`fit|${a.src}`, backdrop(a) + pic(a, Math.round(g.ox), Math.round(g.oy), Math.round(g.w), Math.round(g.h), 'fit', 'pc-lift')); }
    const cap = caption(a, reason, pos ? '' : 'wide');
    return shell(b, `pc-over pc-${layout}`, `${st}${cap ? '<div class="pc-scrim"></div>' : ''}<div class="pc-top"></div>${chip(b, ctx)}${cap}${pos}`, after);
  });

  /* ── photo_caption: the caption alone, bottom-left over whatever is behind it ── */
  BB.define('photo_caption', (b, ctx) => {
    const p = b.photo && typeof b.photo === 'object' ? b.photo : {};
    const ph = { title: str(b.title != null ? b.title : p.title), desc: str(b.description != null ? b.description : p.description),
      taken: b.taken != null ? b.taken : p.taken, source: str(b.source != null ? b.source : p.source), people: people(b.people != null ? b.people : p.people) };
    const pos = position(b.position), cap = caption(ph, str(b.reason), pos ? '' : 'wide');
    if (!cap && !pos) return '';
    return shell(b, 'pc-over pc-solo', `<div class="pc-scrim"></div>${cap}${pos}`, minimalAfter(b, 6));
  });

  /* ── sky_chip: time + weather pill, top-right (corner:"top-left" for the other side) ── */
  BB.define('sky_chip', (b, ctx) => chip(b, ctx, b.corner === 'top-left' ? 'l' : ''));

  /* ── capture: "Here's what I see" ── */
  const stamp = d => { const s = String(d.getSeconds()).padStart(2, '0'); return BB.fmtTime(d).replace(/^(\d+:\d+)/, `$1:${s}`); };
  const ago = (at, now) => { const m = Math.floor((now - at) / 60000); return m < 1 ? 'Just now' : m < 60 ? `${m} min ago` : `${Math.floor(m / 60)} h ago`; };
  BB.define('capture', (b, ctx) => {
    const ph = photo(b.image);
    if (!ph) return none(b, ctx, 'No camera frame to show');
    if (ph.bad) return none(b, ctx, 'The camera frame can’t be shown');
    if (!ph.known) return pending(b, [ph]);
    if (b.people != null && b.people !== '') ph.people = people(b.people, ph.w, ph.h);
    const g = place(ph, FW, FH, 'fill'), say = str(b.say), saved = truthy(b.saved);
    let at = new Date(b.at || ph.taken || NaN); if (isNaN(at)) at = BB.now();
    const tags = ph.people.filter(q => q.box).map(q => {
      const x = q.box[0] * g.w + g.ox, w = q.box[2] * g.w, y = clamp(q.box[1] * g.h + g.oy + q.box[3] * g.h / 2 - 16, 78, 400);
      const left = x + w + 22 > FW - 170;
      return `<div class="pc-tag ${BB.person(q.name)}${left ? ' l' : ''}" style="left:${(left ? x - 22 : x + w + 22).toFixed(0)}px;top:${y.toFixed(0)}px"><i class="dot"></i>${esc(q.name)}</div>`;
    }).join('');
    const loose = ph.people.filter(q => !q.box).slice(0, 5).map(q => who(q.name)).join('');
    return shell(b, 'pc-capt', `${stage(`capt|${ph.src}`, pic(ph, 0, 0, FW, FH, 'fill'))}<div class="pc-scrim"></div><div class="pc-top l"></div>
      <i class="pc-vf tl"></i><i class="pc-vf tr"></i><i class="pc-vf bl"></i><i class="pc-vf br"></i>
      <div class="pc-live" data-at="${at.getTime()}"><i class="pc-rec"></i><b>${esc(ago(at, BB.now()))}</b><span>${esc(stamp(at))}</span><span class="sep"></span>${ico('lens')}<span>Box camera</span></div>
      ${chip(b, ctx)}${tags}
      ${say || loose ? `<div class="pc-say"><div class="pc-lbl">Here’s what I see</div>${say ? `<p>${esc(say)}</p>` : ''}${loose ? `<div class="pc-meta">${loose}</div>` : ''}</div>` : ''}
      <div class="pc-keep">${saved ? `<b>${ico('check')}Saved to photos</b>` : 'Not saved<span>say “keep it” to add it to photos</span>'}</div>
      ${BB.still ? '' : '<i class="pc-flash"></i>'}`, minimalAfter(b, 0));
  });

  /* ── after insert: measure images, arm the minimal-state timer ── */
  function redraw(el) {
    const b = LIVE[el.dataset.pid];
    if (!b || !el.isConnected) return;
    const t = document.createElement('div');
    t.innerHTML = BB.render(b, BB.ctx);
    const n = t.firstElementChild;
    if (n) { el.replaceWith(n); wire(n); } else el.remove();
  }
  function wire(root) {
    const els = root.matches && root.matches('.picture[data-pid]') ? [root] : [...root.querySelectorAll('.picture[data-pid]')];
    els.forEach(el => {
      el.querySelectorAll('img[data-a], img.pc-probe').forEach(img => {
        if (img.closest('.pc-prev')) return;
        const key = img.getAttribute('src');
        const fail = () => { DIMS[key] = { bad: true }; redraw(el); };
        const ok = () => {
          if (!img.naturalWidth || !img.naturalHeight) return fail();
          const a = +img.dataset.a, n = img.naturalWidth / img.naturalHeight;
          if (!a || Math.abs(a / n - 1) > .03) { DIMS[key] = { w: img.naturalWidth, h: img.naturalHeight }; redraw(el); }
        };
        if (img.complete) ok();
        else { img.addEventListener('load', ok, { once: true }); img.addEventListener('error', fail, { once: true }); }
      });
      const s = +el.dataset.min;
      if (s > 0 && !el.classList.contains('min')) setTimeout(() => { if (el.isConnected) el.classList.add('min'); }, s * 1000);
    });
  }
  BB.onMount(root => {
    Object.keys(LIVE).forEach(id => { if (!document.querySelector(`.picture[data-pid="${id}"]`)) delete LIVE[id]; });
    wire(root);
  });
  BB.onTick((d, root) => {
    const t = BB.fmtTime(d, { meridiem: false });
    document.querySelectorAll('.pc-clock').forEach(e => { if (e.textContent !== t) e.textContent = t; });
    root.querySelectorAll('.pc-live[data-at]').forEach(e => { const l = ago(+e.dataset.at, d), b = e.querySelector('b'); if (b && b.textContent !== l) b.textContent = l; });
  });
})();
