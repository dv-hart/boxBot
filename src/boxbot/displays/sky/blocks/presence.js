/* Presence: the voice-state dock, and the agent's status pill. Both sit above the view, so they work over
   any display, a photo or a pushed camera frame.

   BB.presence(el, p, ctx) — el is the #presence layer; p is null (sink away) or
     { state: listening | thinking | speaking | timer | notice | greeting,
       who,        household name → speaker dot (greeting: tints the dome)
       text,       the ONE line (transcript, caption, notice; timer: its name)
       detail,     quiet words after the kicker (timer: remaining time when `until` is absent)
       label,      optional: replaces the state word in the kicker
       progress,   timer: fraction remaining 0..1
       until, duration }   timer, optional: ISO end time + total seconds → the page ticks the pill and the line

   Light leaks up from the bottom edge; colour, shape and motion differ per state so it reads from across
   the room. A running timer is not a conversation: draining line + pill, no dark base.
   Motion is transform/opacity on SVG that is blurred once and rasterised as its own layer — never on the
   shapes inside it. */
(function () {
  'use strict';
  const esc = BB.esc;
  const str = v => (v == null ? '' : String(v));
  const has = v => v != null && v !== '';
  const num = (v, d) => { const n = typeof v === 'number' ? v : parseFloat(v); return isFinite(n) ? n : d; };
  const f1 = v => (+v).toFixed(1);

  ICONS['pz-timer'] = '<circle cx="12" cy="13.5" r="7.5"/><path d="M12 9.5v4l2.6 1.6M9.5 3h5M12 3v3"/>';
  ICONS['pz-bell'] = '<path d="M6.5 16.5v-5a5.5 5.5 0 0 1 11 0v5l1.5 2h-14zM10 20.5a2.1 2.1 0 0 0 4 0"/>';
  ICONS['pz-door'] = '<path d="M4 11.2L12 4.5l8 6.7M6.3 9.8V19.5h11.4V9.8M10.3 19.5v-5h3.4v5"/>';

  /* state → [low colour, high colour] of the light, kicker word, kicker icon */
  const STATES = {
    listening: { c: ['#58ffc0', '#8f74ff'], word: 'Listening' },
    thinking: { c: ['#cdbfff', '#7f6bff'], word: 'Thinking' },
    speaking: { c: ['#ffc65c', '#ff8f45'], word: 'boxBot' },
    timer: { c: ['#fff0d2', '#fff0d2'], word: 'Timer', icon: 'pz-timer' },
    notice: { c: ['#ff5f47', '#ff9a6b'], word: 'Notice', icon: 'pz-bell' },
    greeting: { c: ['#ff8fb4', '#ffd0a0'], word: 'Arrived', icon: 'pz-door' },
  };
  const B = 200, TILE = 1024, TAU = Math.PI * 2;
  let uid = 0;

  function rand(seed) { let s = seed >>> 0; return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296; }
  function defs(id, c) {
    return `<linearGradient id="${id}g" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="${c[0]}"/><stop offset=".4" stop-color="${c[0]}" stop-opacity=".62"/><stop offset="1" stop-color="${c[1]}" stop-opacity="0"/></linearGradient>
      <filter id="${id}b" x="-10%" y="-60%" width="120%" height="220%"><feGaussianBlur stdDeviation="14"/></filter>
      <filter id="${id}m" x="-10%" y="-60%" width="120%" height="220%"><feGaussianBlur stdDeviation="5"/></filter>
      <filter id="${id}s" x="-10%" y="-30%" width="120%" height="160%"><feGaussianBlur stdDeviation="1.6"/></filter>`;
  }
  const svg = (w, cls, body, id, c) => `<svg class="${cls}" width="${w}" height="${B}" viewBox="0 0 ${w} ${B}"><defs>${defs(id, c)}</defs>${body}</svg>`;
  const glow = (id, c, a, h = 26) => `<svg class="pz-glow" width="1024" height="${B}" viewBox="0 0 1024 ${B}"><defs>${defs(id, c)}</defs><rect x="-20" y="${B - h / 2}" width="1064" height="${h}" fill="${c[0]}" opacity="${a}" filter="url(#${id}b)"/></svg>`;

  /* Each builder returns layers; a `.pz-drift` layer is two identical 1024 tiles wide and loops seamlessly. */
  const LIGHT = {
    listening(c) {   // aurora curtain of vertical rays, drifting left and breathing
      const id = 'pz' + (uid++), r = rand(11), rays = [], top = [];
      const env = x => .46 + .54 * Math.pow(.5 + .5 * Math.sin(TAU * 2 * x / TILE + 2.2), 1.4) * (.75 + .25 * Math.sin(TAU * 5 * x / TILE));
      for (let x = 0; x < TILE; x += 8) rays.push([x, 150 * env(x) * (.8 + .3 * r()), 2.5 + 4.5 * r(), .25 + .5 * r()]);
      let body = '', d = `M-10,${B}`;
      for (let t = 0; t < 2; t++) rays.forEach(([x, h, w, o]) => {
        const X = x + t * TILE; top.push(`${X},${(B - h * .9).toFixed(0)}`);
        body += `<rect x="${f1(X - w / 2)}" y="${(B - h).toFixed(0)}" width="${f1(w)}" height="${h.toFixed(0)}" rx="${f1(w / 2)}" opacity="${o.toFixed(2)}"/>`;
      });
      d += 'L' + top.join('L') + `L${TILE * 2 + 10},${B}Z`;
      return `<div class="pz-breathe"><div class="pz-drift pz-d1">${svg(TILE * 2, '', `<path d="${d}" fill="url(#${id}g)" filter="url(#${id}b)" opacity=".85"/><g fill="url(#${id}g)" filter="url(#${id}s)">${body}</g>`, id, c)}</div></div>` + glow(id + 'x', c, .9);
    },
    speaking(c) {    // three warm swells sliding at three speeds
      const wave = (A, k, k2, p, o, f, cls) => {
        const id = 'pz' + (uid++), pts = [];
        for (let x = -10; x <= TILE * 2 + 10; x += 8) pts.push(x + ',' + f1(B - 12 - A * (.5 + .5 * Math.sin(TAU * k * x / TILE + p)) * (.72 + .28 * Math.sin(TAU * k2 * x / TILE + p * 2))));
        return `<div class="pz-swell ${cls}"><div class="pz-drift ${cls}">${svg(TILE * 2, '', `<path d="M-10,${B}L${pts.join('L')}L${TILE * 2 + 10},${B}Z" fill="url(#${id}g)" opacity="${o}" filter="url(#${id}${f})"/>`, id, c)}</div></div>`;
      };
      return wave(118, 1, 2, .6, .5, 'm', 'pz-w1') + wave(84, 2, 3, 2.4, .6, 's', 'pz-w2') + wave(52, 3, 1, 4.4, .95, 's', 'pz-w3') + glow('pz' + (uid++), c, .8);
    },
    thinking(c) {    // a thin line of starlight and one lantern crossing it
      const id = 'pz' + (uid++), cx = 600, W = 780;
      let s = `<linearGradient id="${id}t" gradientUnits="userSpaceOnUse" x1="${cx - 430}" y1="0" x2="${cx}" y2="0"><stop offset="0" stop-color="${c[1]}" stop-opacity="0"/><stop offset="1" stop-color="${c[0]}" stop-opacity=".95"/></linearGradient>
        <rect x="${cx - 430}" y="${B - 26}" width="430" height="24" rx="12" fill="url(#${id}t)" filter="url(#${id}m)"/>
        <ellipse cx="${cx}" cy="${B - 10}" rx="150" ry="58" fill="${c[1]}" opacity=".7" filter="url(#${id}b)"/>
        <ellipse cx="${cx}" cy="${B - 11}" rx="36" ry="10" fill="#fff" filter="url(#${id}m)"/>`;
      [[-92, 44, 5], [-178, 30, 3.4], [-270, 22, 2.6], [-36, 70, 3]].forEach(([dx, dy, k]) => {
        const x = cx + dx, y = B - dy;
        s += `<path d="M${x},${y - k * 2}L${x + k * .36},${y - k * .36}L${x + k * 2},${y}L${x + k * .36},${y + k * .36}L${x},${y + k * 2}L${x - k * .36},${y + k * .36}L${x - k * 2},${y}L${x - k * .36},${y - k * .36}Z" fill="#fff" opacity=".9"/>`;
      });
      return `<i class="pz-edge" style="background:${c[0]}"></i><div class="pz-lantern">${svg(W, '', s, id, c)}</div>`;
    },
    notice(c) {      // ember: a level band that swells twice, then holds
      const id = 'pz' + (uid++);
      return `<div class="pz-ember">${svg(1024, '', `<ellipse cx="512" cy="${B + 6}" rx="700" ry="92" fill="${c[0]}" opacity=".62" filter="url(#${id}b)"/><ellipse cx="512" cy="${B + 4}" rx="560" ry="40" fill="${c[1]}" opacity=".8" filter="url(#${id}b)"/>`, id, c)}</div><i class="pz-edge pz-edge-hot" style="background:${c[1]}"></i>`;
    },
    greeting(c) {    // a small sunrise in the person's colour, rings opening once
      const id = 'pz' + (uid++);
      const dome = `<ellipse cx="512" cy="${B + 40}" rx="520" ry="140" fill="currentColor" opacity=".5" filter="url(#${id}b)"/><ellipse cx="512" cy="${B + 30}" rx="300" ry="90" fill="${c[1]}" opacity=".78" filter="url(#${id}b)"/><ellipse cx="512" cy="${B + 14}" rx="120" ry="30" fill="#fff" opacity=".5" filter="url(#${id}b)"/>`;
      const rings = [150, 210, 280].map((rr, i) => `<path d="M${512 - rr},${B}A${rr},${rr * .62} 0 0 1 ${512 + rr},${B}" fill="none" stroke="${c[1]}" stroke-width="2" opacity="${.6 - i * .17}"/>`).join('');
      return `<div class="pz-dome">${svg(1024, '', dome, id, c)}</div><div class="pz-rings">${svg(1024, '', rings, id + 'r', c)}</div>`;
    },
  };

  /* ── parts ── */
  const TL_W = 916;   // timer line length (1024 − 2 × 54)
  function clockLeft(until, now) {
    const t = new Date(until); if (!has(until) || isNaN(t)) return null;
    const s = Math.max(0, Math.round((t - now) / 1000)), h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), p2 = n => String(n).padStart(2, '0');
    return { s, text: h ? `${h}:${p2(m)}:${p2(s % 60)}` : `${m}:${p2(s % 60)}` };
  }
  function timerNow(p) {
    const left = clockLeft(p.until, BB.now()), dur = num(p.duration, 0);
    let f = left && dur > 0 ? left.s / dur : num(p.progress, 1);
    f = Math.max(0, Math.min(1, f));
    return { f, text: left ? left.text : str(p.detail), last: left ? left.s <= 60 : false };
  }
  function kicker(st, p) {
    const S = STATES[st], who = str(p.who).trim();
    return `<div class="pz-k"><i class="pz-bead"></i>${S.icon ? ico(S.icon) : ''}<span class="pz-word">${esc(has(p.label) ? p.label : S.word)}</span>${
      who && st !== 'greeting' ? `<span class="who ${BB.person(who)}"><i class="dot"></i>${esc(who)}</span>` : ''}${
      has(p.detail) ? `<span class="pz-detail">${esc(p.detail)}</span>` : ''}</div>`;
  }
  function line(st, p) {
    const t = str(p.text).trim();
    return kicker(st, p) + `<div class="pz-one d"><span>${t ? (st === 'listening' ? '“' + esc(t) + '”' : esc(t)) : ''}</span></div>`;
  }
  function timerParts(p) {
    const t = timerNow(p);
    return { t, pill: `${ico('pz-timer')}<b class="d num">${esc(t.text)}</b>${has(p.text) ? `<span>${esc(p.text)}</span>` : ''}` };
  }
  function setTimer(node, p) {
    const { t, pill } = timerParts(p), q = s => node.querySelector(s);
    if (q('.pz-pill').dataset.k !== pill) { q('.pz-pill').innerHTML = pill; q('.pz-pill').dataset.k = pill; }
    q('.pz-tl-fill').style.transform = `scaleX(${t.f.toFixed(4)})`;
    q('.pz-tl-bead').style.transform = `translateX(${(TL_W * t.f).toFixed(1)}px)`;
    node.classList.toggle('pz-last', t.last);
  }

  function build(p) {
    const st = p.state, S = STATES[st], node = document.createElement('div');
    node.dataset.state = st;
    if (st !== 'greeting') node.style.setProperty('--b', S.c[0]);   // greeting: the person's colour (presence.css)
    if (st === 'timer') {
      node.className = 'pz pz-timer pz-quiet';
      node.innerHTML = '<div class="pz-tl"><i class="pz-tl-track"></i><i class="pz-tl-fill"></i><i class="pz-tl-bead"><i></i></i></div><div class="pz-pill"></div>';
      setTimer(node, p);
    } else {
      node.className = `pz pz-${st} ${st === 'greeting' ? BB.person(p.who) : ''}`;
      node.innerHTML = `<div class="pz-scrim"></div><div class="pz-light">${LIGHT[st](S.c)}</div><div class="pz-line">${line(st, p)}</div>`;
    }
    if (BB.still) node.classList.add('pz-still');
    return node;
  }

  /* One line. A transcript that outgrows it keeps its newest words; anything else drops a size and may take two. */
  function fit(node) {
    const one = node.querySelector('.pz-one'), span = one && one.firstElementChild; if (!span) return;
    one.classList.remove('pz-tail', 'pz-two');
    if (span.scrollWidth <= span.clientWidth + 1) return;
    one.classList.add(node.dataset.state === 'listening' ? 'pz-tail' : 'pz-two');
  }

  let current = null, payload = null;

  BB.presence = function (el, p) {
    if (p && !STATES[p.state]) p = null;
    payload = p;
    el.classList.toggle('pz-pilled', !!p && p.state === 'timer');
    if (!p) {                                             // exit: 600 ms sink
      const old = current; current = null;
      if (!old) { el.replaceChildren(); return; }
      if (BB.still) { old.remove(); return; }
      old.classList.add('pz-exit');
      setTimeout(() => old.remove(), 640);
      return;
    }
    if (current && current.dataset.state === p.state && current.isConnected) {   // same state: new words only
      if (p.state === 'timer') setTimer(current, p);
      else {
        current.querySelector('.pz-line').innerHTML = line(p.state, p);
        if (p.state === 'greeting') current.className = current.className.replace(/\bp-\S+/g, '') + ' ' + BB.person(p.who);
        fit(current);
      }
      return;
    }
    const old = current, node = build(p); current = node;
    [...el.children].forEach(c => { if (c !== old) c.remove(); });               // anything still sinking
    if (BB.still) { el.replaceChildren(node); fit(node); return; }
    node.classList.add(old ? 'pz-fade' : 'pz-enter');
    el.appendChild(node); fit(node);
    void node.offsetWidth;                                // commit the start state
    node.classList.remove('pz-fade', 'pz-enter');
    if (old) {                                            // state change: 250 ms cross-fade
      if (p.state === 'timer' || old.dataset.state === 'timer') old.classList.add('pz-fade');   // no shared base to hold
      setTimeout(() => old.remove(), 270);
    }
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { if (current === node) fit(node); });
  };

  BB.onTick(() => {
    if (current && payload && payload.state === 'timer' && has(payload.until) && current.isConnected) setTimer(current, payload);
  });

  /* Status pill: what the agent is doing right now ("Searching the web…"). Top-right, its own dark ground,
     so it reads over any scene or photo. Steps down a row while the timer pill holds the corner. */
  BB.define('status_pill', b => {
    const t = str(b.text).trim();
    return t ? `<div class="sp${BB.still ? ' sp-still' : ''}"><i class="sp-dots"><i></i><i></i><i></i></i><span>${esc(t)}</span></div>` : '';
  });
})();
