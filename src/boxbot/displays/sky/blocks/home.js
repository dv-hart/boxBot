/* Living Sky — home blocks: zones, zone, sky_clock, now, next_event, house.

   Also holds two helpers that blocks/calendar.js builds on (index.html loads this file first):
     BB.live(type, fn, post)  a block whose output depends on the clock. Re-rendered on BB.onTick and
                              swapped only when its HTML changes, so countdowns, now markers and "past"
                              states move without a new push.
     BB.cal                   event normalising and date helpers.

   Blocks take data as params (bindings are resolved in Python) and derive today / next / tomorrow here
   from ISO start times and the page clock. Every value from data goes through BB.esc(). */
(function () {
  'use strict';
  const esc = BB.esc;
  const DOW = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const MONTH = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

  /* ── dates ──────────────────────────────────────────────────── */
  const sod = d => new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const addDays = (d, n) => new Date(d.getFullYear(), d.getMonth(), d.getDate() + n);
  const dayDiff = (a, b) => Math.round((sod(a) - sod(b)) / 864e5);
  const dow3 = d => DOW[d.getDay()].slice(0, 3);
  const hm = d => BB.fmtTime(d, { meridiem: false });
  const DATE_ONLY = /^(\d{4})-(\d{2})-(\d{2})$/;

  /* ISO datetime, date-only ("2026-11-24" = local midnight, not UTC), or a clock time ("5:10 PM", "17:10")
     on the day of `base`. Anything else → null. */
  function when(v, base) {
    if (v == null || v === '') return null;
    if (v instanceof Date) return isNaN(v) ? null : v;
    const s = String(v).trim();
    let m = DATE_ONLY.exec(s);
    if (m) return new Date(+m[1], m[2] - 1, +m[3]);
    m = /^(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?$/i.exec(s);
    if (m) {
      if (!base) return null;
      return new Date(base.getFullYear(), base.getMonth(), base.getDate(), (+m[1] % 12) + (/p/i.test(m[3]) ? 12 : 0), +(m[2] || 0));
    }
    m = /^(\d{1,2}):(\d{2})$/.exec(s);
    if (m) return base ? new Date(base.getFullYear(), base.getMonth(), base.getDate(), +m[1], +m[2]) : null;
    const d = new Date(s);
    return isNaN(d) ? null : d;
  }

  /* ── events ─────────────────────────────────────────────────── */
  /* One calendar event → the shape every block here uses. Only `start` is required; person, people,
     short_title, leave_by, done and kind are optional extras the agent may add. */
  function norm(e) {
    if (!e || typeof e !== 'object') return null;
    const start = when(e.start);
    if (!start) return null;
    const allDay = !!e.all_day || DATE_ONLY.test(String(e.start).trim());
    let end = when(e.end, start);
    if (!end || end < start) end = allDay ? addDays(start, 1) : new Date(+start + 30 * 6e4);
    /* all-day ends are exclusive (Google): the 26th–29th arrives as end = the 30th */
    const lastDay = allDay ? sod(end > start ? new Date(end - 1) : start) : sod(end > start ? new Date(end - 1) : end);
    const people = Array.isArray(e.people) && e.people.length ? e.people
      : (e.person ? String(e.person).split(/\s*(?:\+|,|&|\band\b)\s*/i) : []);
    return {
      id: e.id, title: e.title || e.summary || 'Busy', short: e.short_title || '', start, end, allDay,
      firstDay: sod(start), lastDay, multi: +lastDay > +sod(start), loc: e.location || '', duration: e.duration || '',
      people: people.map(p => String(p).trim()).filter(Boolean), leave: when(e.leave_by, start),
      done: !!e.done, kind: e.kind || '',
    };
  }
  const events = list => (Array.isArray(list) ? list : []).map(norm).filter(Boolean).sort((a, b) => a.start - b.start);
  const status = (e, now) => (e.allDay ? '' : e.done || e.end <= now ? 'past' : e.start <= now ? 'live' : '');
  const onDay = (e, day) => +e.firstDay <= +day && +e.lastDay >= +day;
  /* the event to point at: whatever is running, else the first one still to come */
  function nextOf(evs, now) {
    const timed = evs.filter(e => !e.allDay && !e.done);
    return timed.find(e => e.start <= now && e.end > now) || timed.find(e => e.start > now) || null;
  }

  /* ── people ─────────────────────────────────────────────────── */
  const isFamily = n => /^(family|everyone|all)$/i.test(String(n).trim());
  const whoName = n => (isFamily(n) ? 'Everyone' : String(n));
  const whoText = people => people.map(whoName).join(' + ');
  const pvar = cls => (cls === 'p-family' ? 'var(--family)' : `var(--${cls.slice(2)})`);
  /* class + inline style for one or two owners; a second owner becomes --q (split dots, gradient bars) */
  function owners(people) {
    const cls = (people || []).map(p => BB.person(p)).filter(Boolean);
    return { cls: cls[0] || 'p-none', st: cls[1] && cls[1] !== cls[0] ? `--q:${pvar(cls[1])}` : '' };
  }

  /* ── weather ────────────────────────────────────────────────── */
  const deg = v => { const s = String(v == null ? '' : v).replace(/[°\s]|deg.*$/gi, ''); return s === '' || isNaN(+s) ? '' : String(Math.round(+s)) + '°'; };
  /* forecast row for a date: rows carry day names ("Tue"), sometimes "Today" / "Tonight" */
  function forecastFor(forecast, date, now) {
    if (!Array.isArray(forecast)) return null;
    const key = dow3(date).toLowerCase(), today = dayDiff(date, now) === 0;
    return forecast.find(f => f && typeof f.day === 'string' &&
      (f.day.trim().toLowerCase().slice(0, 3) === key || (today && /^(today|tonight|this )/i.test(f.day.trim())))) || null;
  }

  /* ── live blocks ────────────────────────────────────────────── */
  const reg = new Map();
  let seq = 0;
  /* fn(block, ctx, mem) → html. mem persists for the life of the instance; mem.trim counts how many times
     the block has been asked to shed content because its [data-fit] box overflowed. */
  function live(type, fn, post) {
    BB.define(type, (b, ctx) => {
      const id = 'lv' + (++seq), mem = { trim: 0 }, html = fn(b, ctx, mem) || '';
      reg.set(id, { b, ctx, fn, post, mem, html });
      return `<div class="bb-live" data-live="${id}">${html}</div>`;
    });
  }
  function fit(el, r, c) {
    for (let i = 0; i < 24; i++) {
      const box = el.querySelector('[data-fit]');
      if (!box || box.scrollHeight <= box.clientHeight + 1) return;
      r.mem.trim++;
      const html = r.fn(r.b, c, r.mem) || '';
      if (html === r.html) { r.mem.trim--; return; }          // nothing left to shed
      r.html = html; el.innerHTML = html;
    }
  }
  BB.onTick((date, root) => {
    const seen = new Set();
    root.querySelectorAll('[data-live]').forEach(el => {
      const id = el.dataset.live, r = reg.get(id);
      if (!r) return;
      seen.add(id);
      const c = { ...r.ctx, now: date };
      let html = r.fn(r.b, c, r.mem) || '';
      if (html !== r.html) {
        if (r.mem.trim) { r.mem.trim = 0; html = r.fn(r.b, c, r.mem) || ''; }   // content changed: refit from full
        r.html = html; el.innerHTML = html;
      }
      fit(el, r, c);
      if (r.post) r.post(el, r, date);
    });
    reg.forEach((_, id) => { if (!seen.has(id)) reg.delete(id); });
  });

  BB.live = live;
  BB.cal = { DOW, MONTH, sod, addDays, dayDiff, dow3, hm, when, norm, events, status, onDay, nextOf, isFamily, whoName, whoText, owners, deg, forecastFor };

  /* ── zones / zone: where things sit ─────────────────────────── */
  /* {"type":"zones","children":[ zone | surface … ]}  the root of a Living Sky layout.
     {"type":"zone","place":"sky|sky-right|ground","children":[…]}
       sky        quiet sky, top-left: one column (clock, now, say, cal_header)
       sky-right  top-right corner: one small thing (legend, compact now)
       ground     bottom band, cream ink over a ground guard: a row, first child left, last child right
     A docked `surface` among the children moves the zones clear of it. */
  const PLACES = ['sky', 'sky-right', 'ground'];
  BB.define('zones', (b, ctx) => {
    const kids = b.children || [];
    const docks = kids.filter(k => k && k.type === 'surface').map(k => (k.dock === 'bottom' ? 'bottom' : 'left'));
    const ground = kids.some(k => k && k.type === 'zone' && k.place === 'ground');
    return `<div class="zones ${docks.map(d => 'with-' + d).join(' ')}">${
      ground && !docks.includes('bottom') ? '<div class="ground-guard"></div>' : ''}${BB.children(b, ctx)}</div>`;
  });
  BB.define('zone', (b, ctx) => {
    const place = PLACES.includes(b.place) ? b.place : 'sky';
    const html = BB.children(b, ctx);
    return `<div class="zone ${place} ${place === 'ground' ? 'ground' : 'sky-ink'}">${html}</div>`;
  });

  /* ── sky_clock ──────────────────────────────────────────────── */
  /* {"type":"sky_clock","size":"hero|compact","date":true,"h24":false,"weather":"{weather}"}
     hero = the home clock. compact = the brief's: smaller, with temperature and condition on its baseline. */
  live('sky_clock', (b, ctx) => {
    const d = ctx.now || BB.now(), compact = b.size === 'compact', h24 = b.h24 === true;
    const date = b.date === false ? '' : `<div class="date label">${DOW[d.getDay()]} · ${MONTH[d.getMonth()]} ${d.getDate()}</div>`;
    const time = h24 ? BB.fmtTime(d, { h24: true }) : hm(d);
    const ap = h24 ? '' : `<small>${d.getHours() < 12 ? 'AM' : 'PM'}</small>`;
    let wx = '';
    const w = b.weather;
    if (compact && w && typeof w === 'object' && (deg(w.temp) || w.condition)) {
      wx = `<span class="wx">${w.icon ? ico(w.icon) : ''}${deg(w.temp) ? `<b>${deg(w.temp)}</b>` : ''}${esc(w.condition)}</span>`;
    }
    return `<div class="skyclock ${compact ? 'compact' : 'hero'}">${date}<div class="time"><span class="d num">${time}</span>${ap}${wx}</div></div>`;
  });

  /* ── now ────────────────────────────────────────────────────── */
  /* {"type":"now","weather":"{weather}","style":"line|compact","outlook":"…"}
     line: temperature · icon · condition · one outlook sentence. compact: icon · temperature · condition.
     temp / condition / icon / outlook may also be given as their own params; they win over `weather`. */
  live('now', (b, ctx) => {
    const w = (b.weather && typeof b.weather === 'object') ? b.weather : {};
    const temp = deg(b.temp != null && b.temp !== '' ? b.temp : w.temp), cond = b.condition || w.condition || '', icon = b.icon || w.icon || '';
    if (!temp && !cond) return '';
    const i = icon ? ico(icon) : '';
    if (b.style === 'compact') return `<div class="wxnow compact">${i}${temp ? `<b>${temp}</b>` : ''}${cond ? `<span class="muted">${esc(cond)}</span>` : ''}</div>`;
    let out = b.outlook || w.outlook || '';
    if (!out) {                                             // nothing written: today's forecast, stated plainly
      const f = forecastFor(w.forecast, ctx.now || BB.now(), ctx.now || BB.now());
      if (f) {
        const hi = deg(f.high), lo = deg(f.low), range = hi && lo ? `${hi} / ${lo}` : hi ? `high ${hi}` : '';
        const diff = f.condition && String(f.condition).toLowerCase() !== String(cond).toLowerCase();
        out = diff ? f.condition + (hi ? ` · high ${hi}` : '') : range ? `Today ${range}` : '';
      }
    }
    return `<div class="wxnow">${temp ? `<span class="t d">${temp}</span>` : ''}${i}${cond ? `<span class="c">${esc(cond)}</span>` : ''}${
      out ? `<span class="o muted">${temp || cond ? '<i class="sep"></i>' : ''}${esc(out)}</span>` : ''}</div>`;
  });

  /* ── next_event ─────────────────────────────────────────────── */
  /* {"type":"next_event","events":"{calendar.events}","label":"Next","empty":"Nothing coming up"}
     Picks the running or next timed event itself. Kicker counts down; today → "Next · in 1 h 28 min",
     running → "Now · until 6:30", later days → "Tomorrow · 8:10 AM" / "Thu · 9:00 AM". */
  live('next_event', (b, ctx) => {
    if (!Array.isArray(b.events)) return '';                // source missing: say nothing
    const now = ctx.now || BB.now(), e = nextOf(events(b.events), now), label = b.label || 'Next';
    if (!e) return `<div class="nextev none"><div class="label">${esc(label)}</div><div class="ttl d">${esc(b.empty || 'Nothing coming up')}</div></div>`;
    const days = dayDiff(e.start, now), running = e.start <= now;
    const kick = running ? `Now · until ${hm(e.end)}`
      : days === 0 ? `${label} · ${BB.until(e.start, now)}`
      : `${days === 1 ? 'Tomorrow' : days < 7 ? dow3(e.start) : `${dow3(e.start)} ${MONTH[e.start.getMonth()].slice(0, 3)} ${e.start.getDate()}`} · ${BB.fmtTime(e.start)}`;
    const o = owners(e.people), who = whoText(e.people);
    const late = e.leave && !running && now >= e.leave;
    const parts = [days === 0 ? BB.fmtTime(e.start) : '', who,
      e.leave && !running ? (late ? 'leave now' : `leave by ${hm(e.leave)}`) : '', (!who || !e.leave || running) ? e.loc : ''].filter(Boolean).slice(0, 3);
    return `<div class="nextev ${o.cls}" style="${o.st}"><div class="label">${esc(kick)}</div><div class="ttl d">${esc(e.title)}</div>${
      parts.length ? `<div class="sub muted">${e.people.length ? '<i class="dot"></i>' : ''}<span>${parts.map(esc).join(' · ')}</span></div>` : ''}</div>`;
  });

  /* ── house ──────────────────────────────────────────────────── */
  /* {"type":"house","present":"{people.present}","replies":2,"todos":"{tasks.items}","sun":{"next":"sunset","at":"4:24 PM"}}
     Who is home, then quiet counts. present: [{name}] or [name]. replies / todos: a number or a list
     (open items are counted). sun defaults to state.atmosphere.sun. Zero or missing parts are left out;
     with nothing to show the block renders nothing. */
  const count = v => {
    if (Array.isArray(v)) return v.filter(t => !(t && /^(done|completed|complete|cancelled|canceled)$/i.test(t.status || ''))).length;
    const n = Math.floor(+v);
    return v == null || v === '' || isNaN(n) || n < 0 ? 0 : n;
  };
  BB.define('house', (b, ctx) => {
    const names = [...new Set((Array.isArray(b.present) ? b.present : []).map(p => (p && typeof p === 'object' ? p.name : p)).filter(Boolean).map(String))];
    const house = (ctx.state && ctx.state.people) || [];
    let who = '';
    if (names.length) {
      const all = house.length > 1 && house.every(h => names.some(n => BB.slug(n) === BB.slug(h)));
      if (all) who = `<span class="dots">${house.map(h => `<i class="dot ${BB.person(h)}"></i>`).join('')}</span><span>Everyone’s home</span>`;
      else {
        const shown = names.slice(0, 4), extra = names.length - shown.length;
        who = shown.map(n => `<span class="${BB.person(n)}"><i class="dot"></i>${esc(n)}</span>`).join('') +
          (extra > 0 ? `<span class="muted">+${extra}</span>` : '') + '<em class="muted">home</em>';
      }
    }
    const r = count(b.replies), t = count(b.todos), sun = b.sun || (ctx.atmosphere && ctx.atmosphere.sun) || null;
    const at = sun && sun.at ? String(sun.at).replace(/\s*[ap]\.?m\.?$/i, '') : '';
    const cts = [
      r ? `<span>${ico('reply')}<b>${r}</b> ${r > 1 ? 'replies' : 'reply'}</span>` : '',
      t ? `<span>${ico('todo')}<b>${t}</b> ${t > 1 ? 'to-dos' : 'to-do'}</span>` : '',
      at ? `<span>${ico(sun.next === 'sunrise' ? 'sunrise' : 'sunset')}<b>${esc(at)}</b></span>` : '',
    ].join('');
    if (!who && !cts) return '';
    return `<div class="house">${who ? `<div class="in">${who}</div>` : ''}${cts ? `<div class="ct muted">${cts}</div>` : ''}</div>`;
  });
})();
