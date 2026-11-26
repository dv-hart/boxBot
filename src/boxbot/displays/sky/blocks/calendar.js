/* Living Sky — calendar blocks: surface, say, agenda_timeline, todo_list, forecast_strip, cal_header,
   week_grid, day_grid, legend.

   Needs blocks/home.js first (BB.live, BB.cal). Blocks take raw source data as params — the calendar's
   `events` list, the weather `forecast` list, the task `items` list — and work out today, tomorrow, the
   week, what is past and what is next from ISO start times and the page clock. Per event only `title`
   and `start` are needed; `end`, `location`, `all_day`, `person` / `people`, `short_title`, `leave_by`,
   `done` and `kind:"holiday"` are used when present. */
(function () {
  'use strict';
  ICONS.flag = '<path d="M5.5 21V4M5.5 4.8h11.2l-2.2 3.6 2.2 3.6H5.5"/>';
  const esc = BB.esc, C = BB.cal, live = BB.live;
  const { DOW, MONTH, sod, addDays, dayDiff, dow3, hm, events, status, onDay, owners, whoText, deg, forecastFor } = C;
  const int = (v, dflt, lo, hi) => { const n = Math.round(+v); return v == null || v === '' || isNaN(n) ? dflt : Math.max(lo, Math.min(hi, n)); };
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many || one + 's'}`;
  const DONE = /^(done|completed|complete|cancelled|canceled)$/i;
  const NUM = ['No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve'];
  const word = n => NUM[n] || String(n);
  const unavailable = cls => `<div class="${cls}"><div class="free d">Calendar unavailable</div></div>`;
  /* "morning" | "evening"; anything else (including an unresolved binding) follows the clock */
  const modeOf = (m, now) => (m === 'morning' || m === 'evening' ? m : now.getHours() >= 17 ? 'evening' : 'morning');
  const flags = (list, all) => Object.fromEntries(all.map(k => [k, !Array.isArray(list) || list.includes(k)]));
  function weekStart(start, today) {
    if (start === 'monday') return addDays(today, -((today.getDay() + 6) % 7));
    if (start === 'sunday') return addDays(today, -today.getDay());
    return today;                                            // rolling: today + 6
  }
  const wxSmall = f => (f ? `${f.icon ? ico(f.icon) : ''}${deg(f.high) ? `<b>${deg(f.high)}</b>` : ''}${deg(f.low) ? `<i>${deg(f.low)}</i>` : ''}` : '');

  /* ── surface ────────────────────────────────────────────────── */
  /* {"type":"surface","dock":"left|bottom","children":[…]}  the slab: the one place dense content sits.
     left = 412px column, full height (agenda). bottom = full width, as tall as its content (week, day). */
  BB.define('surface', (b, ctx) =>
    `<div class="slab ${b.dock === 'bottom' ? 'bottom' : 'left'} surface">${BB.children(b, ctx)}</div>`);

  /* ── say ────────────────────────────────────────────────────── */
  /* {"type":"say","greeting":"{brief.greeting}","text":"{brief.summary}","events":"{calendar.events}","who":"…"}
     The agent's sentence, in serif, greeting as its italic lead-in. With no text it writes a plain line
     from `events`; with no greeting it greets by time of day (and `who`, if given). Nothing at all → nothing. */
  const short = (s, n) => (s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s);
  function dayLine(list, now) {
    const evs = events(list).filter(e => !e.allDay), at = e => BB.fmtTime(e.start);
    const today = evs.filter(e => dayDiff(e.start, now) === 0), left = today.filter(e => status(e, now) !== 'past');
    if (!left.length) {
      const tm = evs.find(e => dayDiff(e.start, now) === 1);
      return (today.length ? 'Nothing left on the calendar today.' : 'Nothing on the calendar today.') +
        (tm ? ` Tomorrow starts with ${short(tm.title, 30)} at ${at(tm)}.` : '');
    }
    /* "today" only when it is still morning and nothing has passed: an upcoming-only feed has forgotten what did */
    const first = left[0], fresh = left.length === today.length && now.getHours() < 12, name = short(first.title, 30);
    const head = `${word(left.length)} ${left.length === 1 ? 'thing' : 'things'} ${fresh ? 'today' : 'left today'}.`;
    const tail = status(first, now) === 'live' ? `${name} is on now.` : `${name} ${fresh && left.length > 1 ? 'comes first' : 'is next'} at ${at(first)}.`;
    return `${head} ${tail}`;
  }
  live('say', (b, ctx) => {
    const now = ctx.now || BB.now(), str = v => (typeof v === 'string' ? v.trim() : '');
    let greeting = str(b.greeting), text = str(b.text);
    if (!text && Array.isArray(b.events)) text = dayLine(b.events, now);
    if (!greeting && text) {
      const h = now.getHours();
      greeting = `Good ${h >= 4 && h < 12 ? 'morning' : h >= 12 && h < 17 ? 'afternoon' : 'evening'}${str(b.who) ? ', ' + str(b.who) : ''}`;
    }
    if (!greeting && !text) return '';
    if (greeting && !/[.!?,:;…]$/.test(greeting)) greeting += '.';
    return `<p class="saying d ${text.length + greeting.length > 190 ? 'xlong' : text.length + greeting.length > 106 ? 'long' : ''}">${greeting ? `<i>${esc(greeting)}</i> ` : ''}${esc(text)}</p>`;
  });

  /* ── agenda_timeline ────────────────────────────────────────── */
  /* {"type":"agenda_timeline","events":"{calendar.events}","mode":"auto|morning|evening","forecast":"{weather.forecast}",
      "gap_min":90,"show":["person","place","countdown","leave_by"],"title":"…","sections":[…]}
     An ordered rail, not an hour scale. morning: today, with a now marker, past rows struck to one line
     and the next event raised (countdown, leave-by pill); with three or fewer events today, tomorrow follows. evening: "Tonight" (what is left, done count)
     then "Tomorrow" with its first event raised. auto = evening from 5 PM.
     sections (optional, replaces mode): [{"day":0,"label":"…","remaining":false,"raise":"next|first|none","now":true}]
       day = offset from today (0 today, 1 tomorrow …) or "today" / "tomorrow".
     When the rail would overflow its slab it tightens, then sheds rows — past rows fold into "N done", then later rows
     fold into "+N more" — and never the raised one. */
  live('agenda_timeline', (b, ctx, mem) => {
    if (!Array.isArray(b.events)) return unavailable('agenda');
    const now = ctx.now || BB.now(), today = sod(now), evs = events(b.events);
    const show = flags(b.show, ['person', 'place', 'countdown', 'leave_by']), gapMin = int(b.gap_min, 90, 15, 720);
    const custom = Array.isArray(b.sections) && b.sections.length > 0;
    const mode = modeOf(b.mode || (ctx.args && ctx.args.mode), now);
    const defs = custom ? b.sections.filter(s => s && typeof s === 'object')
      : mode === 'evening' ? [{ day: 0, label: 'Tonight', remaining: true, raise: 'none' }, { day: 1, raise: 'first' }]
      : [{ day: 0, raise: 'next' }];
    /* a quiet day leaves the rail mostly empty: let tomorrow follow */
    if (!custom && mode === 'morning' && evs.filter(e => !e.allDay && dayDiff(e.start, now) === 0).length <= 3 &&
        evs.some(e => dayDiff(e.firstDay, now) <= 1 && dayDiff(e.lastDay, now) >= 1)) defs.push({ day: 1, raise: 'none' });

    const subOf = e => [show.person ? whoText(e.people) : '', show.place ? e.loc : ''].filter(Boolean).join(' · ') || e.duration;
    const leaveText = e => (now >= e.leave && now < e.start ? 'Leave now' : `Leave by ${hm(e.leave)}`);

    const secs = defs.map(s => {
      const off = s.day === 'tomorrow' ? 1 : s.day === 'today' ? 0 : int(s.day, 0, 0, 30);
      const day = addDays(today, off), isToday = off === 0, remaining = s.remaining === true;
      const raise = ['next', 'first', 'none'].includes(s.raise) ? s.raise : isToday ? 'next' : 'first';
      const mine = (Array.isArray(s.events) ? events(s.events) : evs).filter(e => (e.allDay ? onDay(e, day) : +e.firstDay === +day));
      const rows = [];
      let prevEnd = null, nowPut = !(isToday && !remaining && s.now !== false), nextSet = false, firstSet = false, done = 0, left = 0;
      const running = isToday && mine.some(e => status(e, now) === 'live');
      mine.forEach(e => {
        const st = isToday ? status(e, now) : e.done ? 'past' : '';
        if (st !== 'past') left++;
        if (st === 'past' && remaining) { done++; return; }
        if (!nowPut && !e.allDay && e.start > now) { rows.push({ k: 'now' }); nowPut = true; }
        const r = { k: 'ev', e, state: '', time: e.allDay ? 'all day' : hm(e.start), eta: '', leave: '', sub: e.allDay ? '' : subOf(e), gap: false };
        if (!e.allDay) { r.gap = prevEnd !== null && (e.start - prevEnd) / 6e4 >= gapMin; prevEnd = e.end; }
        const lv = !e.allDay && e.leave && show.leave_by;
        if (st === 'past') r.state = 'past';
        else if (st === 'live') { r.eta = 'now'; if (raise !== 'none') r.state = 'live'; }
        else if (isToday && !e.allDay && !nextSet) {
          nextSet = true;
          if (show.countdown) r.eta = BB.until(e.start, now);
          if (raise === 'next' && !running) { r.state = 'next'; if (lv) r.leave = leaveText(e); }
          else if (lv) r.sub = [show.person ? whoText(e.people) : '', leaveText(e).toLowerCase()].filter(Boolean).join(' · ');
        } else if (!isToday && !e.allDay && raise === 'first' && !firstSet) {
          firstSet = true; r.state = 'first'; if (lv) r.leave = `Leave by ${hm(e.leave)}`;
        } else if (lv && !isToday) r.sub = [r.sub, `leave ${hm(e.leave)}`].filter(Boolean).join(' · ');
        rows.push(r);
      });
      if (done) rows.unshift({ k: 'done', text: `${done} done today` });
      if (!mine.length || (remaining && !left)) rows.push({ k: 'free', text: !mine.length ? (isToday ? 'Nothing planned today' : 'Nothing planned') : 'Nothing left tonight' });
      else if (!nowPut) rows.push({ k: 'now' });
      const f = !isToday ? forecastFor(b.forecast, day, now) : null;
      const label = (typeof s.label === 'string' && s.label) || (isToday ? (remaining ? 'Tonight' : 'Today')
        : off === 1 ? `Tomorrow · ${dow3(day)} ${day.getDate()}` : `${DOW[day.getDay()]} · ${MONTH[day.getMonth()].slice(0, 3)} ${day.getDate()}`);
      return { rows, label, wx: f ? `${f.icon ? ico(f.icon) : ''}${deg(f.high)}` : '', total: mine.length, left, more: 0 };
    });

    /* shed content, one step per mem.trim */
    let t = mem.trim;
    const tight = t > 0;                                     // first: closer rows
    if (tight) t--;
    if (t > 0) {
      let folded = false;
      secs.forEach(s => {
        const past = s.rows.filter(r => r.state === 'past');
        if (past.length < 2) return;
        const at = s.rows.indexOf(past[0]);
        s.rows = s.rows.filter(r => r.state !== 'past');
        s.rows.splice(at, 0, { k: 'done', text: `${past.length} done` });
        folded = true;
      });
      if (folded) t--;
    }
    const droppable = s => s.rows.filter(r => r.k === 'ev' && (r.state === '' || r.state === 'past'));
    while (t > 0) {
      const s = secs.map(x => [x, droppable(x).length]).reduce((best, x) => (x[1] > 0 && x[1] >= best[1] ? x : best), [null, 0])[0];
      if (!s) break;
      const d = droppable(s);
      s.rows.splice(s.rows.indexOf(d[d.length - 1]), 1);
      s.more++; t--;
    }
    secs.forEach(s => { if (s.more) s.rows.push({ k: 'more', n: s.more }); });

    const multi = secs.length > 1, out = [];
    secs.forEach(s => {
      if (multi) out.push(`<div class="ar div"><div class="label">${esc(s.label)}</div><div class="dw">${s.wx}</div></div>`);
      s.rows.forEach((r, i) => {
        const nx = s.rows[i + 1], end = !nx || nx.k === 'more' || nx.k === 'free';
        if (r.k === 'now') out.push(`<div class="ar now ${end ? 'end' : ''}"><div class="tm num">${hm(now)}</div><div class="rl"><i></i></div><div class="nl"><span>now</span></div></div>`);
        else if (r.k === 'done') out.push(`<div class="ar done ${end ? 'end' : ''}"><div class="tm"></div><div class="rl"><i></i></div><div class="bd">${esc(r.text)}</div></div>`);
        else if (r.k === 'more') out.push(`<div class="ar more"><div class="tm"></div><div></div><div class="bd">+${r.n} more</div></div>`);
        else if (r.k === 'free') out.push(`<div class="ar free d">${esc(r.text)}</div>`);
        else {
          const o = owners(r.e.people), past = r.state === 'past';
          out.push(`<div class="ar ev ${r.state} ${r.e.allDay ? 'allday' : ''} ${nx && nx.gap ? 'gapa' : ''} ${end ? 'end' : ''} ${o.cls}" style="${o.st}">
            <div class="tm num">${esc(r.time)}</div><div class="rl"><i></i></div>
            <div class="bd"><div class="tt"><span>${esc(r.e.title)}</span>${r.eta ? `<em>${esc(r.eta)}</em>` : ''}</div>${
              !past && r.sub ? `<div class="sb">${esc(r.sub)}</div>` : ''}${
              r.leave ? `<div class="lv">${ico('car')}${esc(r.leave)}</div>` : ''}</div></div>`);
        }
      });
    });

    const s0 = secs[0];
    const title = (typeof b.title === 'string' && b.title) || (custom ? 'Agenda' : mode === 'evening' ? 'Tonight and tomorrow' : multi ? 'Today and tomorrow' : 'Today');
    const ct = multi ? (secs.some(s => s.left) ? secs.map(s => s.left).filter(Boolean).join(' + ') : '')
      : !s0 || !s0.total ? '' : s0.left === s0.total ? plural(s0.total, 'event') : `${s0.left} of ${s0.total} left`;
    return `<div class="agenda"><div class="hd"><div class="label">${esc(title)}</div><div class="ct">${ct}</div></div>
      <div class="tl ${tight ? 'tight' : ''}" data-fit>${out.join('')}</div></div>`;
  });

  /* ── todo_list ──────────────────────────────────────────────── */
  /* {"type":"todo_list","items":"{tasks.items}","max":3,"label":"To-do"}  open items only; due_date → a short tag */
  live('todo_list', (b, ctx) => {
    if (!Array.isArray(b.items)) return '';
    const now = ctx.now || BB.now();
    const open = b.items.filter(t => t && (typeof t === 'string' || (t.description && !DONE.test(t.status || ''))));
    if (!open.length) return '';
    const shown = open.slice(0, int(b.max, 3, 1, 6));
    const due = t => {
      const d = C.when(t.due_date); if (!d) return '';
      const n = dayDiff(d, now);
      return n < 0 ? 'overdue' : n === 0 ? 'due today' : n < 7 ? `by ${dow3(d)}` : `by ${MONTH[d.getMonth()].slice(0, 3)} ${d.getDate()}`;
    };
    const label = (typeof b.label === 'string' && b.label) || 'To-do';
    return `<div class="todo"><div class="label">${esc(label)} · ${shown.length < open.length ? `${shown.length} of ${open.length}` : open.length}</div><ul>${
      shown.map(t => { const tag = typeof t === 'string' ? '' : due(t);
        return `<li>${ico('circle')}<span>${esc(typeof t === 'string' ? t : t.description)}</span>${tag ? `<b>${tag}</b>` : ''}</li>`; }).join('')}</ul></div>`;
  });

  /* ── forecast_strip ─────────────────────────────────────────── */
  /* {"type":"forecast_strip","days":"{weather.forecast}","count":5,"skip_today":true}  day · icon · high */
  live('forecast_strip', (b, ctx) => {
    if (!Array.isArray(b.days)) return '';
    const now = ctx.now || BB.now(), today = b.skip_today === false ? null : forecastFor(b.days, now, now);
    const days = b.days.filter(d => d && d !== today && d.day && (d.icon || deg(d.high))).slice(0, int(b.count, 5, 1, 7));
    if (!days.length) return '';
    return `<div class="fcst">${days.map(d => `<div><span>${esc(String(d.day).slice(0, 3))}</span><p>${d.icon ? ico(d.icon) : ''}<b>${deg(d.high)}</b></p></div>`).join('')}</div>`;
  });

  /* ── cal_header ─────────────────────────────────────────────── */
  /* {"type":"cal_header","range":"week|day","events":"{calendar.events}","start":"rolling|monday|sunday","title":"…","text":"{week.summary}"}
     Serif title, the date range and event count beside it, and (optionally) one agent sentence under it. */
  live('cal_header', (b, ctx) => {
    const now = ctx.now || BB.now(), today = sod(now), have = Array.isArray(b.events), evs = have ? events(b.events) : [];
    const from = b.start || (ctx.args && ctx.args.start);
    const md = d => `${MONTH[d.getMonth()]} ${d.getDate()}`;
    let title, sub, n;
    if (b.range === 'day') {
      title = DOW[today.getDay()]; sub = md(today);
      n = evs.filter(e => onDay(e, today)).length;
    } else {
      const a = weekStart(from, today), z = addDays(a, 6);
      const m3 = d => `${MONTH[d.getMonth()].slice(0, 3)} ${d.getDate()}`;
      title = from === 'monday' || from === 'sunday' ? 'This week' : 'Week ahead';
      sub = a.getMonth() === z.getMonth() ? `${md(a)} – ${z.getDate()}` : `${m3(a)} – ${m3(z)}`;
      n = evs.filter(e => +e.lastDay >= +a && +e.firstDay <= +z).length;
    }
    if (typeof b.title === 'string' && b.title) title = b.title;
    const text = typeof b.text === 'string' ? b.text.trim() : '';
    return `<div class="calhdr"><div class="hdr"><div class="ttl d">${esc(title)}</div><div class="sub">${sub}${have ? ` · ${n ? plural(n, 'event') : 'no events'}` : ''}</div></div>${
      text ? `<div class="line">${esc(text)}</div>` : ''}</div>`;
  });

  /* ── legend ─────────────────────────────────────────────────── */
  /* {"type":"legend","people":["…"],"family":"Everyone"}  people default to the household; family:false hides the shared entry */
  BB.define('legend', (b, ctx) => {
    const ppl = (Array.isArray(b.people) && b.people.length ? b.people : (ctx.state && ctx.state.people) || [])
      .map(p => (p && typeof p === 'object' ? p.name : p)).filter(Boolean).slice(0, 4);
    if (!ppl.length) return '';
    const fam = b.family === false ? '' : `<span class="who p-family"><i class="dot"></i>${esc(typeof b.family === 'string' && b.family ? b.family : 'Everyone')}</span>`;
    return `<div class="legend">${ppl.map(p => `<span class="who ${BB.person(p)}"><i class="dot"></i>${esc(p)}</span>`).join('')}${fam}</div>`;
  });

  /* ── week_grid ──────────────────────────────────────────────── */
  /* {"type":"week_grid","events":"{calendar.events}","forecast":"{weather.forecast}","start":"rolling|monday|sunday","max_per_day":5}
     Seven columns, today wider and raised. All-day and multi-day events are bars across columns (two owners
     = a gradient of both). Per day at most max_per_day chips, then "+N more"; fewer if the slab would overflow. */
  function lanes(bars) {
    const ends = [];
    return { bars: bars.slice().sort((x, y) => (y.b - y.a) - (x.b - x.a) || x.a - y.a).map(x => {
      let l = ends.findIndex(e => e < x.a); if (l < 0) l = ends.length;
      ends[l] = x.b; return { ...x, lane: l };
    }), n: ends.length };
  }
  live('week_grid', (b, ctx, mem) => {
    if (!Array.isArray(b.events)) return unavailable('wk');
    const now = ctx.now || BB.now(), today = sod(now), start = weekStart(b.start || (ctx.args && ctx.args.start), today), tIdx = dayDiff(today, start), N = 7;
    const MAX = Math.max(1, int(b.max_per_day, 5, 1, 8) - mem.trim);
    const evs = events(b.events).filter(e => dayDiff(e.lastDay, start) >= 0 && dayDiff(e.firstDay, start) < N);
    const isHol = e => e.kind === 'holiday', isBar = e => !isHol(e) && (e.allDay || e.end - e.start >= 864e5);
    const L = lanes(evs.filter(isBar).map(e => ({ e, a: dayDiff(e.firstDay, start), b: dayDiff(e.lastDay, start) })).slice(0, 12));
    const timed = evs.filter(e => !isBar(e) && !isHol(e));
    const mine = timed.filter(e => dayDiff(e.firstDay, start) === tIdx);
    const next = mine.some(e => status(e, now) === 'live') ? null : mine.find(e => e.start > now && !e.done);
    const anyWx = Array.isArray(b.forecast) && b.forecast.length > 0;
    const chip = (e, isToday) => {
      const o = owners(e.people), st = isToday ? status(e, now) : e.done ? 'past' : '';
      const label = isToday && e.title.length <= 18 ? e.title : e.short || e.title;
      return `<div class="ev ${o.cls} ${st} ${e === next ? 'nxt' : ''}" style="${o.st}"><b class="num">${hm(e.start)}</b><span>${esc(label)}</span></div>`;
    };
    let h = `<div class="tbg" style="grid-column:${tIdx + 1};grid-row:1/-1"></div>`;
    for (let i = 0; i < N; i++) {
      const d = addDays(start, i), c = i + 1, k = i < tIdx ? 'past' : i === tIdx ? 'today' : '', isToday = i === tIdx;
      const f = i >= tIdx ? forecastFor(b.forecast, d, now) : null;
      h += `<div class="dh ${k}" style="grid-column:${c};grid-row:1"><div><span class="dow">${dow3(d)}</span><b class="d">${d.getDate()}</b></div>${
        anyWx ? `<span class="w">${wxSmall(f)}</span>` : ''}</div>`;
      const all = timed.filter(e => dayDiff(e.firstDay, start) === i), hol = evs.filter(e => isHol(e) && onDay(e, d));
      let ev = all, pre = '';
      if (isToday) {
        const done = ev.filter(e => status(e, now) === 'past');
        if (ev.length > MAX && done.length) { pre = `<div class="done">${done.length} done</div>`; ev = ev.filter(e => status(e, now) !== 'past'); }
      }
      const more = ev.length - MAX, shown = ev.slice(0, more > 0 ? Math.max(1, MAX - 1) : MAX);
      let nowAt = isToday && shown.length ? shown.findIndex(e => e.start > now) : -1;
      if (isToday && shown.length && nowAt < 0) nowAt = shown.length;
      const r0 = 2 + L.bars.filter(x => x.a <= i && x.b >= i).reduce((mx, x) => Math.max(mx, x.lane + 1), 0);
      h += `<div class="st ${k}" style="grid-column:${c};grid-row:${r0}/-1">${
        hol.map(x => `<div class="hol">${ico('flag')}<span>${esc(x.short || x.title)}</span></div>`).join('')}${pre}${
        shown.map((e, j) => (j === nowAt ? '<div class="nowl"></div>' : '') + chip(e, isToday)).join('')}${nowAt === shown.length && shown.length ? '<div class="nowl"></div>' : ''}${
        ev.length > shown.length ? `<div class="more">+${ev.length - shown.length} more</div>` : ''}${
        evs.length && !all.length && !hol.length && i >= tIdx ? '<div class="free d">Nothing planned</div>' : ''}</div>`;
    }
    L.bars.forEach(x => {
      const o = owners(x.e.people), s = Math.max(x.a, 0), e = Math.min(x.b, N - 1);
      h += `<div class="bar ${o.cls} ${x.a < 0 ? 'cl' : ''} ${x.b > N - 1 ? 'cr' : ''} ${e < tIdx ? 'past' : ''}" style="${o.st};grid-column:${s + 1}/${e + 2};grid-row:${x.lane + 2}"><span>${esc(e - s < 1 && x.e.short ? x.e.short : x.e.title)}</span></div>`;
    });
    const colsT = Array.from({ length: N }, (_, i) => `minmax(0,${i === tIdx ? 1.5 : 1}fr)`).join(' ');
    return `<div class="wk" data-fit><div class="grid" style="grid-template-columns:${colsT};grid-template-rows:auto ${L.n ? `repeat(${L.n},auto) ` : ''}1fr">${h}</div>${evs.length ? '' : '<div class="free d none">Nothing planned this week</div>'}</div>`;
  });

  /* ── day_grid ───────────────────────────────────────────────── */
  /* {"type":"day_grid","events":"{calendar.events}","people":["…"],"from":"07:30","to":"20:30",
      "background":[{"label":"School · out 3:10","start":"08:25","end":"15:10","people":["…"]}]}
     Today on an hour scale, a column per person (default: the household). Shared or unowned events span
     every column; if no event names a household member there is one shared column. The scale widens to
     include early or late events. background = dashed context blocks (school hours). */
  const mins = (v, dflt) => { const m = /^(\d{1,2}):(\d{2})$/.exec(String(v || '')); return m ? Math.min(1440, +m[1] * 60 + +m[2]) : dflt; };
  const HEIGHT = 351;
  live('day_grid', (b, ctx) => {
    if (!Array.isArray(b.events)) return unavailable('dg');
    const now = ctx.now || BB.now(), day = sod(now);
    const evs = events(b.events).filter(e => onDay(e, day)), timed = evs.filter(e => !e.allDay), allDay = evs.filter(e => e.allDay);
    const m = d => Math.max(0, Math.min(1440, (d - day) / 6e4));
    let cols = (Array.isArray(b.people) && b.people.length ? b.people : (ctx.state && ctx.state.people) || [])
      .map(p => (p && typeof p === 'object' ? p.name : p)).filter(Boolean).slice(0, 5);
    const col = p => cols.findIndex(c => BB.slug(c) === BB.slug(p));
    const bg = (Array.isArray(b.background) ? b.background : []).filter(x => x && mins(x.start, -1) >= 0 && mins(x.end, -1) > mins(x.start, -1));
    if (!timed.some(e => e.people.some(p => col(p) >= 0)) && !bg.some(x => (x.people || []).some(p => col(p) >= 0))) cols = [];
    const n = Math.max(1, cols.length);
    let H0 = mins(b.from, 450), H1 = mins(b.to, 1230);
    if (H1 - H0 < 120) { H0 = 450; H1 = 1230; }
    timed.forEach(e => { H0 = Math.min(H0, Math.floor(m(e.start) / 60) * 60); H1 = Math.max(H1, Math.ceil(m(e.end) / 60) * 60); });
    const PX = HEIGHT / (H1 - H0), y = v => ((v - H0) * PX).toFixed(1), t = m(now), pct = v => (v * 100).toFixed(3) + '%';
    let g = '';
    const step = H1 - H0 > 15 * 60 ? 3 : 2;
    for (let hh = Math.ceil(H0 / 60 / 2) * 2; hh * 60 <= H1; hh += step) {
      if (Math.abs(hh * 60 - t) <= 45 || hh * 60 === H0) continue;
      g += `<div class="hr" style="top:${y(hh * 60)}px"><span>${hh === 12 ? 'noon' : hh === 24 || hh === 0 ? 'midnight' : ((hh + 11) % 12 + 1) + (hh % 24 < 12 ? ' am' : ' pm')}</span></div>`;
    }
    bg.forEach(x => (Array.isArray(x.people) && x.people.length ? x.people.map(col).filter(c => c >= 0) : [...Array(n).keys()]).forEach(c => {
      const s = Math.max(H0, mins(x.start)), e = Math.min(H1, mins(x.end));
      if (e > s) g += `<div class="sch" style="top:${y(s)}px;height:${((e - s) * PX).toFixed(1)}px;left:${pct(c / n)};width:calc(${pct(1 / n)} - 8px)"><span>${esc(x.label)}</span></div>`;
    }));
    /* an event → one block per run of adjacent columns it covers */
    const blocks = [];
    timed.forEach(e => {
      let idx = [...new Set(e.people.map(col).filter(c => c >= 0))].sort((p, q) => p - q);
      if (!idx.length || e.people.some(C.isFamily)) idx = [...Array(n).keys()];
      for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && idx[j + 1] === idx[j] + 1) j++;
        blocks.push({ e, c: idx[i], span: j - i + 1, s: m(e.start), en: Math.max(m(e.end), m(e.start) + 26 / PX), lane: 0, of: 1 }); i = j + 1; }
    });
    /* events that overlap in one column share its width (a 4px graze from the minimum block height is let be) */
    const TOL = 4 / PX;
    for (let c = 0; c < n; c++) {
      const own = blocks.filter(x => x.span === 1 && x.c === c).sort((p, q) => p.s - q.s);
      let cluster = [], ends = [], cEnd = -1;
      const close = () => { cluster.forEach(x => { x.of = ends.length; }); cluster = []; ends = []; };
      own.forEach(x => {
        if (cluster.length && x.s >= cEnd - TOL) close();
        let l = ends.findIndex(en => en <= x.s + TOL); if (l < 0) l = ends.length;
        ends[l] = x.en; x.lane = l; cluster.push(x); cEnd = Math.max(cEnd, x.en);
      });
      close();
    }
    blocks.forEach(x => {
      const e = x.e, o = owners(e.people), hgt = Math.max(26, (m(e.end) - m(e.start)) * PX), st = status(e, now), wide = x.span === n && n > 1;
      const w = x.span / n / x.of, left = x.c / n + x.lane * w;
      g += `<div class="blk ${o.cls} ${st} ${hgt >= 46 ? 'tall' : ''}" style="${o.st};top:${y(x.s)}px;height:${hgt.toFixed(1)}px;left:${pct(left)};width:${pct(w)}"><div><b class="num">${hm(e.start)}</b><span>${esc(e.title)}</span>${
        (hgt >= 46 || wide || n === 1) && e.loc ? `<em>${esc(e.loc)}</em>` : ''}</div></div>`;
    });
    if (t > H0 && t < H1) g += `<div class="nowl" style="top:${y(t)}px"><span class="num">${hm(now)}</span></div>`;
    const head = cols.length ? `<div class="ph" style="grid-template-columns:76px repeat(${n},1fr)"><div></div>${cols.map(p => `<div class="pn ${BB.person(p)}"><i class="dot"></i>${esc(p)}</div>`).join('')}</div>` : '';
    const strip = allDay.length ? `<div class="ad">${allDay.slice(0, 3).map(e => { const o = owners(e.people);
      return `<span class="${o.cls}" style="${o.st}">${esc(e.title)}</span>`; }).join('')}</div>` : '';
    return `<div class="dg ${cols.length ? '' : 'solo'}">${head}${strip}${
      !timed.length ? '<div class="free d">Nothing on the clock today</div>' : ''}<div class="tg" style="height:${HEIGHT}px"><div class="in">${g}</div></div></div>`;
  });
})();
