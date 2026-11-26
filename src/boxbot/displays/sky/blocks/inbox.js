/* inbox — voice-first mail triage. One block; the agent triages mail and pushes these params (static source).

   The count sits in the sky; everything else is on one solid dock whose height follows the amount of work:
     triage   focused thread large, two more compact, "handled" + FYI footer
     thread   the focused thread with its quote and the drafted reply
     sent     transient: what went out, undo countdown, the next thread
     cleared  a one-line footer; the scene takes over
   privacy:"discreet" (someone who does not own the inbox is in the room) shows counts, senders and whose inbox
   ONLY: no subjects, summaries, quotes, drafts, suggested actions, deadlines, FYIs or filing detail. The block
   drops those fields itself, so the spec says exactly what is and is not on screen.

   Threads are numbered by position (1, 2, 3 = "the first one"); more than three shows a window of three and
   "+N more". Each mode expects its own scene horizon/scale: table in specs/inbox.json.

   Senders, subjects and summaries are written by strangers: every value goes through BB.esc(). */
(function () {
  'use strict';
  ICONS.lock = ICONS.lock || '<rect x="5.5" y="10.5" width="13" height="9.5" rx="2.5"/><path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5"/>';
  ICONS.knob = ICONS.knob || '<circle cx="12" cy="12" r="8.5"/><path d="M12 3.5v5"/>';
  ICONS.clock = ICONS.clock || '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.2V12l3.2 2"/>';
  ICONS.undo = ICONS.undo || '<path d="M9 6L4.5 10.5 9 15M5 10.5h9a5 5 0 0 1 0 10h-3"/>';

  const esc = BB.esc, MODES = ['triage', 'thread', 'sent', 'cleared'], SHOW = 3;
  const str = v => (v == null ? '' : String(v).trim());
  const arr = v => (Array.isArray(v) ? v : v == null || v === '' ? [] : [v]);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const int = v => (v == null || v === '' || !isFinite(+v) ? null : Math.max(0, Math.round(+v)));
  const obj = v => (v && typeof v === 'object' && !Array.isArray(v) ? v : null);
  const ORD = ['first', 'second', 'third'];

  /* "carina" → "Carina" as the household spells it */
  function disp(name) {
    const k = BB.slug(name), hit = ((BB.state && BB.state.people) || []).find(p => BB.slug(p) === k);
    return hit || (str(name).charAt(0).toUpperCase() + str(name).slice(1));
  }
  const who = (name, extra) => `<span class="who ${BB.person(name)}"><i class="dot"></i>${esc(disp(name))}${extra == null ? '' : ' ' + esc(extra)}</span>`;
  const whos = t => t.who.map(n => who(n)).join('');
  const sep = '<i class="ib-sep"></i>';
  const q = s => `“${esc(str(s).replace(/^[“"']+|[”"']+$/g, ''))}”`;
  /* badge fill = whose inbox: one owner → their colour, two → split, more or unknown → the family colour */
  function badge(t, big) {
    const cols = [...new Set(t.who.map(n => BB.person(n)).filter(Boolean).map(c => `var(--${c.slice(2)})`))];
    const bg = cols.length === 1 ? cols[0] : cols.length === 2 ? `linear-gradient(115deg,${cols[0]} 50%,${cols[1]} 50%)` : '';
    return `<span class="ib-bdg d${big ? ' big' : ''}${t.n > 9 ? ' wide' : ''}"${bg ? ` style="background:${bg}"` : ''}>${t.n}</span>`;
  }
  const due = t => (t.due ? `<span class="ib-due${t.urgent ? ' urgent' : ''}">${ico('clock')}${esc(t.due)}</span>` : '');
  const draftOf = v => { const d = obj(v) || (str(v) ? { text: v } : null); return d && str(d.text) ? d : null; };

  function thread(t, i) {
    return { n: i + 1, sender: str(t.sender || t.from || t.name) || 'Unknown sender', role: str(t.role), who: arr(t.for).map(str).filter(Boolean),
      summary: str(t.summary), short: str(t.short || t.subject), due: str(t.due), urgent: t.urgent === true || t.urgent === 1 || t.urgent === 'true',
      suggest: str(t.suggest), quote: str(t.quote), draft: draftOf(t.draft), say: arr(t.say).map(str).filter(Boolean) };
  }

  const head = (n, big, small, cls) => `<div class="ib-hd sky-ink ${cls || ''}"><span class="ib-n d">${n}</span><div><div class="ib-t d">${big}</div>${small ? `<div class="ib-s">${small}</div>` : ''}</div></div>`;
  function now(b, ctx) {
    const t = b.temp == null || b.temp === '' ? '' : (isFinite(+b.temp) ? Math.round(+b.temp) : String(b.temp).replace(/\s*°.*$/, '')) + '°';
    const c = str(b.condition), line = esc(t) + (c ? `<span class="ib-cond">${t ? ' · ' : ''}${esc(c)}</span>` : '');
    return `<div class="ib-now sky-ink"><span class="d ib-clock">${esc(BB.fmtTime(BB.now(), { meridiem: false }))}</span>` +
      (line || b.icon ? `<span class="ib-wx">${b.icon ? ico(b.icon) : ''}<span>${line}</span></span>` : '') + '</div>';
  }
  const dock = (cls, inner) => `<div class="slab solid bottom ib-dock ib-${cls}">${inner}</div>`;

  BB.define('inbox', (b, ctx) => {
    const T = arr(b.threads).filter(obj).map(thread), discreet = b.privacy === 'discreet';
    const count = int(b.count), handled = obj(b.handled) || {}, hc = int(handled.count);
    let mode = MODES.includes(b.mode) ? b.mode : '';
    /* nothing pushed yet: say so; never claim the inbox is clear on no data */
    if (!mode && !T.length && count == null) return `<div class="inbox m-empty">${head(ico('reply'), 'No mail summary yet', '', 'zero quiet')}${now(b, ctx)}</div>`;
    if (!mode) mode = 'triage';
    if ((mode === 'thread' || mode === 'sent') && !T.length) mode = 'triage';

    const f = clamp(int(b.focus) || 1, 1, Math.max(1, T.length)), cur = T[f - 1];
    const owners = obj(b.owners) ? Object.entries(b.owners).map(([k, v]) => [k, int(v) || 0])
      : [...new Set(T.flatMap(t => t.who))].map(k => [k, T.filter(t => t.who.includes(k)).length]);
    /* a window of SHOW threads that always contains the focused one */
    const start = clamp(f - SHOW, 0, Math.max(0, T.length - SHOW)), win = T.slice(start, start + SHOW), more = T.length - win.length;
    const moreTxt = more > 0 ? `<span class="ib-more">+${more} more</span>` : '';
    const view = discreet && (mode === 'triage' || mode === 'thread') ? 'discreet' : mode;
    const tally = hc != null ? `<span class="ib-ok">${ico('check')}<b>${hc} handled</b></span>${!discreet && str(handled.breakdown) ? `<span class="ib-clip">${esc(handled.breakdown)}</span>` : ''}` : '';
    let ui;

    if (view === 'cleared') {
      const arrived = int(b.arrived), last = obj(b.last) || {};
      const filed = str(last.at) ? `<span class="ib-ok">${ico('check')}<b>Last filed ${esc(last.at)}</b></span>` : '';
      const parts = [filed + (!discreet && str(last.text) ? `<span class="ib-clip">${esc(last.text)}</span>` : ''), owners.map(([k, v]) => `<span>${who(k, v)}</span>`).join('')].filter(Boolean);
      ui = head(ico('check'), 'Nothing needs you', arrived != null ? `${arrived} arrived today · all handled` : hc != null ? `${hc} handled` : '', 'zero') + now(b, ctx) +
        dock('slim', `<div class="ib-foot">${parts.join(sep)}<span class="ib-say r">${ico('mic')}“What did you handle?”</span></div>`);
    } else if (view === 'sent') {
      const left = count != null ? count : Math.max(0, T.length - 1), nx = T.length > 1 ? T[f % T.length] : null, d = draftOf(b.draft) || cur.draft || {};
      const secs = int(b.undo_seconds), fx = str(d.effect_done || d.effect);
      ui = (left ? head(left, left === 1 ? 'still needs a reply' : 'still need a reply', `Sent just now${hc != null ? ` · ${hc} handled` : ''}`)
        : head(ico('check'), 'All answered', `Sent just now${hc != null ? ` · ${hc} handled` : ''}`, 'zero')) + now(b, ctx) +
        dock('sent', `${secs ? `<i class="ib-fuse" style="animation-duration:${secs}s"></i>` : ''}<div class="ib-cols${nx ? '' : ' one'}">
          <div class="ib-done"><span class="ib-tick">${ico('check')}</span><div><div class="ib-nm d">Sent to ${esc(cur.sender)}</div>
            ${!discreet && str(d.text) ? `<p class="ib-q d">${q(d.text)}</p>` : ''}${!discreet && fx ? `<div class="ib-fx">${ico('cal')}<span>${esc(fx)}</span></div>` : ''}</div></div>
          ${nx ? `<div class="ib-next"><div class="label">Next</div><div class="ib-r1">${badge(nx)}<b>${esc(nx.sender)}</b></div>${!discreet && nx.short ? `<p>${esc(nx.short)}</p>` : ''}</div>` : ''}</div>
          <div class="ib-foot">${secs ? `<span class="ib-say ib-undo" data-s="${secs}">${ico('undo')}“Undo” <em>for ${secs} more second${secs === 1 ? '' : 's'}</em></span>` : ''}
            ${nx ? `<span class="ib-say r">${ico('mic')}“Next one”${sep}${ico('knob')}turn</span>` : ''}</div>`);
    } else if (view === 'thread') {
      const d = draftOf(b.draft) || cur.draft, quote = str(b.quote) || cur.quote;
      const say = (arr(b.say).map(str).filter(Boolean).length ? arr(b.say).map(str).filter(Boolean) : cur.say.length ? cur.say : d ? ['Send it', 'Skip'] : ['Skip']).slice(0, 3);
      const pips = T.length <= 8 ? `<span class="ib-pips">${T.map(t => `<i${t.n === f ? ' class="on"' : ''}></i>`).join('')}</span>` : '';
      ui = head(f, `of ${T.length} ${pips}`, T.length > 1 ? `${ico('knob')}turn for the next · press to hear it again` : `${ico('knob')}press to hear it again`) + now(b, ctx) +
        dock('thread', `<div class="ib-cols${d ? '' : ' one'}">
          <div class="ib-hero"><div class="ib-r1">${badge(cur, 1)}<span class="ib-nm d">${esc(cur.sender)}</span>${due(cur)}</div>
            ${ctxLine(cur)}${cur.summary ? `<p class="ib-sum">${esc(cur.summary)}</p>` : ''}${quote ? `<p class="ib-q d">${q(quote)}</p>` : ''}</div>
          ${d ? `<div class="ib-note"><div class="label">Draft reply${str(d.from) ? ` · from ${esc(disp(d.from))}` : ''}</div><p class="d">${esc(d.text)}${str(d.sign) ? `<span> — ${esc(d.sign)}</span>` : ''}</p>
            ${str(d.effect) ? `<div class="ib-fx">${ico('cal')}<span>On send: ${esc(d.effect)}</span></div>` : ''}</div>` : ''}</div>
          <div class="ib-foot"><span class="ib-say">${ico('mic')}${say.map(q).join(sep)}</span>
            ${d ? `<span class="r">Nothing sends until ${str(d.from) ? esc(disp(d.from)) + ' says' : 'you say'} so</span>` : ''}</div>`);
    } else if (view === 'discreet') {
      const n = count != null ? count : T.length, names = owners.map(([k]) => disp(k));
      const note = str(b.privacy_note) || 'Someone else is in the room';
      ui = head(n, n === 1 ? 'needs a reply' : 'need a reply', `${ico('lock')}Details hidden${hc != null ? ` · ${hc} handled for you` : ''}${more > 0 ? sep + moreTxt : ''}`) + now(b, ctx) +
        dock('discreet', `<div class="ib-trio">${win.map(t => `<div class="ib-it${mode === 'thread' && t === cur ? ' on' : ''}"><div class="ib-r1">${badge(t)}<b>${esc(t.sender)}</b></div>
            <div class="ib-r2">${whos(t)}</div></div>`).join('')}</div>
          <div class="ib-foot"><span class="ib-ok">${ico('lock')}<b>${esc(note)}</b></span><span>senders only</span>
            <span class="ib-say r">${ico('mic')}${names.length ? esc(names.slice(0, 3).join(' or ')) + ': ' : ''}“Show details”</span></div>`);
    } else {
      const n = count != null ? count : T.length, rest = win.filter(t => t !== cur);
      const hint = T.length > 1 ? `${ico('knob')}turn, or say “the ${ORD[f === 2 ? 0 : 1]} one”` : '';
      const sub = owners.map(([k, v]) => who(k, v)).join('') + moreTxt + (hint && (owners.length || more > 0) ? sep : '') + hint;
      const fyi = arr(b.fyi).map(str).filter(Boolean);
      const foot = [tally, fyi.length ? `<span class="ib-fyi ib-clip"><b>FYI</b>${esc(fyi.join(' · '))}</span>` : ''].filter(Boolean).join(sep);
      ui = head(n, n === 1 ? 'needs a reply' : 'need a reply', sub) + now(b, ctx) + (cur ? dock('triage', `<div class="ib-cols${rest.length ? '' : ' one'}">
          <div class="ib-hero"><div class="ib-r1">${badge(cur, 1)}<span class="ib-nm d">${esc(cur.sender)}</span>${due(cur)}</div>
            ${ctxLine(cur)}${cur.summary || cur.short ? `<p class="ib-sum">${esc(cur.summary || cur.short)}</p>` : ''}
            <div class="ib-r4">${cur.suggest ? `<span class="ib-go">${ico('reply')}<span>${esc(cur.suggest)}</span></span>` : ''}${cur.suggest ? `<span class="ib-say">${ico('mic')}“Send it”</span><span class="ib-say">${sep}“Read it to me”</span>` : `<span class="ib-say">${ico('mic')}“Read it to me”</span>`}</div></div>
          ${rest.length ? `<div class="ib-rest${rest.length === 1 ? ' single' : ''}">${rest.map(t => `<div class="ib-it"><div class="ib-r1">${badge(t)}<b>${esc(t.sender)}</b>${due(t)}</div>
            ${t.short || t.summary ? `<p>${esc(t.short || t.summary)}</p>` : ''}${t.suggest ? `<div class="ib-sg">${ico('reply')}<span>${esc(t.suggest)}</span></div>` : ''}</div>`).join('')}</div>` : ''}</div>
          ${foot ? `<div class="ib-foot">${foot}</div>` : ''}`) : foot ? dock('slim', `<div class="ib-foot">${foot}</div>`) : '');
    }
    return `<div class="inbox m-${view}">${ui}</div>`;
  });

  function ctxLine(t) {
    if (!t.role && !t.who.length) return '';
    return `<div class="ib-ctx"><span>${esc(t.role)}</span>${t.who.length ? `<span class="ib-for">${t.role ? sep : ''}for ${whos(t)}</span>` : ''}</div>`;
  }

  /* undo countdown: the bar is a CSS transform; the seconds are counted from when the pane landed */
  BB.onMount(root => root.querySelectorAll('.ib-undo[data-s]').forEach(e => { e.dataset.until = Date.now() + (+e.dataset.s) * 1000; }));
  BB.onTick((d, root) => {
    const t = BB.fmtTime(d, { meridiem: false });
    root.querySelectorAll('.ib-clock').forEach(e => { if (e.textContent !== t) e.textContent = t; });
    root.querySelectorAll('.ib-undo[data-until]').forEach(e => {
      const s = Math.ceil((+e.dataset.until - Date.now()) / 1000), em = e.querySelector('em');
      if (s <= 0) e.classList.add('gone');
      else if (em) em.textContent = `for ${s} more second${s === 1 ? '' : 's'}`;
    });
  });
})();
