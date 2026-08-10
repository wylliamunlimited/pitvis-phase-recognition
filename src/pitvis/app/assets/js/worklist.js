// The procedure worklist — fourteen steps, canonical order, never reordered.
//
// This is the piece that stops the interface reading as a video editor. A
// timeline answers "where am I in the file"; a worklist answers "how far
// through the operation are we, and what has been done" — which is the
// question a procedure display exists to answer.
//
// WHY A FIXED CHECKLIST AND NOT A LOG OF WHAT HAPPENED
//
// Measured on the predictions on disk: 28 segments on video_25, 48 on
// video_01, and the surgeon's own annotation of video_25 has 59. A log of
// visits is a scroll, not a display. What IS near-monotonic is the order of
// first visits — truth for video_25 gives 1,2,3,4,8,5,6,7,9,10,14,12 — so a
// fixed canonical order is measured, not imposed.
//
// The decisive argument is absence. video_25 never predicts steps 3, 5, 6, 11,
// 12 or 13. "Durotomy was never detected" is arguably the most clinically
// interesting fact about that case, and a log *cannot express it*: there is no
// row for a thing that did not happen. Same principle as the timeline's dotted
// absent lane — missing data must look different from data, and say so.
//
// Two consequences that follow:
//
//   - A step is not one block. Steps are revisited 13-26 times per case, so a
//     row aggregates ACROSS visits. Sum `duration_s`, never `end_s - start_s`:
//     end_s is inclusive, so subtracting loses one second per visit — seven
//     seconds on video_25's step 8 alone.
//   - Background is not a row. A worklist row implies something you complete,
//     and out-of-patient is not a surgical step. It becomes one footnote line.

import { hms, prob } from './format.js';

const STEPS = Array.from({ length: 14 }, (_, i) => i + 1);

/**
 * Aggregate a case into fourteen rows. Called once per case, never per second.
 */
export function build(doc) {
  const by = new Map(STEPS.map((k) => [k, {
    step: k, name: doc.names.steps[String(k)] || `step ${k}`,
    total: 0, visits: 0, firstStart: null, lastEnd: null,
    confWeighted: 0, segments: [],
  }]));

  let background = { total: 0, visits: 0 };

  for (const s of doc.steps.segments) {
    if (s.step === -1) {
      background.total += s.duration_s;
      background.visits += 1;
      continue;
    }
    const r = by.get(s.step);
    if (!r) continue;                       // defensive: an unknown step id
    r.total += s.duration_s;
    r.visits += 1;
    r.segments.push(s);
    if (r.firstStart === null) r.firstStart = s.start_s;
    r.lastEnd = s.end_s;
    if (s.confidence) r.confWeighted += s.confidence.mean * s.duration_s;
  }

  const rows = STEPS.map((k) => {
    const r = by.get(k);
    r.seen = r.visits > 0;
    r.conf = r.seen && r.confWeighted ? r.confWeighted / r.total : null;
    return r;
  });

  markRegressions(rows);
  return { rows, background, seen: rows.filter((r) => r.seen).length };
}

/**
 * A step the model RETURNED to after a later step had already begun.
 *
 * Not "any temporal overlap" — that marks nearly every row, because steps
 * interleave constantly. This marks the two or three per case where the
 * sequence genuinely went backwards, which is what makes the mark mean
 * something when you see it.
 */
function markRegressions(rows) {
  for (const r of rows) {
    r.regress = r.seen && rows.some(
      (o) => o.seen && o.step > r.step
             && o.firstStart < r.lastEnd && o.firstStart > r.firstStart);
  }
}

export function render(el, model, doc) {
  el.innerHTML = '';
  for (const r of model.rows) {
    const li = document.createElement('li');
    li.className = 'wl-row' + (r.seen ? ' seen click' : ' unseen')
                 + (r.regress ? ' regress' : '');
    li.dataset.step = String(r.step);
    if (!r.seen) li.setAttribute('aria-disabled', 'true');
    li.title = r.regress
      ? `${r.name} — the model returned to this step after a later one had begun`
      : r.name;

    li.innerHTML =
      `<span class="wl-mark"${r.seen ? ` style="background:${
        doc.names.ramp[String(r.step)]}"` : ''}></span>` +
      `<span class="wl-num">${String(r.step).padStart(2, '0')}</span>` +
      `<span class="wl-name">${r.name}</span>` +
      `<span class="wl-meta">` +
        `<span class="wl-time">${r.seen ? hms(r.total) : '—'}</span>` +
        `<span class="wl-visits">${r.visits > 1 ? `×${r.visits}` : ''}</span>` +
        // The confidence NUMBER is in the default view on purpose: a row
        // saying "the model spent 33:22 here" with no quality mark reads as
        // fact. The bar that goes with it is DETAIL — the honesty rule forbids
        // a bare bar, never a bare number.
        `<span class="wl-conf">${r.conf == null ? '' : prob(r.conf)}</span>` +
      `</span>` +
      `<span class="wl-sub"></span>`;
    el.appendChild(li);
  }
}

/** Per second: move one class and write one string. Never rebuild. */
export function update(el, model, doc, t) {
  const segIdx = doc.steps.segAt[t];
  const seg = doc.steps.segments[segIdx];
  const cur = seg && seg.step !== -1 ? seg.step : null;

  for (const li of el.children) {
    const k = Number(li.dataset.step);
    const on = k === cur;
    li.classList.toggle('cur', on);
    if (!on) continue;
    const r = model.rows[k - 1];
    const visit = r.segments.findIndex((s) => s.i === seg.i) + 1;
    li.querySelector('.wl-sub').textContent =
      `${hms(t - seg.start_s)} of ${hms(seg.duration_s)}`
      + (r.visits > 1 ? ` · visit ${visit} of ${r.visits}` : '');
  }
}

/** The footnote: time the model assigned to no step at all. */
export function footnote(model, doc) {
  const b = model.background;
  if (!b.total) return '';
  const pct = Math.round((b.total / doc.video.seconds) * 100);
  return `${hms(b.total)} (${pct}%) assigned to no step, across ${b.visits} `
       + `span${b.visits === 1 ? '' : 's'}. Out-of-patient and between-step time `
       + `is not a procedure step, so it has no row.`;
}

/**
 * Seek target for a row: the next visit at or after `t`, wrapping to the first.
 * That is what makes a row with six visits reachable with one gesture.
 */
export function seekTarget(model, step, t) {
  const r = model.rows[step - 1];
  if (!r || !r.seen) return null;
  const next = r.segments.find((s) => s.start_s > t);
  return (next || r.segments[0]).start_s;
}
