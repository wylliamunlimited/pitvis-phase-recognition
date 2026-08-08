// The rail: what the model believes right now, and how sure it is.
//
// Every honesty rule in this file is load-bearing rather than decorative. The
// model scores 0.331; an interface that looks like an instrument makes that
// number read as authority, so anything uncertain, absent or overruled has to
// say so where it is being shown, not in a footnote.

import { hms, prob, upper } from './format.js';

const $ = (id) => document.getElementById(id);
const TOP_N = 5;

/** Write text, and animate the write when the value actually changed.
 *
 * `renderStatus` runs on every tick, so the equality check is doing two jobs:
 * it stops the swap animation from re-triggering once a second on a value that
 * has not moved, and it stops the DOM being written at all when nothing did.
 *
 * The reflow read is not removable. Dropping and re-adding a class inside one
 * task is coalesced by the browser into no change at all, so the animation
 * would restart on the first boundary and never again — which looks exactly
 * like a bug that only appears after the first step change.
 *
 * Values that move every second (elapsed, confidence) must NOT come through
 * here: a step card that pulses once a second is worse than one that snaps.
 */
function setText(el, value) {
  if (el.textContent === value) return;
  el.textContent = value;
  // A hidden tab PAUSES css animations. The swap opens at opacity 0, so a
  // value written while backgrounded would sit invisible until the tab came
  // back — a blank step card on the one surface whose whole job is to say
  // what is happening. Measured: currentTime frozen at 19ms, opacity 0.006.
  // The readable state must never depend on an animation having run.
  if (document.hidden) return;
  el.classList.remove('swap');
  void el.offsetWidth;
  el.classList.add('swap');
}

export function renderStatus(doc, t, extra = {}) {
  renderStep(doc, t);
  renderInstruments(doc, t, extra.iprobs);
  renderReference(doc, t);
}

// -- current step -----------------------------------------------------------

function renderStep(doc, t) {
  const s = doc.steps;
  const seg = s.segments[s.segAt[t]];
  if (!seg) return;

  const step = seg.step;
  // Background is -1 in the challenge encoding, which is not a number worth
  // setting in 68px type. "BG" is the abbreviation of the name already shown
  // beside it, not a new label.
  setText($('step-num'), step === -1 ? 'BG' : String(step).padStart(2, '0'));
  setText($('step-name'), upper(doc.names.steps[String(step)] || 'unknown'));
  $('step-tint').style.background = doc.names.ramp[String(step)] || 'var(--faint)';
  $('step-elapsed').textContent = hms(t - seg.start_s);
  $('step-total').textContent = hms(seg.duration_s);

  // The CCI-hold marker lives in the detail layer, not beside the step name.
  // It says the displayed label is being held by post-processing against the
  // model's current belief — a real caveat, but one that belongs with the
  // confidence it qualifies rather than in the glanceable line.
  const held = s.held ? !!s.held[t] : false;

  const row = $('conf-row');
  if (!s.hasConfidence) {
    row.style.display = 'none';
    $('step-alt').hidden = true;
    return;
  }
  row.style.display = '';
  const c = s.confidence[t];
  $('conf-fill').style.width = `${Math.round(c * 100)}%`;
  // Always a number as well as a bar. A bar alone invites reading "mostly
  // full" as "confident" when the value is 0.31.
  $('conf-val').textContent = prob(c);
  $('conf-fill').style.background =
    held ? 'var(--warn)' : c < 0.4 ? 'var(--dim)' : 'var(--accent)';

  const alt = $('step-alt');
  if (held) {
    const top = s.top1[t];
    alt.hidden = false;
    alt.innerHTML =
      `<b>CCI HOLD</b> — the decoder preferred ` +
      `<span style="color:var(--text)">${upper(doc.names.steps[String(top)])}</span> ` +
      `at ${prob(s.top1Prob[t])}, but the consistency constraint is holding the ` +
      `previous step pending 10 s of agreement. Confidence above is the ` +
      `probability of the step actually shown.`;
  } else {
    alt.hidden = true;
  }
}

// -- instruments ------------------------------------------------------------

function renderInstruments(doc, t, iprobs) {
  const inst = doc.instruments;
  const one = $('inst1'), two = $('inst2'), list = $('inst-probs');

  if (!inst.available) {
    setText(one, 'task 2 not run');
    one.className = 'v empty';
    setText(two, '--');
    two.className = 'v empty';
    list.innerHTML = '';
    return;
  }

  const state = inst.state[t];
  if (state === 'none') {
    // NOT "out of patient". SANO's head is 19 sigmoids with no such class, so
    // this can only mean nothing cleared the bar. Showing the runner-up makes
    // the difference between "nothing there" and "nearly something" visible.
    const best = inst.maxClass ? doc.names.instruments[String(inst.maxClass[t])] : null;
    const p = inst.maxProb ? inst.maxProb[t] : null;
    one.className = 'v empty';
    setText(one, `nothing above ${prob(inst.threshold)}`);
    two.className = 'v empty';
    // Three decimals here specifically: the runner-up sits just under the
    // threshold by definition, and at two decimals a 0.498 prints as "0.50"
    // directly beside "nothing above 0.50", which reads as a contradiction.
    setText(two, best ? `closest: ${best} ${p.toFixed(3)}` : '--');
  } else {
    one.className = 'v';
    setText(one, upper(doc.names.instruments[String(inst.slot1[t])] || '--'));
    const s2 = inst.slot2[t];
    two.className = s2 == null ? 'v empty' : 'v';
    setText(two, s2 == null ? 'none' : upper(doc.names.instruments[String(s2)]));
  }

  list.innerHTML = '';
  const active = new Set([inst.slot1[t], inst.slot2[t]].filter((v) => v != null));

  let rows;
  if (iprobs) {
    rows = iprobs[t]
      .map((p, id) => ({ id, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, TOP_N);
  } else {
    // Before the 19-way distribution arrives, show what the document carries.
    rows = [...active].map((id) => ({
      id, p: id === inst.slot1[t] ? inst.conf1?.[t] : inst.conf2?.[t],
    }));
  }

  for (const r of rows) {
    const li = document.createElement('li');
    if (active.has(r.id)) li.className = 'on';
    li.innerHTML =
      `<span class="tick">&rsaquo;</span>` +
      `<span class="nm">${doc.names.instruments[String(r.id)]}</span>` +
      `<span class="mb"><i style="width:${Math.round((r.p ?? 0) * 100)}%"></i></span>` +
      `<span class="pv">${prob(r.p)}</span>`;
    list.appendChild(li);
  }
}

// -- ground truth -----------------------------------------------------------

function renderReference(doc, t) {
  const el = $('ref-body');
  const truth = doc.truth;

  if (!truth.available) {
    el.innerHTML = `<div class="absent">${truth.reason}</div>`;
    return;
  }

  const ts = truth.step[t];
  const ok = truth.agreement[t] === 1;
  const inst = truth.instState ? truth.instState[t] : null;

  let html =
    `<div class="row"><span class="tstep">` +
    `${ts === -1 ? '--' : String(ts).padStart(2, '0')} ` +
    `${upper(doc.names.steps[String(ts)])}</span>` +
    `<span class="verdict ${ok ? 'ok' : 'bad'}">${ok ? 'MATCH' : 'MISMATCH'}</span></div>`;

  if (inst) {
    const label = inst === 'out_of_patient'
      // The one place this phrase is legitimate: the annotations really do
      // record the scope leaving the patient. A prediction never can.
      ? 'scope out of patient'
      : [truth.instSlot1[t], truth.instSlot2[t]]
          .filter((v) => v != null)
          .map((v) => doc.names.instruments[String(v)])
          .join(' + ') || '--';
    html += `<div class="row"><span>${label}</span></div>`;
  }

  const sc = doc.scores;
  if (sc.available && sc.steps && !sc.steps.error) {
    html +=
      `<div class="scores">` +
      `<div><span>steps · challenge metric</span><span>${sc.steps.metric.toFixed(3)}</span></div>` +
      `<div><span>steps · frame accuracy</span><span>${sc.steps.frame_accuracy.toFixed(3)}</span></div>` +
      (sc.instruments && !sc.instruments.error
        ? `<div><span>instruments · official</span><span>${sc.instruments.metric.toFixed(3)}</span></div>` +
          `<div><span>instruments · name-aligned</span><span>${sc.instruments.weighted.toFixed(3)}</span></div>`
        : '') +
      `</div>` +
      `<div class="absent" style="margin-top:8px">${sc.scope}${
        doc.split === 'train'
          ? '. This video was TRAINED ON — these measure fit, not generalisation.'
          : ''}</div>`;
  }
  el.innerHTML = html;
}
