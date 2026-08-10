// The analyst layer: ground truth, the score, and what post-processing did.
//
// What is happening *now* no longer lives here — it is burned into the corners
// of the image (burn.js) and aggregated in the worklist (worklist.js). What is
// left is the second question: how well is the model doing, which is only ever
// asked deliberately and therefore only ever rendered behind + DETAIL.
//
// Every honesty rule in this file is load-bearing rather than decorative. The
// step model scores 0.461 and gets 40.5% of seconds right on the case it is
// most often shown with; an interface that looks like an instrument makes any
// number on it read as authority, so anything uncertain, absent or overruled
// has to say so where it is being shown, not in a footnote.

import { prob, upper } from './format.js';

const $ = (id) => document.getElementById(id);

/**
 * How to name the bar a prediction had to clear.
 *
 * Shared rather than duplicated: the burn-in says this over the video and the
 * rail says it here, and a model with per-class thresholds has no single
 * number to quote. Two copies of this line is how that honesty rule dies in
 * one of the two places without anyone noticing.
 */
export function thresholdPhrase(doc) {
  const inst = doc.instruments;
  if (!inst.available) return '—';
  return inst.perClassThresholds ? 'its per-class threshold' : prob(inst.threshold);
}

export function renderStatus(doc, t) {
  renderReference(doc, t);
  renderHold(doc, t);
}

// -- what the consistency constraint did ------------------------------------

function renderHold(doc, t) {
  const s = doc.steps;
  const card = $('alt-card');
  const held = s.hasConfidence && s.held && s.held[t];
  card.hidden = !held;
  if (!held) return;

  const top = s.top1[t];
  $('step-alt').innerHTML =
    `<b>CCI HOLD</b> — the decoder preferred ` +
    `<span style="color:var(--text)">${upper(doc.names.steps[String(top)])}</span> ` +
    `at ${prob(s.top1Prob[t])}, but the consistency constraint is holding the ` +
    `previous step pending 10 s of agreement. The confidence shown is the ` +
    `probability of the step actually displayed, which is why it reads low here.`;
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
      // record the scope leaving the patient. A prediction never can — the
      // model has no class for it.
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
