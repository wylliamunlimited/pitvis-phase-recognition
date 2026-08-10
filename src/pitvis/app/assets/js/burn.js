// The burn-in: case identity and live state, in the corners of the image.
//
// PACS and endoscopy viewers burn this into the pixels. Doing the same here is
// not a costume — it collapses four saccades (header clock, header case,
// step card, instrument card) into the fixation the eye is already holding on
// the video, and that is what let the rail drop two cards.
//
// DOM ABOVE THE CANVAS, NEVER PAINTED INTO IT. The #overlay canvas belongs to
// the agentic explanation layer (roadmap 5.4) and works in VIDEO pixels — text
// that scaled with the video would be 6px in a small window. Corner text wants
// screen pixels at a fixed size. Mixing the two conventions on one surface
// would ruin the layer registry before it has a single layer. DOM also keeps
// the text selectable, translatable and reachable by a screen reader, and
// avoids a canvas redraw every second for four short strings.
//
// It does share the GEOMETRY: overlay.geometry() already computes the
// letterboxed video rect, and at 1440x900 that matters — the video renders
// 1031x580 inside a 1031x688 frame, so ~54px of paper sits above and below.
// Corners pinned to the frame would float off the image.

const $ = (id) => document.getElementById(id);

/** Position the burn layer over the video's rendered rect. */
export function place(geom) {
  const el = $('burn');
  if (!el || !geom) return;
  el.style.left = `${geom.left}px`;
  el.style.top = `${geom.top}px`;
  el.style.width = `${geom.width}px`;
  el.style.height = `${geom.height}px`;
}

export function identity(doc, ref) {
  $('b-case').textContent = doc.id.replace('video_', 'CASE ').toUpperCase();
  const split = ref?.split || doc.split;
  $('b-split').textContent = split ? `${split.toUpperCase()} SPLIT` : '';
}

export function clock(text) {
  // Written every rAF. Plain textContent, never the animated setText — four
  // surfaces pulsing in unison once a second is worse than one.
  const el = $('b-clock');
  if (el.textContent !== text) el.textContent = text;
}

export function state(doc, t, thresholdPhrase) {
  const seg = doc.steps.segments[doc.steps.segAt[t]];
  const step = seg ? seg.step : null;
  const el = $('b-step');
  const label = step == null ? '--'
    : step === -1 ? '[--] BETWEEN STEPS'
    : `[${String(step).padStart(2, '0')}] ${doc.names.steps[String(step)]}`;
  if (el.textContent !== label) el.textContent = label;

  const inst = doc.instruments;
  const out = $('b-inst');
  let text, empty = false;
  if (!inst.available) {
    text = 'instruments not run'; empty = true;
  } else if (inst.state[t] === 'none') {
    // Never "out of patient" — this model has no such class. The phrase comes
    // from the shared helper so the per-class-threshold wording cannot drift
    // between here and the rail.
    const best = inst.maxClass ? doc.names.instruments[String(inst.maxClass[t])] : null;
    const p = inst.maxProb ? inst.maxProb[t] : null;
    text = `nothing above ${thresholdPhrase}`
         + (best ? ` · closest ${best} ${p.toFixed(3)}` : '');
    empty = true;
  } else {
    text = [inst.slot1[t], inst.slot2[t]]
      .filter((v) => v != null)
      .map((v) => doc.names.instruments[String(v)])
      .join(' + ');
  }
  if (out.textContent !== text) out.textContent = text;
  out.classList.toggle('empty', empty);
}

export function show(on) {
  const el = $('burn');
  if (el) el.hidden = !on;
}
