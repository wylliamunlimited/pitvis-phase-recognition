// The timeline. A pure function of its arguments — it reads no globals.
//
// That purity is not style. It is what makes two future features cheap:
// rendering a corrected timeline is passing different segments, and comparing
// cases side by side is calling this N times with N documents against a shared
// axis. A renderer that reached for module state would have to be rewritten for
// either.
//
// Canvas rather than SVG because of the dense lanes, not the segments. Thirty
// segment rectangles would be fine as DOM; a confidence trace, an agreement
// strip and an instrument band at one column per second are ~13,000 elements
// for a 72-minute case, which is a paint catastrophe on every resize. Canvas
// draws the whole thing in one pass and is redrawn only on resize, case change
// or a step boundary — the playhead is a separate DOM element moved by
// transform, so the per-frame path touches no canvas at all.

// Two lane sets, not one with things hidden.
//
// The default is a single strip: where are we in the operation. Six stacked
// channels is what a video editor looks like, and it invites reading the
// timeline as a workspace rather than as an answer. Confidence, ground truth
// and agreement answer "how did the model do", which is a different question
// asked at a different moment — so they arrive only when asked for.
// One strip, no ruler, no label. Where you are in the operation and nothing
// else. The ruler is gone from the default because absolute time is now an
// exact numeral burned into the video's top-right corner, which strictly beats
// reading a value off a tick scale.
const MINIMAL = [
  { key: 'predicted', label: '', y: 0, h: 16 },
];

const DETAIL = [
  { key: 'ruler', label: '', y: 0, h: 12 },
  { key: 'predicted', label: 'STEP', y: 18, h: 30 },
  { key: 'confidence', label: 'CONFIDENCE', y: 56, h: 22 },
  { key: 'reference', label: 'TRUTH', y: 86, h: 14 },
  { key: 'agreement', label: 'ERRORS', y: 106, h: 4 },
  { key: 'instruments', label: 'TOOLS', y: 120, h: 14 },
];

export function laneSet(detail) {
  const lanes = detail ? DETAIL : MINIMAL;
  const last = lanes[lanes.length - 1];
  return { lanes, height: last.y + last.h };
}

const TICKS = [30, 60, 120, 300, 600, 900, 1800, 3600, 7200];

/**
 * @param ctx   2D context, already scaled for devicePixelRatio
 * @param doc   parsed case document
 * @param geom  {width, height} in CSS pixels
 * @param opts  {colors, segIndex, detail, highlightLane}
 *
 * `highlightLane` is an instrument class id, or null. It rides in `opts`
 * rather than being read from module state — that is what keeps this a pure
 * function, and a pure function is what makes multi-case comparison N calls
 * instead of a rewrite.
 */
export function renderTimeline(ctx, doc, geom, opts) {
  const { width: W } = geom;
  const { colors: C, segIndex, detail, highlightLane } = opts;
  const N = doc.video.seconds;
  if (!N || W <= 0) return;

  const { lanes } = laneSet(detail);
  const lane = (k) => lanes.find((l) => l.key === k);

  ctx.clearRect(0, 0, W, geom.height);
  // Sans, not mono. These segment numbers are the only numerals the default
  // timeline draws, and a monospace label under a sans interface was a gap
  // that predates this redesign.
  ctx.font = '400 8.5px ui-sans-serif, -apple-system, "SF Pro Text", system-ui, sans-serif';
  ctx.textBaseline = 'middle';

  const x = (s) => (s / N) * W;

  if (detail) drawRuler(ctx, lane('ruler'), W, N, x, C);
  drawSegments(ctx, doc.steps.segments, lane('predicted'), x, doc.names.ramp, C,
               { current: segIndex, labels: true });
  if (!detail) return;

  drawConfidence(ctx, doc, lane('confidence'), W, N, C);

  if (doc.truth.available) {
    drawSegments(ctx, doc.truth.segments, lane('reference'), x, doc.names.ramp, C,
                 { alpha: 0.85 });
    drawAgreement(ctx, doc.truth.agreement, lane('agreement'), W, N, C);
  } else {
    absent(ctx, lane('reference'), W, C, 'NO GROUND TRUTH — THIS CASE IS UNLABELLED');
    absent(ctx, lane('agreement'), W, C, '');
  }

  drawInstruments(ctx, doc.instruments, lane('instruments'), W, N, C, highlightLane);
}

// -- lanes ------------------------------------------------------------------

function drawRuler(ctx, l, W, N, x, C) {
  const step = TICKS.find((s) => (N / s) * 40 < W) || TICKS[TICKS.length - 1];
  ctx.fillStyle = C.faint;
  ctx.textAlign = 'left';
  for (let s = 0; s <= N; s += step) {
    const px = Math.round(x(s)) + 0.5;
    ctx.fillRect(px, l.y + l.h - 4, 1, 4);
    if (px < W - 34) {
      ctx.fillText(clock(s), px + 4, l.y + l.h - 8);
    }
  }
}

function drawSegments(ctx, segments, l, x, ramp, C, o = {}) {
  ctx.save();
  if (o.alpha) ctx.globalAlpha = o.alpha;
  for (let i = 0; i < segments.length; i++) {
    const s = segments[i];
    const x0 = x(s.start_s);
    // end_s is INCLUSIVE, so the right edge is the START of the next second.
    const x1 = x(s.end_s + 1);
    const w = Math.max(1, x1 - x0 - 1);          // 1px gap: boundaries visible
    const cur = o.current === i;
    ctx.fillStyle = cur ? C.accent : (ramp[String(s.step)] || C.rule);
    ctx.fillRect(x0, l.y, w, l.h);

    if (o.labels && w > 17) {
      ctx.fillStyle = cur ? C.bg : C.text;
      ctx.globalAlpha = cur ? 1 : 0.72;
      ctx.textAlign = 'center';
      ctx.fillText(s.step === -1 ? '·' : String(s.step), x0 + w / 2, l.y + l.h / 2);
      ctx.globalAlpha = o.alpha || 1;
    }
  }
  ctx.restore();
}

function drawConfidence(ctx, doc, l, W, N, C) {
  ctx.fillStyle = C.raised;
  ctx.fillRect(0, l.y + l.h - 1, W, 1);

  const conf = doc.steps.confidence;
  if (!conf) {
    absent(ctx, l, W, C, 'NO CONFIDENCE — RE-RUN WITH --probs');
    return;
  }
  const held = doc.steps.held;
  // One column per pixel, aggregating the seconds that fall inside it. At a
  // 72-minute case that is ~3 s per column; taking the MINIMUM rather than the
  // mean keeps a brief collapse in confidence visible instead of averaging it
  // away, which is the whole reason to look at this lane.
  for (let px = 0; px < W; px++) {
    const s0 = Math.floor((px / W) * N);
    const s1 = Math.max(s0 + 1, Math.floor(((px + 1) / W) * N));
    let lo = 1, anyHeld = false;
    for (let s = s0; s < s1 && s < N; s++) {
      if (conf[s] < lo) lo = conf[s];
      if (held && held[s]) anyHeld = true;
    }
    const h = Math.max(1, lo * l.h);
    ctx.fillStyle = anyHeld ? C.warn : C.faint;
    ctx.globalAlpha = anyHeld ? 0.95 : 0.42;
    ctx.fillRect(px, l.y + l.h - h, 1, h);
  }
  ctx.globalAlpha = 1;
}

function drawAgreement(ctx, agree, l, W, N, C) {
  ctx.fillStyle = C.alarm;
  for (let px = 0; px < W; px++) {
    const s0 = Math.floor((px / W) * N);
    const s1 = Math.max(s0 + 1, Math.floor(((px + 1) / W) * N));
    let wrong = 0, n = 0;
    for (let s = s0; s < s1 && s < N; s++, n++) if (!agree[s]) wrong++;
    if (!wrong) continue;
    ctx.globalAlpha = 0.25 + 0.75 * (wrong / Math.max(1, n));
    ctx.fillRect(px, l.y, 1, l.h);
  }
  ctx.globalAlpha = 1;
}

function drawInstruments(ctx, inst, l, W, N, C, highlight) {
  if (!inst.available) {
    absent(ctx, l, W, C, 'TASK 2 NOT RUN FOR THIS CASE');
    return;
  }
  // A class selected in the tray takes over the lane. This is where the
  // per-tool interval geometry lives — one lane plus a selection, instead of
  // nineteen stacked tracks. The density view is still drawn underneath, so
  // the selection reads as a highlight rather than as a different chart.
  if (highlight != null) {
    const lane = inst.lanes.find((x) => x.id === highlight);
    if (lane) {
      ctx.fillStyle = C.raised;
      ctx.fillRect(0, l.y, W, l.h);
      ctx.fillStyle = C.accent;
      for (const [a, b] of lane.intervals) {
        const x0 = (a / N) * W;
        const w = Math.max(1, ((b + 1) / N) * W - x0);   // end is INCLUSIVE
        ctx.fillRect(x0, l.y, w, l.h);
      }
      return;
    }
  }
  // Density, not identity: how many tools are in view, in the same lightness
  // language the phase ramp uses. Identity lives in the rail, where there is
  // room to name a thing; nineteen colour-coded rows here would be unreadable
  // at 16px and would need nineteen more colours.
  //
  // Averaged, not peaked. At ~3 s per pixel the count flickers between one and
  // two constantly, and taking the maximum turns that into a barcode that
  // draws the eye without telling it anything. The mean reads as tool activity
  // rising and falling, which is the actual shape of the data.
  const count = inst.count;
  ctx.fillStyle = C.dim;
  for (let px = 0; px < W; px++) {
    const s0 = Math.floor((px / W) * N);
    const s1 = Math.max(s0 + 1, Math.floor(((px + 1) / W) * N));
    let sum = 0, n = 0;
    for (let s = s0; s < s1 && s < N; s++, n++) sum += count[s];
    const mean = n ? sum / n : 0;
    ctx.globalAlpha = 0.12 + 0.55 * (mean / 2);
    ctx.fillRect(px, l.y, 1, l.h);
  }
  ctx.globalAlpha = 1;
}

// -- absence, stated ---------------------------------------------------------

function absent(ctx, l, W, C, text) {
  // An empty lane would read as "all background" or "nothing detected". Missing
  // data has to look different from data, and say why.
  ctx.save();
  ctx.fillStyle = C.raised;
  for (let x = 0; x < W; x += 6) ctx.fillRect(x, l.y + l.h / 2, 3, 1);
  if (text) {
    ctx.fillStyle = C.faint;
    ctx.textAlign = 'left';
    ctx.fillRect(0, l.y, 1, l.h);
    ctx.fillText(text, 6, l.y + l.h / 2);
  }
  ctx.restore();
}

function clock(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  const p = (n) => String(n).padStart(2, '0');
  return h ? `${h}:${p(m)}:${p(s % 60)}` : `${p(m)}:${p(s % 60)}`;
}
