// The ONLY module that touches raw response keys.
//
// There is no build step and therefore no type checking at this boundary, so
// the discipline replacing it is that every raw key appears exactly once, here.
// A schema change breaks one file loudly instead of six files subtly.

export const SCHEMA = 1;

export class SchemaError extends Error {}

async function get(url) {
  const r = await fetch(url, { headers: { Accept: 'application/json' } });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const e = body.error || {};
    const err = new Error(e.message || `${r.status} ${url}`);
    err.code = e.code;
    err.hint = e.hint;
    err.status = r.status;
    throw err;
  }
  return body;
}

export async function listCases() {
  const r = await get('/api/cases');
  // `legacy` means the features are on disk but in the pre-space layout, so
  // nothing can find them — a different problem from having none, and a
  // different instruction.
  return { cases: r.cases, cacheState: r.cache_state || 'ok' };
}

export async function loadCase(id) {
  return parseCase(await get(`/api/cases/${encodeURIComponent(id)}`));
}

export async function startPredict(id) {
  const r = await fetch(`/api/cases/${encodeURIComponent(id)}/predict`,
                        { method: 'POST' });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const e = body.error || {};
    const err = new Error(e.message || 'could not start');
    err.hint = e.hint;
    throw err;
  }
  return body.job_id;
}

/**
 * Raw document -> the shape the renderers consume.
 *
 * Two things are derived here and never recomputed later:
 *
 *  - `segAt`, an Int32Array mapping every second to its segment index. Built
 *    once at load; a binary search per animation frame would be wasted work
 *    thirty times a second for the whole length of a case.
 *  - per-second step labels, reconstructed from the segments. They are a
 *    lossless run-length encoding of the same column, so sending 4,337 rows
 *    over the wire as well would be pure duplication.
 */
export function parseCase(j) {
  if (j.schema_version !== SCHEMA) {
    throw new SchemaError(
      `case document is schema ${j.schema_version}, this UI speaks ${SCHEMA}`);
  }

  const seconds = j.video.seconds;
  const p = j.prediction;
  const ps = p.per_second || {};

  const doc = {
    id: j.case_id,
    split: j.split,
    generatedAt: j.generated_at,
    video: {
      url: j.video.url,
      seconds,
      duration: j.video.duration_s,
      width: j.video.width,
      height: j.video.height,
      fps: j.video.fps,
      faststart: j.video.faststart,
    },
    names: {
      steps: j.labels.steps,
      instruments: j.labels.instruments,
      ramp: j.labels.ramp,
    },
    steps: {
      segments: p.segments,
      segAt: indexSegments(p.segments, seconds),
      step: ps.step,
      confidence: ps.confidence || null,
      top1: ps.top1_step || null,
      top1Prob: ps.top1_prob || null,
      held: ps.cci_held || null,
      hasConfidence: !!p.confidence_meta?.available,
      heldFrac: p.confidence_meta?.held_frac ?? 0,
      caveat: p.confidence_meta?.caveat || '',
      // Mapped key by key rather than passed through. `p.model?.task1 || {}`
      // handed a raw sub-object downstream, so every renderer reading
      // `.mask_excluded` was touching a response key outside this function —
      // the one discipline that stands in for type checking here.
      model: mapModel(p.model?.task1),
      stale: !!p.stale,
      computedAt: p.computed_at,
    },
    instruments: parseInstruments(j.instruments, seconds),
    truth: parseTruth(j.truth, seconds),
    scores: j.scores || { available: false },
    // Seams. Present and empty, so renderers branch on contents not existence.
    corrections: j.corrections,
    explanations: j.explanations,
    live: j.live,
  };
  return doc;
}

/**
 * The task-1 model card.
 *
 * `name`, `variant` and `space` are null for anything predicted before the
 * variant work landed, which is most of what exists. That is reported, not
 * hidden — with four checkpoint families and three feature spaces, and every
 * v2 checkpoint named `model.pt`, a bare filename cannot say what produced the
 * numbers on screen.
 */
function mapModel(m) {
  m = m || {};
  return {
    name: m.name || null,
    variant: m.variant || null,
    space: m.space || null,
    checkpoint: m.checkpoint || null,
    width: m.width ?? null,
    cci: m.cci ?? null,
    maskExcluded: m.mask_excluded ?? null,
    recorded: !!(m.name || m.variant || m.space),
  };
}

function indexSegments(segments, seconds) {
  const at = new Int32Array(seconds);
  for (let i = 0; i < segments.length; i++) {
    const s = segments[i];
    at.fill(i, s.start_s, Math.min(s.end_s + 1, seconds)); // end_s is INCLUSIVE
  }
  return at;
}

function parseInstruments(inst, seconds) {
  if (!inst || !inst.available) {
    return { available: false, reason: inst?.reason || 'task 2 not run' };
  }
  const q = inst.per_second;
  return {
    available: true,
    threshold: inst.threshold,
    // null unless the checkpoint carries one bar per class, in which case
    // there is no single number to quote to the viewer.
    perClassThresholds: inst.per_class_thresholds || null,
    checkpoint: inst.checkpoint || null,
    variant: inst.variant || null,
    space: inst.space || null,
    classesPredicted: inst.classes_predicted ?? null,
    recorded: !!(inst.variant || inst.space),
    note: inst.note,
    lanes: inst.lanes || [],
    // 'none' means nothing cleared the threshold. It is NOT out-of-patient —
    // SANO has no such class. The server resolved that; nothing downstream
    // should ever see the raw (-1, -2) pair.
    state: q.state,
    slot1: q.slot1,
    slot2: q.slot2,
    conf1: q.conf1 || null,
    conf2: q.conf2 || null,
    maxProb: q.max_prob || null,
    maxClass: q.max_class || null,
    capped: q.capped || null,
    count: q.state.map((s) => (s === 'two' ? 2 : s === 'one' ? 1 : 0)),
  };
}

function parseTruth(t, seconds) {
  if (!t || !t.available) {
    return { available: false, reason: t?.reason || 'no ground truth' };
  }
  return {
    available: true,
    source: t.source,
    segments: t.segments,
    segAt: indexSegments(t.segments, seconds),
    step: t.per_second.step,
    instState: t.per_second.inst_state || null,
    instSlot1: t.per_second.inst_slot1 || null,
    instSlot2: t.per_second.inst_slot2 || null,
    agreement: t.agreement.step,
    frameAccuracy: t.agreement.frame_accuracy,
  };
}
