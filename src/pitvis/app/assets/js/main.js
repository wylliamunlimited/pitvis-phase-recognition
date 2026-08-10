// Wiring. Owns the store; every other module is a pure renderer or an adapter.

import * as api from './api.js';
import * as burn from './burn.js';
import * as tray from './tray.js';
import * as worklist from './worklist.js';
import { follow, phaseOf } from './jobs.js';
import { CanvasHost } from './overlay.js';
import { VideoTimeSource } from './player.js';
import { renderStatus, thresholdPhrase } from './status.js';
import { laneSet, renderTimeline } from './timeline.js';
import { hmsFixed, prob } from './format.js';

const $ = (id) => document.getElementById(id);

const state = {
  caseId: null,
  cases: [],
  doc: null,
  t: -1,
  segIndex: -1,
  iprobs: null,          // 19-way distribution, fetched lazily
  wl: null,              // worklist aggregate, built once per case
  lane: null,            // instrument class selected in the tray, or null
  cacheState: 'ok',      // ok | legacy | absent — see api.listCases
  colors: {},
  // OFF by default. The interface answers "what is happening now" until asked
  // to answer "how well is the model doing", which is a different question at
  // a different moment. Remembered, because someone who wants it wants it every
  // time.
  detail: localStorage.getItem('pitvis.detail') === '1',
};

let clock = null;
let overlay = null;
let tlCtx = null;

// -- boot -------------------------------------------------------------------

async function boot() {
  state.colors = readColors();
  applyDetail();

  overlay = new CanvasHost($('overlay'), $('video'));
  tlCtx = $('tl-canvas').getContext('2d');

  // The burn-in is pinned to the video's rendered rect, and that rect moves for
  // reasons a resize listener never sees: metadata arriving (before which
  // videoWidth is 0 and geometry() is guessing), and the DETAIL toggle growing
  // the footer — which is a 360ms grid transition, so even re-placing on the
  // click would land on the pre-transition geometry. Observing the element
  // itself catches every frame of both.
  new ResizeObserver(() => {
    overlay.resize();
    burn.place(overlay.geometry());
  }).observe($('video'));

  const listing = await api.listCases();
  state.cases = listing.cases;
  state.cacheState = listing.cacheState;
  fillPicker();

  const wanted = new URLSearchParams(location.search).get('case');
  const first = state.cases.find((c) => c.prediction.available) || state.cases[0];
  await open(wanted && state.cases.some((c) => c.case_id === wanted)
    ? wanted : first?.case_id);

  wireControls();
  wireKeyboard();
  addEventListener('resize', onResize);
  $('app').dataset.state = 'ready';
}

/** One source for the palette: CSS owns it, canvas borrows it. */
function readColors() {
  const s = getComputedStyle(document.documentElement);
  const get = (n) => s.getPropertyValue(`--${n}`).trim();
  return Object.fromEntries(
    ['bg', 'surface', 'raised', 'rule', 'text', 'dim', 'faint', 'accent',
     'warn', 'alarm'].map((n) => [n, get(n)]));
}

/** Detail on/off drives the lane set, the footer height and every `.more`. */
function applyDetail() {
  const app = $('app');
  if (state.detail) app.dataset.detail = '1'; else delete app.dataset.detail;
  $('detail-toggle').textContent = state.detail ? '– DETAIL' : '+ DETAIL';
  $('detail-toggle').setAttribute('aria-pressed', String(state.detail));

  const { lanes, height } = laneSet(state.detail);
  document.documentElement.style.setProperty('--tl', `${height + 28}px`);

  const labels = $('lane-labels');
  labels.innerHTML = '';
  for (const l of lanes) {
    if (!l.label) continue;
    const el = document.createElement('span');
    el.textContent = l.label;
    el.style.top = `${14 + l.y + l.h / 2 - 4}px`;
    labels.appendChild(el);
  }
}

function fillPicker() {
  const sel = $('case-select');
  sel.innerHTML = '';
  for (const c of state.cases) {
    const o = document.createElement('option');
    o.value = c.case_id;
    const mark = c.prediction.available ? (c.prediction.stale ? ' *' : '')
      : c.features_cached ? ' (not run)' : ' (no features)';
    o.textContent = c.case_id.replace('video_', 'CASE ') + mark;
    sel.appendChild(o);
  }
}

// -- open a case ------------------------------------------------------------

async function open(id) {
  if (!id) return veil('No cases found. Expected videos at 26531686/video_NN.mp4.');
  state.caseId = id;
  $('case-select').value = id;
  const ref = state.cases.find((c) => c.case_id === id);
  // A training video scores far better than a held-out one — video_02 reads
  // 0.89 frame accuracy against video_25's 0.41 — so showing its numbers
  // without saying why would flatter the model by a wide margin. The split is
  // the single most important caveat on any score in this interface.
  const split = $('split');
  split.textContent = ref?.split ? `[ ${ref.split.toUpperCase()} SPLIT ]` : '';
  split.className = ref?.split === 'train' ? 'split trained' : 'split';
  split.title = ref?.split === 'train'
    ? 'This video was TRAINED ON. Its scores measure fit, not generalisation, '
      + 'and are not comparable to the validation numbers.'
    : ref?.split === 'val'
      ? 'Held out from training. These scores are honest.' : '';

  state.doc = null;
  state.iprobs = null;
  state.wl = null;
  state.lane = null;
  state.t = -1;
  burn.show(false);
  $('tray').dataset.all = '0';

  const video = $('video');
  video.src = `/api/cases/${encodeURIComponent(id)}/video`;
  clock?.stop();
  clock = null;

  try {
    state.doc = await api.loadCase(id);
  } catch (err) {
    if (err.code !== 'no_prediction') return veil(err.message);
    if (state.cacheState === 'legacy') {
      // Do not tell someone to sit through a 20-minute decode when the
      // features are already on disk, one rename away.
      return veil(
        `${err.message}.\n\nThe feature cache on this machine still uses the `
        + `pre-space layout,\nso nothing can find it and every case reads as `
        + `uncached. Migrating\nis a rename, not a re-extraction:`,
        'uv run pitvis-extract --migrate');
    }
    return veil(
      `${err.message}.\n\n` + (ref?.features_cached
        ? 'Its features are already cached, so running it takes about 45 s —\npress RE-RUN, or run it yourself:'
        : 'This video has no cached features, so predicting it means a full\n1 fps decode of the whole file (10-25 min). Run it yourself:'),
      err.hint);
  }

  veil(null);
  const doc = state.doc;

  $('rerun').disabled = false;

  // Once per case, never per second: fourteen rows aggregated from the
  // segments, and the tray built from the lanes.
  state.wl = worklist.build(doc);
  worklist.render($('worklist'), state.wl, doc);
  $('wl-foot').textContent = worklist.footnote(state.wl, doc);
  $('t-seen').textContent = String(state.wl.seen);
  renderProvenance(doc);
  burn.identity(doc, ref);
  burn.show(true);

  clock = new VideoTimeSource(video, doc.video.seconds)
    .onSecond(onSecond)
    .onFrame(onFrame)
    .start();

  onResize();
  onSecond(0);

  // ~500 KB, and only the probability list needs it — so it arrives after the
  // case is already usable rather than delaying first paint.
  fetch(`/api/cases/${encodeURIComponent(id)}/instrument_probs`)
    .then((r) => (r.ok ? r.json() : null))
    .then((j) => {
      if (j && state.caseId === id) {
        state.iprobs = j.probs;
        renderStatus(state.doc, Math.max(0, state.t));
        renderTray();
      }
    })
    .catch(() => {});
}

function veil(message, command) {
  const el = $('veil');
  if (!message) {
    delete el.dataset.on;
    $('rerun').classList.add('more');     // back behind the toggle
    return;
  }
  el.dataset.on = '1';
  const span = el.querySelector('span');
  span.textContent = message;
  if (command) {
    const code = document.createElement('code');
    code.textContent = command;
    span.appendChild(code);
  }
  $('rerun').disabled = false;
  // RE-RUN normally lives in the analyst layer — re-running inference is an
  // analyst action. But this message tells you to press it, so it has to be
  // reachable regardless of the toggle. An instruction pointing at a hidden
  // control is worse than no instruction.
  $('rerun').classList.remove('more');
}

// -- the clock drives everything -------------------------------------------

function onFrame(time) {
  const doc = state.doc;
  if (!doc) return;
  const frac = Math.min(1, time / doc.video.seconds);
  // transform only — no layout, no canvas work, thirty times a second
  $('playhead').style.transform = `translateX(${frac * trackWidth()}px)`;
  // This runs per rAF, but the clock changes at most once a second and the
  // play label only on toggle. Assigning textContent replaces the child text
  // node and invalidates layout whether or not the string differs, so both
  // are guarded — the one place in the app where the write really is hot.
  burn.clock(`${hmsFixed(time)} / ${hmsFixed(doc.video.duration)}`);
  write($('play'), clock?.playing ? 'PAUSE' : 'PLAY');
}

/**
 * The drawable width of the timeline, in CSS pixels.
 *
 * Read off the canvas, never derived. Three places used to compute
 * `track.clientWidth - 20`, which encoded the track's right padding as a magic
 * number in three files' worth of arithmetic — and the moment the label gutter
 * collapses and the canvas gains a left inset, all three are wrong by 20px:
 * the playhead drifts from the pointer and a click seeks to the wrong second.
 * The canvas is positioned by CSS `inset`, so its own clientWidth is the
 * answer under any padding.
 */
function trackWidth() {
  return $('tl-canvas').clientWidth;
}

/** textContent, but only when it would actually change. */
function write(el, value) {
  if (el.textContent !== value) el.textContent = value;
}

function onSecond(t) {
  const doc = state.doc;
  if (!doc) return;
  state.t = t;
  renderStatus(doc, t);
  burn.state(doc, t, thresholdPhrase(doc));
  worklist.update($('worklist'), state.wl, doc, t);
  renderTiles(doc, t);

  const seg = doc.steps.segAt[t];
  if (seg !== state.segIndex) {
    state.segIndex = seg;
    flashStepCard();
    drawTimeline();                    // only the current-segment highlight moved
    if (state.detail) renderTray();    // the ›-marks track what is in view
  }
}

function renderTiles(doc, t) {
  const seg = doc.steps.segments[doc.steps.segAt[t]];
  const step = seg ? seg.step : null;
  write($('t-step'), step == null ? '--' : step === -1 ? '—'
                                         : String(step).padStart(2, '0'));
  write($('t-elapsed'), seg ? `${hmsFixed(t - seg.start_s).replace(/^0:/, '')} in step`
                            : '--:--');

  const s = doc.steps;
  if (!s.hasConfidence) {
    write($('t-conf'), '--');
    write($('t-conf-note'), 'no probabilities recorded');
    return;
  }
  const held = s.held ? !!s.held[t] : false;
  write($('t-conf'), prob(s.confidence[t]));
  // Reads low exactly where the consistency constraint is holding a phase the
  // current frame does not support. That is the signal, not a defect.
  write($('t-conf-note'), held ? 'CCI holding' : 'of the step shown');
  $('t-conf-note').classList.toggle('held', held);
}

/** Which model produced what is on screen. */
function renderProvenance(doc) {
  const el = $('prov');
  const m = doc.steps.model;
  const i = doc.instruments;
  if (!m.recorded && !i.recorded) {
    // The common case, not the edge one: nothing predicted before the variant
    // work carries these tags. Saying so is the honest render, and it doubles
    // as the reason to re-run.
    el.className = 'prov unrecorded';
    el.textContent = 'MODEL NOT RECORDED';
    el.title = 'This prediction predates provenance capture, so which variant '
             + 'and feature space produced it is not recoverable from the '
             + `artifact. Checkpoint on file: ${m.checkpoint || '?'}`
             + `${i.checkpoint ? ` / ${i.checkpoint}` : ''}. Re-run to attach it.`;
    return;
  }
  el.className = 'prov';
  const parts = [m.name, m.variant, m.space].filter(Boolean);
  el.textContent = parts.join(' · ').toUpperCase();
  el.title = [
    `steps: ${[m.name, m.variant, m.space].filter(Boolean).join(' / ') || m.checkpoint}`
      + `${m.width != null ? ` · W=${m.width}` : ''}`
      + `${m.cci != null ? ` · CCI ${m.cci ? 'on' : 'off'}` : ''}`
      + `${m.maskExcluded ? ' · masked' : ''}`,
    `instruments: ${[i.variant, i.space].filter(Boolean).join(' / ') || i.checkpoint || '—'}`,
  ].join('\n');
}

function renderTray() {
  if (!state.doc) return;
  tray.render($('tray'), $('tray-foot'), state.doc, state.lane,
              Math.max(0, state.t));
}

/** A step boundary: the card's brackets pull to accent, then release.
 *
 * The timing lives in CSS (`@keyframes brk-flash`) with the rest of the motion
 * system, so it is one token rather than a JS `setTimeout` racing a CSS
 * transition — and `prefers-reduced-motion` reaches it, which a timer never
 * could. Removing and re-adding the class needs the reflow read between them,
 * or the browser coalesces the pair into no change and the flash fires once.
 */
function flashStepCard() {
  const card = $('worklist-card');
  card.classList.remove('flash');
  void card.offsetWidth;
  card.classList.add('flash');
}

// -- rendering --------------------------------------------------------------

function drawTimeline() {
  if (!state.doc) return;
  const { height } = laneSet(state.detail);
  const cv = $('tl-canvas');
  const w = trackWidth();
  const dpr = devicePixelRatio || 1;
  cv.style.height = `${height}px`;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(height * dpr);
  tlCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  renderTimeline(tlCtx, state.doc, { width: w, height },
                 { colors: state.colors, segIndex: state.segIndex,
                   detail: state.detail, highlightLane: state.lane });
}

function onResize() {
  drawTimeline();       // the video's own rect is handled by the ResizeObserver
}

// -- controls ---------------------------------------------------------------

function wireControls() {
  $('case-select').addEventListener('change', (e) => open(e.target.value));
  $('detail-toggle').addEventListener('click', () => {
    state.detail = !state.detail;
    localStorage.setItem('pitvis.detail', state.detail ? '1' : '0');
    applyDetail();
    drawTimeline();
    if (state.doc) {
      renderStatus(state.doc, Math.max(0, state.t));
      renderTiles(state.doc, Math.max(0, state.t));
      if (state.detail) renderTray();
    }
  });
  $('play').addEventListener('click', () => clock?.toggle());
  $('back10').addEventListener('click', () => clock?.nudge(-10));
  $('fwd10').addEventListener('click', () => clock?.nudge(10));
  $('prevseg').addEventListener('click', () => jumpSegment(-1));
  $('nextseg').addEventListener('click', () => jumpSegment(1));
  $('rerun').addEventListener('click', rerun);
  $('console-close').addEventListener('click', () => { $('console').hidden = true; });

  // A worklist row seeks to that step's NEXT visit at or after now, wrapping to
  // the first. That is what makes a step visited six times reachable with one
  // gesture and no modifier.
  $('worklist').addEventListener('click', (e) => {
    const li = e.target.closest?.('li[data-step]');
    if (!li || !state.wl || !clock) return;
    const to = worklist.seekTarget(state.wl, Number(li.dataset.step), state.t);
    if (to != null) clock.seek(to);
  });

  // Selecting a tool lights its intervals on the timeline's TOOLS lane —
  // nineteen tracks collapsed to one lane plus a selection.
  $('tray').addEventListener('click', (e) => {
    const id = tray.laneAt(e.target);
    if (id == null) return;
    state.lane = state.lane === id ? null : id;
    renderTray();
    drawTimeline();
  });

  const track = $('track');
  track.addEventListener('click', (e) => {
    if (!state.doc) return;
    clock?.seek(secondAt(e));
  });
  track.addEventListener('mousemove', (e) => {
    if (!state.doc) return;
    const t = secondAt(e);
    const seg = state.doc.steps.segments[state.doc.steps.segAt[t]];
    const hover = $('tl-hover');
    hover.hidden = false;
    hover.style.left = `${e.clientX - track.getBoundingClientRect().left}px`;
    hover.textContent = `${hmsFixed(t)}  ${seg
      ? `[${seg.step === -1 ? '--' : String(seg.step).padStart(2, '0')}] ` +
        state.doc.names.steps[String(seg.step)] : ''}`;
  });
  track.addEventListener('mouseleave', () => { $('tl-hover').hidden = true; });
}

function secondAt(e) {
  // Against the CANVAS rect, not the track's: the canvas is where the bar
  // actually is, and its left inset changes when the label gutter collapses.
  const r = $('tl-canvas').getBoundingClientRect();
  const frac = (e.clientX - r.left) / r.width;
  return Math.max(0, Math.min(state.doc.video.seconds - 1,
                              Math.floor(frac * state.doc.video.seconds)));
}

function jumpSegment(dir) {
  const doc = state.doc;
  if (!doc || !clock) return;
  const segs = doc.steps.segments;
  const i = doc.steps.segAt[state.t];
  // Going back from mid-segment returns to this segment's start first, which
  // is what "previous boundary" means when scrubbing.
  const target = dir < 0
    ? (state.t - segs[i].start_s > 1 ? i : Math.max(0, i - 1))
    : Math.min(segs.length - 1, i + 1);
  clock.seek(segs[target].start_s);
}

function wireKeyboard() {
  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || !clock) return;
    const k = e.key;
    if (k === ' ') { e.preventDefault(); clock.toggle(); }
    else if (k === 'ArrowLeft') { e.preventDefault(); clock.nudge(e.shiftKey ? -10 : -1); }
    else if (k === 'ArrowRight') { e.preventDefault(); clock.nudge(e.shiftKey ? 10 : 1); }
    else if (k === '[') jumpSegment(-1);
    else if (k === ']') jumpSegment(1);
    else if (k === 'f') $('frame').requestFullscreen?.();
  });
}

// -- on-demand inference ----------------------------------------------------

async function rerun() {
  const id = state.caseId;
  const log = $('console-log');
  $('console').hidden = false;
  log.textContent = '';
  $('rerun').disabled = true;

  let jobId;
  try {
    jobId = await api.startPredict(id);
  } catch (err) {
    log.textContent = `${err.message}\n\n${err.hint ? `Run this instead:\n  ${err.hint}` : ''}`;
    $('rerun').disabled = false;
    return;
  }

  follow(jobId, {
    onLine: (line) => {
      const phase = phaseOf(line);
      if (phase) log.insertAdjacentHTML('beforeend', `<b>${phase}</b>\n`);
      log.insertAdjacentText('beforeend', `${line}\n`);
      log.scrollTop = log.scrollHeight;
    },
    onDone: async () => {
      $('rerun').disabled = false;
      log.insertAdjacentText('beforeend', '\n— reloading case —\n');
      const listing = await api.listCases();
      state.cases = listing.cases;
      state.cacheState = listing.cacheState;
      fillPicker();
      await open(id);
    },
  });
}

boot().catch((err) => {
  console.error(err);
  $('app').dataset.state = 'ready';
  veil(`Could not start: ${err.message}`);
});
