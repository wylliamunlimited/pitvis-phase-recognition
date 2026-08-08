// Wiring. Owns the store; every other module is a pure renderer or an adapter.

import * as api from './api.js';
import { follow, phaseOf } from './jobs.js';
import { CanvasHost } from './overlay.js';
import { VideoTimeSource } from './player.js';
import { renderStatus } from './status.js';
import { laneSet, renderTimeline } from './timeline.js';
import { hmsFixed } from './format.js';

const $ = (id) => document.getElementById(id);

const state = {
  caseId: null,
  cases: [],
  doc: null,
  t: -1,
  segIndex: -1,
  iprobs: null,          // 19-way distribution, fetched lazily
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

  state.cases = await api.listCases();
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
  state.t = -1;

  const video = $('video');
  video.src = `/api/cases/${encodeURIComponent(id)}/video`;
  clock?.stop();
  clock = null;

  try {
    state.doc = await api.loadCase(id);
  } catch (err) {
    if (err.code !== 'no_prediction') return veil(err.message);
    return veil(
      `${err.message}.\n\n` + (ref?.features_cached
        ? 'Its features are already cached, so running it takes about 45 s —\npress RE-RUN, or run it yourself:'
        : 'This video has no cached features, so predicting it means a full\n1 fps decode of the whole file (10-25 min). Run it yourself:'),
      err.hint);
  }

  veil(null);
  const doc = state.doc;

  $('rerun').disabled = false;

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
        renderStatus(state.doc, Math.max(0, state.t), { iprobs: state.iprobs });
      }
    })
    .catch(() => {});
}

function veil(message, command) {
  const el = $('veil');
  if (!message) { delete el.dataset.on; return; }
  el.dataset.on = '1';
  const span = el.querySelector('span');
  span.textContent = message;
  if (command) {
    const code = document.createElement('code');
    code.textContent = command;
    span.appendChild(code);
  }
  $('rerun').disabled = false;
}

// -- the clock drives everything -------------------------------------------

function onFrame(time) {
  const doc = state.doc;
  if (!doc) return;
  const track = $('track');
  const w = track.clientWidth - 20;
  const frac = Math.min(1, time / doc.video.seconds);
  // transform only — no layout, no canvas work, thirty times a second
  $('playhead').style.transform = `translateX(${frac * w}px)`;
  $('clock').textContent =
    `${hmsFixed(time)} / ${hmsFixed(doc.video.duration)}`;
  $('play').textContent = clock?.playing ? 'PAUSE' : 'PLAY';
}

function onSecond(t) {
  const doc = state.doc;
  if (!doc) return;
  state.t = t;
  renderStatus(doc, t, { iprobs: state.iprobs });

  const seg = doc.steps.segAt[t];
  if (seg !== state.segIndex) {
    state.segIndex = seg;
    flashStepCard();
    drawTimeline();                    // only the current-segment highlight moved
  }
}

/** The one animation in the product: brackets pull to accent, then release. */
function flashStepCard() {
  const card = $('step-card');
  card.style.setProperty('--brk', state.colors.accent);
  clearTimeout(flashStepCard._t);
  flashStepCard._t = setTimeout(
    () => card.style.setProperty('--brk', state.colors.rule), 420);
}

// -- rendering --------------------------------------------------------------

function drawTimeline() {
  if (!state.doc) return;
  const { height } = laneSet(state.detail);
  const w = $('track').clientWidth - 20;
  const dpr = devicePixelRatio || 1;
  const cv = $('tl-canvas');
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(height * dpr);
  cv.style.width = `${w}px`;
  cv.style.height = `${height}px`;
  tlCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  renderTimeline(tlCtx, state.doc, { width: w, height },
                 { t: state.t, colors: state.colors, segIndex: state.segIndex,
                   detail: state.detail });
}

function onResize() {
  drawTimeline();
  overlay.resize();
}

// -- controls ---------------------------------------------------------------

function wireControls() {
  $('case-select').addEventListener('change', (e) => open(e.target.value));
  $('detail-toggle').addEventListener('click', () => {
    state.detail = !state.detail;
    localStorage.setItem('pitvis.detail', state.detail ? '1' : '0');
    applyDetail();
    drawTimeline();
    if (state.doc) renderStatus(state.doc, Math.max(0, state.t),
                                { iprobs: state.iprobs });
  });
  $('play').addEventListener('click', () => clock?.toggle());
  $('back10').addEventListener('click', () => clock?.nudge(-10));
  $('fwd10').addEventListener('click', () => clock?.nudge(10));
  $('prevseg').addEventListener('click', () => jumpSegment(-1));
  $('nextseg').addEventListener('click', () => jumpSegment(1));
  $('rerun').addEventListener('click', rerun);
  $('console-close').addEventListener('click', () => { $('console').hidden = true; });

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
  const r = $('track').getBoundingClientRect();
  const frac = (e.clientX - r.left) / (r.width - 20);
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
      state.cases = await api.listCases();
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
