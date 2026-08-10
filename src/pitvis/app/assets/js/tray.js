// Instrument use — a record, not a chart.
//
// `instruments.lanes` ships every tool's intervals and has never been rendered.
// The obvious thing to do with intervals is draw them, and that is the trap:
// suction alone has 201 intervals on video_25 and 319 on video_01. Nineteen
// rows of that is a barcode, and a barcode with labels is the multitrack
// timeline this redesign exists to remove.
//
// So the tray carries NO horizontal geometry mapped to time. It answers "which
// tools, for how long, how fragmented" — a usage record, the way an instrument
// count is a record. Position stays in the one progress bar.
//
// The interval count is the point, not a footnote. A tool picked up 38 times
// is a different clinical story from one held for 41 minutes straight, and
// that distinction is invisible everywhere else in the app.
//
// Where the interval geometry DOES go: clicking a row selects that class, and
// the timeline's existing TOOLS lane lights it up. Nineteen lanes become one
// lane plus a selection — which is the identity/density split the timeline
// already documents, honoured rather than contradicted.

import { hms } from './format.js';

const TOP = 6;

export function render(el, footEl, doc, selected, t) {
  const inst = doc.instruments;
  el.innerHTML = '';
  footEl.textContent = '';

  if (!inst.available) {
    footEl.textContent = inst.reason || 'task 2 was not run for this case';
    return;
  }

  const total = doc.video.seconds;
  const active = new Set([inst.slot1[t], inst.slot2[t]].filter((v) => v != null));
  const lanes = inst.lanes;                       // server-sorted by seconds desc
  const shown = el.dataset.all === '1' ? lanes : lanes.slice(0, TOP);

  for (const lane of shown) {
    const share = lane.seconds / total;
    const li = document.createElement('li');
    li.className = (active.has(lane.id) ? 'on' : '')
                 + (selected === lane.id ? ' sel' : '');
    li.dataset.id = String(lane.id);
    li.title = `${lane.name} — ${lane.intervals.length} separate appearances`;
    li.innerHTML =
      `<span class="tick">&rsaquo;</span>` +
      `<span class="nm">${lane.name}</span>` +
      `<span class="mb"><i style="width:${Math.round(share * 100)}%"></i></span>` +
      `<span class="tv">${hms(lane.seconds)}</span>` +
      `<span class="xn">×${lane.intervals.length}</span>`;
    el.appendChild(li);
  }

  if (lanes.length > TOP && el.dataset.all !== '1') {
    const more = document.createElement('button');
    more.className = 'tray-more';
    more.textContent = `+ ${lanes.length - TOP} MORE`;
    more.addEventListener('click', () => {
      el.dataset.all = '1';
      render(el, footEl, doc, selected, t);
    });
    el.appendChild(more);
  }

  footEl.innerHTML = notes(doc);
}

/**
 * What the record leaves out, said out loud.
 *
 * `capped` marks seconds where more than two classes cleared the threshold and
 * `decide` kept only the top two. A usage record that silently drops the third
 * tool is an incomplete record presented as complete — and this is the first
 * surface where that number has anywhere to go.
 */
function notes(doc) {
  const inst = doc.instruments;
  const out = [];
  const capped = inst.capped ? inst.capped.reduce((a, b) => a + b, 0) : 0;
  if (capped) {
    out.push(`On ${hms(capped)} more than two tools cleared the bar; the label `
           + `is a pair of columns, so only the top two are recorded.`);
  }
  const missing = 19 - inst.lanes.length;
  if (missing > 0) {
    out.push(`${missing} of 19 classes never cleared the bar in this case.`);
  }
  return out.join('<br>');
}

/** Which lane a click landed on, or null. */
export function laneAt(target) {
  const li = target.closest?.('li[data-id]');
  return li ? Number(li.dataset.id) : null;
}
