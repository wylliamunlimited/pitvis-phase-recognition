// Floating panels: draggable, independently collapsible, position remembered.
//
// WHY NOT A RAIL. A fixed column stacks categories vertically, so the moment
// the content exceeds the viewport it scrolls — and scrolling is the one
// interaction that guarantees you cannot see two things at once. The pairing
// that matters here is exactly that: which step the model thinks it is in, and
// which instruments it can see. Those are the two halves of one judgement, and
// a column asks you to hold one in memory while you scroll to the other.
//
// Panels solve it by letting the reader decide what is open. Collapsed, a
// panel costs one header row, so five categories fit in the height of one card
// and any two of them can be open together.
//
// WHY THEY MAY SIT OVER THE IMAGE. The endoscope's circle does not fill the
// 16:9 frame. Measured over the 5 validation videos at 3 timestamps each, the
// left gutter is optical black for at least 217 of 1280 px (17.0%) in every
// frame sampled. The right gutter is NOT reliably dead — video_01 runs out to
// x=1176, leaving only 8% — so the defaults park panels on the pillarbox paper
// beside the video and let the reader move them onto the gutter if they want
// the image bigger. Nothing is pinned over tissue by default.

const KEY = 'pitvis.panels';
const MARGIN = 14;       // gap from the layer's edges for default placement
const KEEPALIVE = 56;    // px of a panel that must stay on screen after a drag

const readSaved = () => {
  try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
};

export class Panels {
  /** @param {HTMLElement} layer the positioned container the panels live in */
  constructor(layer) {
    this.layer = layer;
    this.saved = readSaved();
    this.items = [];
    this.z = 10;

    for (const el of layer.querySelectorAll('.panel')) this._add(el);
    this.layout();
    addEventListener('resize', () => this.layout());

    // `alt-card` is shown and hidden by the renderer as CCI holds come and go.
    // Observing the attribute keeps that renderer ignorant of the layout — it
    // sets `hidden` and the stack closes up on its own.
    new MutationObserver(() => this.layout())
      .observe(layer, { attributes: true, attributeFilter: ['hidden'], subtree: true });
  }

  _add(el) {
    const id = el.dataset.panel;
    // Read the markup's default BEFORE any saved state overwrites the
    // attribute, or the first reset() has nothing left to restore to.
    const item = { id, el, head: el.querySelector('.phead'), dragged: false,
                   byDefaultCollapsed: el.hasAttribute('data-collapsed') };
    this.items.push(item);

    // Saved state wins; failing that the markup's own `data-collapsed` is the
    // default. Normalising through _setFold keeps aria-expanded and the title
    // honest whichever of the two supplied the value.
    const st = this.saved[id] || {};
    this._setFold(item, st.collapsed ?? el.hasAttribute('data-collapsed'));
    if (st.x != null && st.y != null) item.dragged = true;

    el.querySelector('.fold').addEventListener('click', (e) => {
      e.stopPropagation();
      this._setFold(item, !el.hasAttribute('data-collapsed'));
      this._save();
      this.layout();          // un-dragged neighbours close the gap
    });
    item.head.addEventListener('pointerdown', (e) => this._drag(item, e));
  }

  _setFold(item, collapsed) {
    const btn = item.el.querySelector('.fold');
    if (collapsed) item.el.setAttribute('data-collapsed', ''); else item.el.removeAttribute('data-collapsed');
    btn.setAttribute('aria-expanded', String(!collapsed));
    btn.title = `${collapsed ? 'expand' : 'collapse'} (${this.items.indexOf(item) + 1})`;
  }

  /**
   * Drag by the header.
   *
   * Pointer capture rather than window listeners: it survives the cursor
   * leaving the panel mid-drag, which happens constantly when you throw a panel
   * at an edge, and it releases itself if the pointer is cancelled.
   */
  _drag(item, e) {
    if (e.button !== 0) return;
    const el = item.el;
    const host = this.layer.getBoundingClientRect();
    const box = el.getBoundingClientRect();
    const offX = e.clientX - box.left, offY = e.clientY - box.top;

    el.style.zIndex = String(++this.z);
    el.setAttribute('data-dragging', '');
    item.dragged = true;      // from here it keeps its own position, not the stack's
    // Capture is an optimisation — it keeps the drag alive when the cursor
    // outruns the panel. If the pointer is not capturable the drag must still
    // work, so a failure here is not allowed to abort the rest of this method.
    try { item.head.setPointerCapture(e.pointerId); } catch { /* uncapturable */ }

    const move = (ev) => {
      this._place(item,
        ev.clientX - host.left - offX,
        ev.clientY - host.top - offY);
    };
    const up = () => {
      item.head.removeEventListener('pointermove', move);
      item.head.removeEventListener('pointerup', up);
      item.head.removeEventListener('pointercancel', up);
      el.removeAttribute('data-dragging');
      this._save();
    };
    item.head.addEventListener('pointermove', move);
    item.head.addEventListener('pointerup', up);
    item.head.addEventListener('pointercancel', up);
    e.preventDefault();
  }

  /**
   * Position in layer coordinates, clamped so the header stays reachable, and
   * capped so the panel cannot extend past the bottom of the layer.
   *
   * The cap is what keeps the promise this whole layer exists to make. Without
   * it a tall panel — PROCEDURE STEPS is fourteen rows — is simply cut off by
   * the timeline, which is worse than the column it replaced: at least a column
   * could be scrolled to reach the end.
   */
  _place(item, x, y) {
    const b = this._bounds();
    const box = item.el.getBoundingClientRect();
    const top = clamp(y, 0, b.height - 26);
    item.el.style.left = `${clamp(x, KEEPALIVE - box.width, b.width - KEEPALIVE)}px`;
    item.el.style.top = `${top}px`;
    item.el.style.maxHeight = `${Math.max(64, b.height - top - MARGIN)}px`;
    item.el.dataset.placed = '1';
  }

  /**
   * The usable box: the layer, minus the transport strip.
   *
   * Panels take pointer events, so one covering PLAY does not merely hide the
   * control — it eats the click. The strip is measured rather than hard-coded
   * because its height follows the button metrics, and a constant here would
   * drift silently the first time those change.
   */
  _bounds() {
    const host = this.layer.getBoundingClientRect();
    const tr = document.querySelector('.transport');
    const keep = tr ? Math.max(0, host.bottom - tr.getBoundingClientRect().top) : 0;
    return { width: host.width, height: Math.max(120, host.height - keep) };
  }

  /**
   * Apply saved positions, and give anything unplaced its default.
   *
   * Defaults are computed rather than written into the stylesheet because they
   * depend on the panel's own measured height — stacking `data-top` order down
   * an edge needs the heights of the panels above it, which CSS cannot see.
   */
  layout() {
    const b = this._bounds();
    if (!b.width) return;
    const run = { left: MARGIN, right: MARGIN };

    for (const item of this.items) {
      // A panel the analyst layer is hiding has no box, so it must not consume
      // a slot in the stack — otherwise turning DETAIL on drops three panels
      // onto the same coordinates.
      if (!rendered(item.el)) continue;
      const st = this.saved[item.id] || {};
      const side = item.el.dataset.side === 'right' ? 'right' : 'left';

      if (st.x != null && st.y != null) {
        this._place(item, st.x, st.y);
      } else {
        const x = side === 'left' ? MARGIN : b.width - item.el.offsetWidth - MARGIN;
        this._place(item, x, run[side]);
      }
      run[side] += item.el.offsetHeight + 10;
    }
  }

  /** Re-clamp everything after a layout change (DETAIL toggle, resize). */
  reflow() { this.layout(); }

  reset() {
    this.saved = {};
    localStorage.removeItem(KEY);
    for (const item of this.items) {
      item.el.style.left = item.el.style.top = item.el.style.zIndex = '';
      item.el.style.maxHeight = '';
      item.dragged = false;
      delete item.el.dataset.placed;
      this._setFold(item, item.byDefaultCollapsed);
    }
    this.layout();
  }

  /**
   * Merge, never replace, and record coordinates only for panels the reader
   * actually moved.
   *
   * Two reasons. A panel the analyst layer is hiding has no box to read, so
   * writing the whole set would record it at 0,0 and it would reappear in the
   * corner next time DETAIL came on. And pinning an untouched panel to wherever
   * it happened to sit would freeze the auto-stack — collapsing its neighbour
   * would then leave a hole instead of closing up.
   */
  _save() {
    const host = this.layer.getBoundingClientRect();
    for (const item of this.items) {
      const { id, el } = item;
      if (!rendered(el)) continue;
      const st = { collapsed: el.hasAttribute('data-collapsed') };
      if (item.dragged) {
        const box = el.getBoundingClientRect();
        st.x = Math.round(box.left - host.left);
        st.y = Math.round(box.top - host.top);
      }
      this.saved[id] = st;
    }
    try {
      localStorage.setItem(KEY, JSON.stringify(this.saved));
    } catch { /* private mode — the layout is simply not remembered */ }
  }
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));
const rendered = (el) => el.offsetWidth > 0 || el.offsetHeight > 0;
