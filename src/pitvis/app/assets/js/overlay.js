// A canvas over the video, and the layer registry that draws on it.
//
// Nothing registers a layer today — the corner brackets around the video are
// CSS. This exists because roadmap 5.4 is an agent that circles a region of the
// frame and captions it, and "circle a region" is the same bracket primitive
// the rest of the interface is built from, just moved and resized. Having the
// host, the sizing and the coordinate space settled now means that work is a
// layer, not a refactor.
//
// Coordinates are in VIDEO pixels (1280x720) and mapped to the displayed rect
// here, so a layer never has to know how the element is letterboxed.

export class CanvasHost {
  constructor(canvas, video) {
    this.canvas = canvas;
    this.video = video;
    this.layers = [];
    this.ctx = canvas.getContext('2d');
    this._raf = 0;
  }

  register(layer) {
    this.layers.push(layer);
    this.layers.sort((a, b) => (a.z || 0) - (b.z || 0));
    this.invalidate();
    return () => {
      this.layers = this.layers.filter((l) => l !== layer);
      this.invalidate();
    };
  }

  /** The video's rendered content box — object-fit letterboxes it. */
  geometry() {
    const box = this.video.getBoundingClientRect();
    const host = this.canvas.getBoundingClientRect();
    const vw = this.video.videoWidth || 1280;
    const vh = this.video.videoHeight || 720;
    const scale = Math.min(box.width / vw, box.height / vh) || 1;
    const w = vw * scale, h = vh * scale;
    return {
      left: box.left - host.left + (box.width - w) / 2,
      top: box.top - host.top + (box.height - h) / 2,
      width: w, height: h, scale, videoWidth: vw, videoHeight: vh,
    };
  }

  /**
   * The bitmap follows layout. Layout must never follow the bitmap.
   *
   * Measured from the PARENT, not from the canvas itself. A canvas is a
   * replaced element, so its CSS box falls back to its bitmap attribute unless
   * something else sizes it — measuring its own rect therefore closes a loop
   * that multiplies by devicePixelRatio on every call. On a 2x display it
   * doubled per resize until the bitmap passed what Chrome will allocate and
   * the element painted opaque white over the video. `#overlay` also carries an
   * explicit width/height in CSS; both halves are needed, because either one
   * alone leaves the element sized by the thing it is supposed to size.
   */
  resize() {
    const r = (this.canvas.parentElement || this.canvas).getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.invalidate();
  }

  invalidate(state) {
    if (state !== undefined) this.state = state;
    cancelAnimationFrame(this._raf);
    this._raf = requestAnimationFrame(() => this.draw());
  }

  draw() {
    const { ctx, canvas } = this;
    const dpr = window.devicePixelRatio || 1;
    ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);
    if (!this.layers.length) return;
    const geom = this.geometry();
    for (const layer of this.layers) {
      ctx.save();
      layer.draw(ctx, geom, this.state);
      ctx.restore();
    }
  }
}
