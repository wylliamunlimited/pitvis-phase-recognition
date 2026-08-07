// The clock, behind an interface.
//
// Everything downstream asks "what second is it?" and is told. Nothing else in
// the app touches the <video> element. That is the seam for streaming input
// (roadmap 5.8): a StreamTimeSource with the same four methods swaps in, and no
// renderer changes, because none of them know a file is involved.

export class VideoTimeSource {
  /** @param {HTMLVideoElement} el @param {number} seconds row count, authoritative */
  constructor(el, seconds) {
    this.el = el;
    this.seconds = seconds;
    this._t = -1;
    this._onSecond = () => {};
    this._onFrame = () => {};
    this._raf = 0;
  }

  /**
   * Integer second, clamped to the label count.
   *
   * The clamp is load-bearing, not defensive: `video.duration` is 4337.42 for
   * video_25 while there are 4337 label rows, so the final fraction of a
   * second would index past the end of every per-second array.
   */
  get t() {
    return Math.min(Math.floor(this.el.currentTime || 0), this.seconds - 1);
  }

  get time() { return this.el.currentTime || 0; }
  get duration() { return this.el.duration || this.seconds; }
  get playing() { return !this.el.paused && !this.el.ended; }

  seek(t) { this.el.currentTime = Math.max(0, Math.min(t, this.duration - 0.01)); }
  nudge(dt) { this.seek(this.time + dt); }
  toggle() { this.el.paused ? this.el.play() : this.el.pause(); }

  onSecond(fn) { this._onSecond = fn; return this; }
  onFrame(fn) { this._onFrame = fn; return this; }

  /**
   * Drive from requestAnimationFrame, never from `timeupdate`.
   *
   * `timeupdate` fires at an irregular 4-66 Hz and lags the real position, so a
   * playhead driven by it visibly stutters. rAF reads the exact current time at
   * the moment the browser is about to paint.
   */
  start() {
    const tick = () => {
      this._onFrame(this.time);
      const t = this.t;
      if (t !== this._t) { this._t = t; this._onSecond(t); }
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
    return this;
  }

  stop() { cancelAnimationFrame(this._raf); }

  /** Force a re-emit — used after loading a case, when t hasn't moved. */
  refresh() { this._t = -1; }
}
