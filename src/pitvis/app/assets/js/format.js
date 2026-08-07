// Formatting only. No state, no DOM.

export const pad = (n) => String(n).padStart(2, '0');

/** Seconds -> H:MM:SS, or MM:SS under an hour. */
export function hms(s) {
  s = Math.max(0, Math.floor(s || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${pad(m)}:${pad(s % 60)}`;
}

/** Always H:MM:SS, so the header clock never changes width mid-case. */
export function hmsFixed(s) {
  s = Math.max(0, Math.floor(s || 0));
  return `${Math.floor(s / 3600)}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
}

/** A probability as two decimals — never a bare bar, never a rounded integer. */
export const prob = (p) => (p == null || Number.isNaN(p) ? '--' : p.toFixed(2));

export const upper = (s) => (s || '').toUpperCase();
