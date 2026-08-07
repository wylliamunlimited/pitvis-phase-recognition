// Streaming console for on-demand inference.
//
// The server pipes `pitvis-predict`'s own stdout through as server-sent events,
// so what appears here is the CLI's output verbatim — the official scoring
// table, the leaked-class note, the instrument column-divergence warning. None
// of it is parsed. There is deliberately no progress bar: the command prints a
// cache-hit line, decodes silently for ~40 s, then prints a timing line, so any
// percentage would be invented.

export function follow(jobId, { onLine, onState, onDone }) {
  const es = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);

  es.addEventListener('line', (e) => onLine?.(e.data));
  es.addEventListener('state', (e) => onState?.(e.data));
  es.addEventListener('done', (e) => { onDone?.(e.data); es.close(); });
  es.addEventListener('error', () => {
    // The server closes the stream when the job ends, which surfaces here as
    // an error. Only report it if we never reached a terminal state.
    if (es.readyState === EventSource.CLOSED) onDone?.('closed');
  });

  return () => es.close();
}

/** Coarse phase from a log line — milestones, since no fraction exists. */
export function phaseOf(line) {
  if (line.startsWith('features')) return 'EMBEDDING';
  if (line.startsWith('task 1')) return 'STEPS';
  if (line.startsWith('task 2')) return 'INSTRUMENTS';
  if (line.startsWith('wrote')) return 'WRITING';
  return null;
}
