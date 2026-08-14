#!/usr/bin/env bash
# Restart the instance after a spot preemption, until the job is done.
#
#     BUCKET=gs://your-private-bucket infra/babysit.sh
#
# WHY THIS EXISTS. A preempted spot VM is STOPPED, not restarted. Nothing in
# GCE brings it back — `--provisioning-model=SPOT` buys the discount and the
# eviction, not the recovery. So "it resumes automatically" is only true if
# something is watching, and this is that something. Run it in a terminal, or
# under nohup, on any machine with gcloud.
#
# It is safe to stop and restart this script at any time: all the state lives
# in the bucket and on the instance's boot disk, never here.
#
# WHAT MAKES A RESTART CHEAP rather than a rerun:
#   * finished encoders were uploaded as each one completed (run_job.sh)
#   * the encoder in flight has resume.pt, written every epoch, on the boot
#     disk — which survives because termination action is STOP, not DELETE
#   * the frame cache is on that same disk, so no 3.6 GB re-download
# Worst case is the epoch in flight.
#
# WHAT THIS DELIBERATELY DOES NOT DO: give up. A spot instance can be evicted
# repeatedly when a zone is tight. If that is happening, the log will show it,
# and the answer is a different zone or on-demand — not a smarter loop here.

set -uo pipefail

NAME="${NAME:-pitvis-ft}"
EVERY="${EVERY:-120}"          # seconds between checks
# Consecutive restarts that produce no new encoder before we stop restarting.
# See the stall check below for why "don't give up" has an exception.
MAX_STALLED="${MAX_STALLED:-3}"

[ -n "${BUCKET:-}" ] || { echo "set BUCKET" >&2; exit 1; }

# FIND THE INSTANCE, DO NOT ASSUME ITS ZONE.
#
# launch.sh walks eight zones and stops at whichever has L4 capacity — GPU
# stockouts are routine, so the fallback firing is expected rather than rare.
# A hardcoded default here therefore describes an empty zone whenever it did,
# gets nothing back, and reports the instance as deleted. That is the same
# mistake launch.sh already carries a comment about having made, and the
# consequence is worse on this side: the watcher exits, the next preemption
# goes unhandled, and the job stalls while the disk keeps billing.
find_zone() {
  gcloud compute instances list --filter="name=$NAME" \
    --format="value(zone.basename())" 2>/dev/null | head -1
}

ZONE="${ZONE:-$(find_zone)}"
if [ -z "$ZONE" ]; then
  echo "no instance named '$NAME' in any zone of this project." >&2
  echo "  Create one:  BUCKET=$BUCKET infra/launch.sh" >&2
  echo "  Or set NAME= if yours is called something else." >&2
  exit 1
fi

# How much work is already banked. Compared across restarts to tell "preempted
# mid-encoder, resuming" from "finished, but the DONE marker never landed".
progress() { gsutil ls -r "$BUCKET/out/backbone/**" 2>/dev/null | wc -l | tr -d ' '; }

echo "watching $NAME in $ZONE every ${EVERY}s — Ctrl-C to stop watching"
echo "(stopping this script does not stop the job)"

starts=0
stalled=0
banked="$(progress)"
while true; do
  if gsutil -q stat "$BUCKET/out/DONE" 2>/dev/null; then
    echo
    echo "=== DONE marker found — the job finished ==="
    gsutil cat "$BUCKET/out/DONE" 2>/dev/null || true
    echo
    echo "Pull the results, then delete the instance to stop paying for its disk:"
    echo "  gsutil -m rsync -r $BUCKET/out/backbone data/backbone"
    echo "  gcloud compute instances delete $NAME --zone=$ZONE"
    exit 0
  fi

  STATUS=$(gcloud compute instances describe "$NAME" --zone="$ZONE" \
             --format="value(status)" 2>/dev/null || echo "MISSING")

  case "$STATUS" in
    RUNNING|STAGING|PROVISIONING)
      printf "\r%s  %-12s (restarts: %d)   " "$(date +%H:%M:%S)" "$STATUS" "$starts"
      ;;
    TERMINATED|STOPPED|SUSPENDED)
      # No DONE marker and not running: preempted, or hit max-run-duration.
      # Either way the fix is the same, and startup.sh re-runs from the top.
      #
      # BUT: DONE is written with `|| echo` in run_job.sh, so a bucket
      # permission change makes it non-fatal AND unwritable — the job completes,
      # stops, and this loop restarts a finished job forever, at an accelerator
      # per cycle. "Never give up" was written about preemption, where every
      # restart resumes real work. It was not written about this.
      #
      # So the exception is narrow: give up only when consecutive restarts bank
      # no new encoder, which is the signature of a job with nothing left to do.
      now="$(progress)"
      if [ "$now" = "$banked" ]; then
        stalled=$((stalled + 1))
      else
        stalled=0
        banked="$now"
      fi
      if [ "$stalled" -ge "$MAX_STALLED" ]; then
        echo
        echo "=== STOPPING: $stalled restarts produced no new encoder ==="
        echo "The job is almost certainly finished and could not write the DONE"
        echo "marker — run_job.sh treats that write as non-fatal, so the run"
        echo "succeeds and this watcher never learns of it. Restarting again"
        echo "would rent a GPU to redo nothing."
        echo
        echo "Check what actually landed:"
        echo "  gsutil ls -r $BUCKET/out/backbone"
        echo "  gcloud compute ssh $NAME --zone=$ZONE -- tail -50 /var/log/pitvis-job.log"
        echo
        echo "If the encoders are there, pull them and delete the instance —"
        echo "its boot disk bills whether or not the instance is running:"
        echo "  gsutil -m rsync -r $BUCKET/out/backbone data/backbone"
        echo "  gcloud compute instances delete $NAME --zone=$ZONE"
        exit 1
      fi
      starts=$((starts + 1))
      echo
      echo "$(date +%H:%M:%S)  $STATUS with no DONE marker — restarting (#$starts)"
      [ "$stalled" -gt 0 ] && echo "           WARNING: no new encoder since the last restart ($stalled/$MAX_STALLED)"
      if gcloud compute instances start "$NAME" --zone="$ZONE" >/dev/null 2>&1; then
        echo "           started; it will resume from the last completed epoch"
      else
        # Usually means the zone has no spot capacity right now. Keep trying:
        # capacity returns, and every attempt is free.
        echo "           start FAILED (most likely no spot capacity in $ZONE)."
        echo "           retrying in ${EVERY}s; if this persists, try another"
        echo "           zone or drop --provisioning-model=SPOT for the last run."
      fi
      ;;
    MISSING)
      # Look project-wide before believing it. A zone-scoped describe answers
      # "not in THIS zone", which is not the same question — and an explicit
      # ZONE= that disagrees with where launch.sh actually landed produces
      # exactly this, for an instance that is alive and running.
      ELSEWHERE="$(find_zone)"
      if [ -n "$ELSEWHERE" ] && [ "$ELSEWHERE" != "$ZONE" ]; then
        echo
        echo "$(date +%H:%M:%S)  $NAME is in $ELSEWHERE, not $ZONE — following it."
        ZONE="$ELSEWHERE"
        continue
      fi
      echo
      echo "$(date +%H:%M:%S)  instance $NAME does not exist in any zone."
      echo "  If you deleted it, its boot disk went too — resume.pt and the frame"
      echo "  cache with it. Re-create with infra/launch.sh; finished encoders are"
      echo "  still in $BUCKET/out/backbone and will be skipped."
      exit 1
      ;;
    *)
      printf "\r%s  %-12s   " "$(date +%H:%M:%S)" "$STATUS"
      ;;
  esac
  sleep "$EVERY"
done
