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

ZONE="${ZONE:-us-central1-a}"
NAME="${NAME:-pitvis-ft}"
EVERY="${EVERY:-120}"          # seconds between checks

[ -n "${BUCKET:-}" ] || { echo "set BUCKET" >&2; exit 1; }

echo "watching $NAME in $ZONE every ${EVERY}s — Ctrl-C to stop watching"
echo "(stopping this script does not stop the job)"

starts=0
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
      starts=$((starts + 1))
      echo
      echo "$(date +%H:%M:%S)  $STATUS with no DONE marker — restarting (#$starts)"
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
      echo
      echo "$(date +%H:%M:%S)  instance $NAME does not exist in $ZONE."
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
