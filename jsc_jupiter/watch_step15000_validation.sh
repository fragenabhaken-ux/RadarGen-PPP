#!/bin/bash
set -euo pipefail
cd /e/project1/nxtaim-1/huber7/repos/RadarGen
while true; do
  JOB_STATE="$(sacct -n -X -j 2792838 --format=State --parsable2 | head -1)"
  case "$JOB_STATE" in
    COMPLETED*) echo 'Step 15000 validation completed; no resubmission.'; exit 0 ;;
    TIMEOUT*)
      echo '30-minute job timed out; resubmitting with one hour and automatic resume.'
      sbatch --parsable jsc_jupiter/FULL_validation_step15000.sbatch
      exit 0 ;;
    FAILED*|CANCELLED*|OUT_OF_MEMORY*|NODE_FAIL*)
      echo "Validation stopped with $JOB_STATE; inspect before resubmitting."; exit 1 ;;
  esac
  sleep 20
done
