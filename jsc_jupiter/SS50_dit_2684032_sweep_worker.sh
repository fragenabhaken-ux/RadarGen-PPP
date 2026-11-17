#!/usr/bin/env bash
set -euo pipefail
MANIFEST=evaluation/configs/SS50_dit_2684032_checkpoint_sweep/manifest.json
INDEX="$((SLURM_ARRAY_TASK_ID * 4 + SLURM_PROCID))"
STEP="$(python - "$MANIFEST" "$INDEX" <<'PY'
import json,sys
entries=[e for e in json.load(open(sys.argv[1]))['entries'] if not e['reused']]
i=int(sys.argv[2])
print(f"{entries[i]['step']:06d}" if i<len(entries) else '')
PY
)"
[[ -n "$STEP" ]] || exit 0
ROOT=/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_sweep
REPORT=/e/scratch/nxtaim-1/huber7/work_dirs_RadarGen/validation/SS50_dit_2684032_checkpoint_comparison.md
mkdir -p "$ROOT"
update_report() {
  (
    flock 9
    python scripts/compare_ss50_checkpoint_sweep.py --manifest "$MANIFEST" --output "$REPORT"
    cp "$REPORT" docs/SS50_dit_2684032_checkpoint_comparison.md
  ) 9>"$ROOT/comparison.lock"
}
trap update_report EXIT
ARGS=()
if [[ -f "$ROOT/step_$STEP/validation_config.json" ]]; then ARGS+=(--resume); fi
python -u evaluation/scripts/validate.py \
  --config "evaluation/configs/SS50_dit_2684032_checkpoint_sweep/step_${STEP}.yaml" \
  --output-dir "$ROOT/step_$STEP" "${ARGS[@]}"
