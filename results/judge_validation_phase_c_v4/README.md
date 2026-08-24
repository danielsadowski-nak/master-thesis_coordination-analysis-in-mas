# Phase C v4 Commands

## QC

uv run python experiments/check_phase_c_annotation_quality.py \
  --sample-root results/judge_validation_phase_c_v4

## Adjudication build

uv run python experiments/prepare_phase_c_adjudication.py \
  results/judge_validation_phase_c_v4/annotation_sheet_blinded_reviewer1.csv \
  results/judge_validation_phase_c_v4/annotation_sheet_blinded_reviewer2.csv \
  --output-csv results/judge_validation_phase_c_v4/annotation_adjudication.csv

## Agreement

uv run python experiments/run_phase_c_agreement.py \
  --adjudication-csv results/judge_validation_phase_c_v4/annotation_adjudication.csv \
  --master-csv results/judge_validation_phase_c_v4/annotation_master.csv \
  --output-dir results/judge_validation_phase_c_v4/agreement
