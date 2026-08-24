# Phase C Data Card (v4)

## Purpose

This data card documents the current Phase C reviewer package used for human-vs-judge validation.

Primary objective:

- build a double-coded reference set for H5-level agreement analysis.

## Current cycle status

Current active sample root:

- `results/judge_validation_phase_c_v4`

Deprecated for H5:

- all v3 `balanced_k2` success-only assets under `results/judge_validation_phase_c_cli_v3`

## Source lineage and exclusions

Source experiment root:

- `results/coordination_baseline_2026-08-09/experiments`

Excluded frameworks:

- `metagpt`

Excluded runtime failures:

- timeout-like outputs containing `timed out before completion`
- excluded count: `60`

No cross-root mixing was used.

## Sampling design

Manifest:

- `results/judge_validation_phase_c_v4/annotation_manifest.json`

Design parameters:

- seed: `42`
- requested sample size: `60`
- realized sample size: `60`
- stratification success label: `criteria_success` (not runner `success`)
- `success_only=false`

Post-filter class balance:

- `n_success=26`
- `n_failure=34`

Framework counts:

- crewai: `26`
- langgraph: `19`
- autogen: `15`

## Unit of analysis

One row equals one run instance with immutable metadata and evidence text:

- `annotation_item_id`
- `framework`, `benchmark`, `run_index`, `run_id`
- `raw_log_path`
- `final_output`

## Blinding policy

Reviewer sheets are strictly blinded against automated labels.

Removed from blinded reviewer sheets:

- `success`
- `judge_task_successful`
- `judge_primary_failure_modes`
- `judge_summary`

Reviewer sheets retain only operational metadata, output evidence, and empty manual/adjudication fields.

## Files

Non-reviewer files:

- template: `results/judge_validation_phase_c_v4/annotation_template.csv`
- master: `results/judge_validation_phase_c_v4/annotation_master.csv`

Reviewer files:

- reviewer1 sheet: `results/judge_validation_phase_c_v4/annotation_sheet_blinded_reviewer1.csv`
- reviewer2 sheet: `results/judge_validation_phase_c_v4/annotation_sheet_blinded_reviewer2.csv`
- worklist: `results/judge_validation_phase_c_v4/annotation_worklist.csv`
- forms output folder: `results/judge_validation_phase_c_v4/review_forms/`

## Cell balance caveat

The manifest includes `unbalanced_cells` and must be checked before interpretation.

Interpretation rule:

- unbalanced cells are disclosed as design limitations;
- H5 subgroup claims should not over-interpret framework-task cells lacking both classes.

## QC policy for this stage

Current stage is pre-annotation shipping.

Allowed now:

- `--allow-incomplete` smoke QC
- missing adjudication file as warning

Not allowed yet:

- strict final QC claims before both reviewer sheets are filled
- agreement reporting before adjudication

Smoke command:

```bash
uv run python experiments/check_phase_c_annotation_quality.py \
  --sample-root results/judge_validation_phase_c_v4 \
  --allow-incomplete
```

## H5 reporting guardrails

- Runtime failures are not part of the annotation set.
- MetaGPT is excluded in this cycle.
- Success stratification uses `criteria_success` for sample balancing.
- Any interpretation must reference `unbalanced_cells` from the v4 manifest.
