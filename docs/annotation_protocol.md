# Annotation Protocol And Judge Validation

This document defines how traces are manually reviewed to validate the automated MAST judge.

## 1) Annotation unit

One annotation unit equals one completed run with:

- trace log path
- final output
- automated task-success label
- automated primary failure-mode set

## 2) Required annotator outputs

For each run, the reviewer should provide:

- `manual_task_successful`: `true` or `false`
- `manual_primary_failure_modes`: semicolon-separated MAST labels
- `manual_summary`: one short evidence summary
- `reviewer_id`
- `notes`

Optional adjudication columns are used when disagreements are resolved:

- `adjudicated_task_successful`
- `adjudicated_primary_failure_modes`
- `adjudicated_summary`

If adjudicated columns are present and non-empty, they are treated as the reference labels for agreement analysis.

## 3) Annotation workflow

1. Generate an annotation template from an existing results directory.
2. Sample traces stratified across frameworks and benchmark tasks.
3. Annotate runs independently using the Coordination Suite codebook.
4. Resolve disagreements in an adjudication pass.
5. Run the judge-validation script to produce agreement tables.

For double-coding, first build an adjudication table from both reviewer sheets:

```bash
uv run python experiments/prepare_phase_c_adjudication.py \
  results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded.csv \
  results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded_reviewer2.csv \
  --output-csv results/judge_validation_phase_c_cli_v3/annotation_adjudication_balanced_k2.csv
```

Interpretation of the adjudication helper columns:

- `needs_annotation=true`: at least one reviewer has not completed labels for that row.
- `needs_adjudication=true`: both reviewer labels are present and they disagree on task success and/or primary modes.

Compute the final agreement report from adjudication artifacts (QC preflight is executed automatically):

```bash
uv run python experiments/run_phase_c_agreement.py \
  --adjudication-csv results/judge_validation_phase_c_cli_v3/annotation_adjudication_balanced_k2.csv \
  --master-csv results/judge_validation_phase_c_cli_v3/annotation_master_balanced_k2.csv \
  --output-dir results/judge_validation_phase_c_cli_v3/agreement_balanced_k2
```

Run annotation quality-control before the strict agreement run:

```bash
uv run python experiments/check_phase_c_annotation_quality.py \
  --reviewer1-csv results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded.csv \
  --reviewer2-csv results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded_reviewer2.csv \
  --adjudication-csv results/judge_validation_phase_c_cli_v3/annotation_adjudication_balanced_k2.csv \
  --output-dir results/judge_validation_phase_c_cli_v3/agreement_balanced_k2
```

QC outputs:

- `annotation_qc_summary.json`: completeness and pass/fail status.
- `annotation_qc_issues.csv`: row-level format/content violations.
- `annotation_missing_manifest.csv`: per-reviewer list of rows with missing labels/evidence.
- `annotation_progress_matrix.csv`: completion rates by reviewer x framework x benchmark.
- `annotation_worklist.csv`: deterministic, block-balanced annotation order per reviewer.
- `adjudication_backlog_manifest.csv`: rows that require adjudication due to disagreement.

Independent-reviewer HTML forms can be generated from the blinded reviewer sheets:

```bash
uv run python experiments/generate_phase_c_review_form.py \
  --reviewer-csv results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded.csv \
  --worklist-csv results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/annotation_worklist.csv \
  --reviewer-tag reviewer1 \
  --output-html results/judge_validation_phase_c_cli_v3/review_forms/reviewer1_form.html

uv run python experiments/generate_phase_c_review_form.py \
  --reviewer-csv results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded_reviewer2.csv \
  --worklist-csv results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/annotation_worklist.csv \
  --reviewer-tag reviewer2 \
  --output-html results/judge_validation_phase_c_cli_v3/review_forms/reviewer2_form.html
```

Reviewer workflow for the HTML form:

- open the HTML file in a browser,
- annotate each item with radio buttons / checkboxes,
- download the completed CSV from the form,
- return that CSV as the completed reviewer sheet.

Distribution assets for independent reviewers:

- briefing: results/judge_validation_phase_c_cli_v3/review_forms/reviewer_briefing_de.md
- email template: results/judge_validation_phase_c_cli_v3/review_forms/email_template_reviewer_de.txt

QC also enforces cross-file integrity checks across reviewer and adjudication tables:

- unique `annotation_item_id` per file,
- identical item sets across reviewer and adjudication files,
- immutable metadata consistency (`framework`, `benchmark`, `run_index`, `run_id`, `raw_log_path`, `success`).

Default behavior is strict: QC fails when any reviewer sheet is incomplete.
For exploratory checks during ongoing annotation, use:

```bash
uv run python experiments/check_phase_c_annotation_quality.py --allow-incomplete
```

If you need a preliminary agreement run while annotation is still in progress, use:

```bash
uv run python experiments/run_phase_c_agreement.py --allow-partial
```

`--allow-partial` passes through to QC as `--allow-incomplete`.

Quick one-command readiness snapshot:

```bash
uv run python experiments/phase_c_readiness_status.py
```

The command returns exit code `0` only when strict final-readiness is reached.

For a preliminary (partial) report before all rows are fully resolved:

```bash
uv run python experiments/run_phase_c_agreement.py \
  --allow-partial \
  --adjudication-csv results/judge_validation_phase_c_cli_v3/annotation_adjudication_balanced_k2.csv \
  --master-csv results/judge_validation_phase_c_cli_v3/annotation_master_balanced_k2.csv \
  --output-dir results/judge_validation_phase_c_cli_v3/agreement_balanced_k2
```

## 4) Evidence standard

- Do not infer a failure mode from the task design alone.
- Only label a mode when the trace contains direct textual evidence.
- Prefer fewer, better-supported modes over broad over-labeling.

## 5) Recommended sample size

- Minimum acceptable pilot: `30` runs.
- Stronger thesis-grade target: `60+` runs with coverage across task pressures and frameworks.

## 6) CLI workflow

Generate the template:

```bash
uv run python experiments/run_judge_validation.py \
  /path/to/results_root \
  --output-dir results/judge_validation \
  --sample-size 40 \
  --real-model-only \
  --template-only
```

After manual annotation, compute agreement:

```bash
uv run python experiments/run_judge_validation.py \
  /path/to/results_root \
  --output-dir results/judge_validation \
  --real-model-only \
  --annotations-csv results/judge_validation/annotation_template.csv
```

## 7) Thesis reporting requirements

- Report task-success agreement and Cohen's kappa.
- Report exact-match rate and Jaccard overlap for primary failure-mode sets.
- Report per-mode precision, recall, and F1.
- Include a short qualitative discussion of the highest-value disagreement cases.

## 8) Current cycle assets (2026-08-01)

Current generated artifacts for the active Phase C round:

- Broad real-model pool (`n=57`): [results/judge_validation_phase_c_cli_v3/annotation_template.csv](results/judge_validation_phase_c_cli_v3/annotation_template.csv)
- Strict balanced subset (`n=24`, `k=2` per framework x benchmark cell): [results/judge_validation_phase_c_cli_v3/annotation_template_balanced_k2.csv](results/judge_validation_phase_c_cli_v3/annotation_template_balanced_k2.csv)
- Blinded reviewer sheet 1: [results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded.csv](results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded.csv)
- Blinded reviewer sheet 2 (duplicate for double-coding): [results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded_reviewer2.csv](results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded_reviewer2.csv)
- Non-blinded master (for later alignment computation only): [results/judge_validation_phase_c_cli_v3/annotation_master_balanced_k2.csv](results/judge_validation_phase_c_cli_v3/annotation_master_balanced_k2.csv)
- Reproducibility manifest (seed and cell counts): [results/judge_validation_phase_c_cli_v3/annotation_manifest_balanced_k2.json](results/judge_validation_phase_c_cli_v3/annotation_manifest_balanced_k2.json)

Recommended analysis order for strict internal validity:

1. Run primary human agreement on the strict balanced subset (`n=24`) to minimize framework/task-pressure confounding.
2. Run secondary robustness analysis on the broader pool (`n=57`) for higher power.
3. Compare whether conclusions are directionally stable across both sets.