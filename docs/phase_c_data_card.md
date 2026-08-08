# Phase C Data Card (Judge-Validation Sample)

## Purpose

This document describes the provenance, construction logic, quality gates, and limitations of the Phase C annotation dataset used for human-vs-judge validation.

Primary objective:

- create a manually labeled reference set for evaluating agreement with automated MAST judgments.

## Dataset variants

Current variants in this repository:

- broad real-model pool (`n=57`): [results/judge_validation_phase_c_cli_v3/annotation_template.csv](results/judge_validation_phase_c_cli_v3/annotation_template.csv)
- strict balanced subset (`n=24`): [results/judge_validation_phase_c_cli_v3/annotation_template_balanced_k2.csv](results/judge_validation_phase_c_cli_v3/annotation_template_balanced_k2.csv)

This data card focuses on the strict balanced subset (`n=24`) because it is the primary internal-validity sample for double-coding.

## Unit of analysis

One row equals one completed run with:

- framework
- benchmark/task id
- run index and run id
- raw trace path
- final output
- automated judge fields
- empty manual/adjudication fields (to be completed later)

## Source lineage

The balanced subset was created from existing run artifacts generated in earlier phases. The source mapping is encoded via `raw_log_path`.

Observed source roots for the current `n=24` subset:

- `coordination_baseline_2026-08-01`: 8 rows
- `phase_b_autogen_repair_v5`: 5 rows
- `phase_b_crewai_balance_v1`: 4 rows
- `phase_b_autogen_repair`: 3 rows
- `validation_20260801_175116`: 3 rows
- `validation_20260801_093923`: 1 row

## Sampling and balancing logic

Construction metadata is recorded in:

- [results/judge_validation_phase_c_cli_v3/annotation_manifest_balanced_k2.json](results/judge_validation_phase_c_cli_v3/annotation_manifest_balanced_k2.json)

Design parameters:

- seed: `42`
- subset size: `24`
- all rows must satisfy `success=true`
- exact cell balancing: `k=2` per framework x benchmark cell

Resulting marginals:

- framework: autogen `8`, crewai `8`, langgraph `8`
- benchmark tasks: each of 4 coordination tasks contributes `6`

Interpretation:

- this is a controlled, stratified validation sample optimized for comparability, not a natural-frequency sample.

## Inclusion and exclusion rules

Implemented inclusion intent:

- real-model runs only (scaffold/fallback excluded)
- successful runs only (`success=true`)
- balanced coverage across framework x task-pressure cells

Operational checks are enforced through the Phase C QC pipeline.

## Quality-control gates

QC script:

- [experiments/check_phase_c_annotation_quality.py](experiments/check_phase_c_annotation_quality.py)

QC outputs:

- [results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/annotation_qc_summary.json](results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/annotation_qc_summary.json)
- [results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/annotation_qc_issues.csv](results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/annotation_qc_issues.csv)
- [results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/annotation_missing_manifest.csv](results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/annotation_missing_manifest.csv)
- [results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/annotation_progress_matrix.csv](results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/annotation_progress_matrix.csv)
- [results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/annotation_worklist.csv](results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/annotation_worklist.csv)
- [results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/adjudication_backlog_manifest.csv](results/judge_validation_phase_c_cli_v3/agreement_balanced_k2/adjudication_backlog_manifest.csv)

Key guarantees:

- required columns present
- boolean format validity
- MAST label whitelist validation
- cross-file integrity (unique IDs, equal item sets, immutable metadata consistency)

## Current readiness status

At the time of writing:

- reviewer completion is pending (manual fields not yet filled)
- strict QC remains failing due to incompleteness
- structural integrity checks are passing

This means:

- data collection infrastructure is ready
- inferential agreement reporting is not yet ready until double-coding and adjudication are completed.

## Intended claims and non-claims

Defensible claims after completion:

- agreement quality between automated judge and adjudicated human labels under a controlled balanced sample
- disagreement patterns by mode/category in this constrained setup

Non-claims:

- population-level prevalence estimates for all possible real-world MAS tasks
- external generalization beyond the Coordination Suite without additional benchmarks

## Known limitations

- custom benchmark design may favor specific coordination pressures
- sample size `n=24` is validation-oriented and limited for broad subgroup inference
- source runs originate from multiple experiment roots; balancing controls comparability but not full temporal/model drift
- automated judge fields currently include heuristic-fallback traces in parts of the pool; this should be disclosed in final reporting

## Recommended thesis wording anchor

Use this sample as:

- an internally controlled validation set for measurement-alignment analysis,
- complemented by broader robustness checks on the `n=57` pool and external benchmarks for transfer claims.
