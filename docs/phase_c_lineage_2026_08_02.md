# Phase C Lineage Snapshot (2026-08-02)

## Scope

This snapshot documents the concrete provenance of the current strict balanced Phase C subset (`n=24`).

Reference file:

- [results/judge_validation_phase_c_cli_v3/annotation_template_balanced_k2.csv](results/judge_validation_phase_c_cli_v3/annotation_template_balanced_k2.csv)

## Fixed design properties

- rows: 24
- frameworks: 3 (`autogen`, `crewai`, `langgraph`)
- benchmark task types: 4
- per framework x task cell: 2 rows
- all `success=true`

Manifest:

- [results/judge_validation_phase_c_cli_v3/annotation_manifest_balanced_k2.json](results/judge_validation_phase_c_cli_v3/annotation_manifest_balanced_k2.json)

## Marginal distributions

Framework counts:

- autogen: 8
- crewai: 8
- langgraph: 8

Task-type counts:

- coordination_suite/coord-conflicting-done-criteria: 6
- coordination_suite/coord-info-asymmetry-constraint: 6
- coordination_suite/coord-unclear-handoff-incomplete-artifact: 6
- coordination_suite/coord-weak-final-verification: 6

## Source-root provenance

Rows grouped by trace-root origin (`raw_log_path` pattern):

- coordination_baseline_2026-08-01: 8
- phase_b_autogen_repair_v5: 5
- phase_b_crewai_balance_v1: 4
- phase_b_autogen_repair: 3
- validation_20260801_175116: 3
- validation_20260801_093923: 1

Framework x source-root table:

| framework | coordination_baseline_2026-08-01 | phase_b_autogen_repair | phase_b_autogen_repair_v5 | phase_b_crewai_balance_v1 | validation_20260801_093923 | validation_20260801_175116 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| autogen | 0 | 3 | 5 | 0 | 0 | 0 |
| crewai | 4 | 0 | 0 | 4 | 0 | 0 |
| langgraph | 4 | 0 | 0 | 0 | 1 | 3 |

## Interpretation

- The subset is balanced at the framework x task level by design.
- Provenance roots are mixed; this improves sample availability and balance but introduces potential temporal/runtime heterogeneity.
- Since trace-root membership is confounded with framework in parts of this snapshot, lineage should be transparently disclosed in the thesis.

## Recommendation for thesis reporting

Report both facts together:

- controlled balancing (strong internal comparability), and
- multi-root provenance (possible heterogeneity/drift risk).

This avoids overstating homogeneity while preserving the validity of the controlled comparison intent.
