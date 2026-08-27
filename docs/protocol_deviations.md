# Protocol Deviations Log

This log records deviations from `docs/study_protocol.md` and freeze memos.

## DEV-001 — Phase C success-only sampling (critical)

- **Detected:** 2026-08-24
- **Description:** `filter_annotation_candidates(..., real_model_only=True)` additionally restricted rows to `success=true`, producing an all-success Phase C sample (`n=24`).
- **Impact:** Binary task-success agreement (H5) becomes degenerate; failure-mode validation is biased toward successful traces.
- **Correction:** Sampling code no longer drops failures under `real_model_only`. Rebuild sample with `experiments/build_phase_c_sample_v4.py`.
- **Thesis handling:** Do not report H5 from the old `balanced_k2` all-success sheet.

## DEV-002 — MetaGPT scaffold contamination in baseline 2026-08-09

- **Detected:** 2026-08-24
- **Description:** MetaGPT rows show ~0.003s latency and 100% success, consistent with adapter scaffold fallback rather than native MetaGPT execution.
- **Impact:** RQ1 framework comparisons including MetaGPT are not analytically valid.
- **Correction:** Exclude MetaGPT from primary tables until native runs exist; reanalysis script defaults to exclusion.
- **Thesis handling:** Report as implementation limitation or replace with native Docker runs.

## DEV-003 — Mitigation phase underpowered

- **Detected:** 2026-08-24
- **Description:** Existing mitigation artifact is effectively N=1, one framework, one task, one strategy.
- **Impact:** H4 cannot be tested from current mitigation artifacts.
- **Correction:** Re-run Phase D after clean baseline freeze.
- **Thesis handling:** Treat current mitigation folder as smoke test only.

## DEV-004 — Pairwise success tests used Welch t on binary outcomes

- **Detected:** 2026-08-24
- **Description:** Stored pairwise success tables used Welch t-tests, while the protocol specifies Chi-square/Fisher.
- **Impact:** Inferential claims for success rates are not protocol-conformant.
- **Correction:** Use `compare_success_rates` plus Holm adjustment via `experiments/reanalyze_baseline.py`.
- **Thesis handling:** Prefer the reanalysis tables in the results chapter.

## DEV-005 — Multi-root Phase C provenance

- **Detected:** 2026-08-02 / reconfirmed 2026-08-24
- **Description:** Earlier Phase C rows mixed validation and repair experiment roots, partly confounded with framework.
- **Impact:** Reduced internal homogeneity of the validation sample.
- **Correction:** Rebuild Phase C from a single baseline root (`coordination_baseline_2026-08-09`) when possible.
- **Thesis handling:** Disclose lineage if any legacy sample is referenced.

## DEV-006 — Native MetaGPT baseline execution status

- **Detected:** [YYYY-MM-DD]
- **Description:** Native MetaGPT baseline [executed successfully / failed] under dockerized runtime.
- **Results root:** [results/coordination_baseline_YYYY-MM-DD or other path]
- **Sample size:** planned [N planned], valid [N valid]
- **Primary RQ1 inclusion:** [included / excluded]
- **Residual scaffold rows:** [count or unknown]
- **Exclusion handling:** [e.g., filtered by runtime_mode=scaffold and is_valid_analytical=false]

## DEV-007 — Phase C v4 stratification switch and timeout exclusion

- **Detected:** 2026-08-24
- **Description:** Reviewer-shipping sample moved to v4 with stratification by `criteria_success` instead of runner `success`; runtime timeout rows (`timed out before completion`) were excluded from reviewer sheets.
- **Impact:** Better alignment with protocol intent (`Runtime-Failures != Task-Failures`) and reduced risk of annotating infrastructure failures as task behavior.
- **Correction:** Rebuilt `results/judge_validation_phase_c_v4` from `coordination_baseline_2026-08-09` only, excluding MetaGPT and timeout rows.
- **Thesis handling:** Report the exact v4 manifest fields (`stratification_success_column`, `excluded_runtime_failures`, `unbalanced_cells`) in Phase C methods and limitations.

## DEV-008 — Phase D mitigations are prompt-level operationalizations

- **Detected:** 2026-08-27
- **Description:** Registered Phase D mitigation strategies (`structured_output_validation`, `supervisor_orchestrator`, `reflection_independent_verification`) are currently implemented as prompt augmentation plugins (`augment_system_prompt`), not architecture-level agent rewrites.
- **Impact:** H4 interpretation is bounded to this operationalization level; results must not be framed as architectural intervention effects.
- **Correction:** Freeze Phase D claims to prompt-level mitigation efficacy only; keep registry names and runner wiring stable for confirmatory runs.
- **Thesis handling:** State explicitly in methods and limitations that H4 is tested only under prompt-level operationalization in the current implementation.
