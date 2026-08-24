# Thesis Readiness Plan

This document turns the research-design review into an executable remediation plan.
Status date: 2026-08-24.

## Non-negotiable standards for this thesis

1. Primary analyses use only analytically valid runs (no scaffold/fallback).
2. Task success is defined independently of the MAST judge where possible.
3. Phase C includes successes **and** failures.
4. Statistical tests match the protocol (Chi-square/Fisher for binary outcomes; Holm adjustment for families of pairwise tests).
5. MetaGPT enters the main comparison only after native (non-scaffold) evidence exists.
6. Mitigation results are reported only after N and coverage are comparable to baseline intent.
7. Every protocol deviation is logged.

## Priority workstream

### P0 — must complete before result claims

| Item | Status | Action |
| --- | --- | --- |
| Scaffold contamination | tooling added | Run `experiments/reanalyze_baseline.py` on `coordination_baseline_2026-08-09` |
| MetaGPT validity | open | Either native Docker re-run or exclude from primary RQ1 tables |
| Phase C sample bias (`success=true` only) | fixed in code | Rebuild with `experiments/build_phase_c_sample_v4.py` |
| Double-coding | open | Human reviewers must complete blinded sheets |
| Independent success operationalization | tooling added | Use `criteria_success` as secondary outcome; human adjudication remains reference |

### P1 — required for full Phase D claims

| Item | Status | Action |
| --- | --- | --- |
| Mitigation N=1 smoke | open | Re-run all mitigations on fixed task set with N>=30 after clean baseline |
| Multiple-comparison policy | tooling added | Use Holm-adjusted tables from reanalysis |
| Provider model snapshot IDs | open | Log returned model identifiers during future runs |

### P2 — strengthens external validity

| Item | Status | Action |
| --- | --- | --- |
| SWE-Bench / GAIA | optional | Only after internal validity is frozen |
| Multi-seed robustness | optional | 2–3 seeds if compute allows |

## Commands

### Clean baseline reanalysis

```bash
uv run python experiments/reanalyze_baseline.py \
  results/coordination_baseline_2026-08-09/experiments \
  --output-dir results/coordination_baseline_2026-08-09/reports/thesis_clean_v1 \
  --exclude-frameworks metagpt
```

### Rebuild Phase C sample

```bash
uv run python experiments/build_phase_c_sample_v4.py \
  results/coordination_baseline_2026-08-09/experiments \
  --output-dir results/judge_validation_phase_c_v4 \
  --sample-size 60 \
  --exclude-frameworks metagpt
```

### After both reviewers finish

```bash
uv run python experiments/check_phase_c_annotation_quality.py \
  --reviewer1-csv results/judge_validation_phase_c_v4/annotation_sheet_blinded_reviewer1.csv \
  --reviewer2-csv results/judge_validation_phase_c_v4/annotation_sheet_blinded_reviewer2.csv \
  --output-dir results/judge_validation_phase_c_v4/agreement

uv run python experiments/prepare_phase_c_adjudication.py \
  results/judge_validation_phase_c_v4/annotation_sheet_blinded_reviewer1.csv \
  results/judge_validation_phase_c_v4/annotation_sheet_blinded_reviewer2.csv \
  --output-csv results/judge_validation_phase_c_v4/annotation_adjudication.csv

uv run python experiments/run_phase_c_agreement.py \
  --adjudication-csv results/judge_validation_phase_c_v4/annotation_adjudication.csv \
  --master-csv results/judge_validation_phase_c_v4/annotation_master.csv \
  --output-dir results/judge_validation_phase_c_v4/agreement
```

## Recommended thesis framing after remediation

- Primary framework comparison: LangGraph, AutoGen, CrewAI on analytically valid runs.
- MetaGPT: implementation note / limitation unless native data are collected.
- Phase C: measurement validation under stratified success/failure sampling.
- Phase D: confirmatory only after clean baseline + validated judge.

## What remains human work

- Independent annotation and adjudication.
- Final interpretation of disagreement cases.
- Decision on MetaGPT inclusion vs. exclusion.
- Optional native re-runs under Docker if MetaGPT is kept in scope.
