# Phase D Analysis Plan (Design Freeze)

Date: 2026-08-27
Status: Frozen for review before any Phase D runtime execution.

## Scope

- Baseline reference: `results/coordination_baseline_2026-08-09` with reanalysis root `thesis_clean_v1`.
- Primary frameworks only: `langgraph`, `autogen`, `crewai`.
- Excluded framework: `metagpt` (DEV-002 / scaffold contamination risk).
- No new sampling from Phase C.

## Outcomes

- Primary outcome: `criteria_success`.
- Secondary outcomes:
  - `latency_seconds`
  - `token` and `cost` only when fields are present and non-null/non-empty in the analyzed rows.
- Exploratory only (separate from H4 primary inference): Judge MAST-mode outputs.

## Experimental Design Cell

- Cartesian design target:
  - 3 frameworks x 8 Coordination Suite tasks x 4 conditions x N=30 runs.
- Conditions:
  - `none`
  - `structured_output_validation`
  - `supervisor_orchestrator`
  - `reflection_independent_verification`

## Comparison Structure

- All confirmatory comparisons are within-framework.
- For each framework, compare `none` versus each of the three mitigation conditions.
- No multi-plugin combination cells.

## Statistical Plan

- Binary primary outcome (`criteria_success`):
  - Use Chi-square when assumptions are met, otherwise Fisher exact.
  - Do not use Welch t-test for binary outcomes.
- Continuous secondary outcome (`latency_seconds`):
  - Report Mann-Whitney U and Welch t-test for pairwise `none` vs strategy contrasts.
- Multiplicity control:
  - Holm correction across the 3 strategy comparisons per framework (family-wise per framework).

## Exclusions And Missingness

- Exclude scaffold runs.
- Exclude timeout runs.
- Exclude runs with empty/missing traces.
- No imputation.

## Guardrail For Design Reductions

- If any change to N, number of tasks, or number of strategies is requested, do not apply implicitly.
- Document this as "Alternative A" and wait for explicit human approval before execution.

### Alternative A (Document-Only, Not Approved)

- Description: Reduced design (N and/or tasks and/or strategies lower than frozen target).
- Status: Not approved.
- Action: Block execution until explicit human sign-off.
