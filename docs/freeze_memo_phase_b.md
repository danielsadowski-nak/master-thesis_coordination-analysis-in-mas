# Freeze Memo For Phase B

This memo records the frozen internal-validity reference state before the thesis baseline phase on the Coordination Suite.

## Scope of freeze

The following artifacts are frozen for Phase B:

- Coordination Suite source: [data/coordination_tasks/coordination_suite_v1.jsonl](data/coordination_tasks/coordination_suite_v1.jsonl)
- Task design document: [docs/task_design.md](docs/task_design.md)
- Coordination Suite codebook: [docs/coordination_suite_codebook.md](docs/coordination_suite_codebook.md)
- Study protocol: [docs/study_protocol.md](docs/study_protocol.md)

## Artifact hashes

- `study_protocol.md`: `2bd39f8553f8ebd9566691d7458569d585630dc02ff64b74c2575513abf8656f`
- `coordination_suite_codebook.md`: `6b2a03436d511dea09b77ab8e31979a903c15e9a89674a3e48af3c6170c7aaca`
- `task_design.md`: `f9265bcdaeb8ea97bf3ed152702a12cdd8ac36e052c77fdea0668c749b3d5457`
- `coordination_suite_v1.jsonl`: `c046bc5a7d8a1f1ba93ae641d8374dffd5bded911981343d6fe216c8a571991f`

## Runtime baseline

- Python: `3.12.13`
- `langgraph`: `0.6.11`
- `langchain-openai`: `0.3.35`
- `openai`: `2.52.0`
- `pandas`: `2.3.3`
- `numpy`: `2.5.1`
- `scipy`: `1.18.0`
- `statsmodels`: `0.14.6`
- `matplotlib`: `3.11.1`
- `seaborn`: `0.13.2`

## Experimental defaults frozen for the main internal-validity study

- Primary benchmark: Coordination Suite only
- Baseline runner: [experiments/run_coordination_baseline.py](experiments/run_coordination_baseline.py)
- Mitigation runner: [experiments/run_coordination_mitigation.py](experiments/run_coordination_mitigation.py)
- Judge validation runner: [experiments/run_judge_validation.py](experiments/run_judge_validation.py)
- Model family for the current plan: `gpt-4o-mini`
- Temperature: `0.0`
- Seed strategy: fixed deterministic seed (`42`) unless a documented deviation is introduced
- Maximum steps: `25`
- Main baseline target: `N=30..50` runs per framework-task cell

## Repository state

- Git metadata is not currently available inside this workspace snapshot (`NO_GIT_REPO`).
- Because no `.git` metadata is present, artifact hashes and runtime versions act as the primary freeze identifiers.

## Judge-validation readiness

- A Phase C annotation template has already been generated at [results/judge_validation_phase_c/annotation_template.csv](results/judge_validation_phase_c/annotation_template.csv).
- Current available annotation sample size from existing runs: `23` rows.
- A filtered real-model annotation pool has been written to [results/judge_validation_phase_c/annotation_template_real_model_runs.csv](results/judge_validation_phase_c/annotation_template_real_model_runs.csv).
- Current real-model annotation sample size: `19` rows.
- An updated multi-framework annotation snapshot has been written to [results/judge_validation_phase_c/annotation_template_after_multiframe.csv](results/judge_validation_phase_c/annotation_template_after_multiframe.csv).
- Current updated annotation sample size: `30` rows total, `25` real-model rows.
- A refreshed real-model-only Phase C template has been written to [results/judge_validation_phase_c_cli_v2/annotation_template.csv](results/judge_validation_phase_c_cli_v2/annotation_template.csv).
- Current refreshed sample size: `49` rows (`success=True` only; scaffold/fallback excluded).
- Framework distribution in the refreshed template: `langgraph=21`, `autogen=20`, `crewai=8`.
- Benchmark distribution in the refreshed template: `coord-conflicting-done-criteria=16`, `coord-info-asymmetry-constraint=15`, `coord-unclear-handoff-incomplete-artifact=12`, `coord-weak-final-verification=6`.
- A further updated real-model-only Phase C template has been written to [results/judge_validation_phase_c_cli_v3/annotation_template.csv](results/judge_validation_phase_c_cli_v3/annotation_template.csv).
- Current updated sample size: `57` rows (`success=True` only; scaffold/fallback excluded).
- Framework distribution in the updated template: `langgraph=21`, `autogen=20`, `crewai=16`.
- Benchmark distribution in the updated template: `coord-conflicting-done-criteria=18`, `coord-info-asymmetry-constraint=17`, `coord-unclear-handoff-incomplete-artifact=14`, `coord-weak-final-verification=8`.
- A strict balanced annotation subset (equal rows per framework x benchmark cell) has been written to [results/judge_validation_phase_c_cli_v3/annotation_template_balanced_k2.csv](results/judge_validation_phase_c_cli_v3/annotation_template_balanced_k2.csv).
- Balanced subset size: `24` rows (`k=2` per cell across `3` frameworks x `4` task pressures).
- This provides two defensible Phase C options: higher-power (`n=57`) and strictly balanced (`n=24`).

## Required action before full Phase C evaluation

- Optional quality upgrade: add targeted `crewai` and weak-verification runs to improve framework/task balance before final annotation assignment.

## Deviation rule

Any change to task wording, codebook logic, protocol logic, framework implementation, model family, temperature, or study-phase runner configuration after this memo must be documented as a protocol deviation in the thesis and in the corresponding result directory metadata.