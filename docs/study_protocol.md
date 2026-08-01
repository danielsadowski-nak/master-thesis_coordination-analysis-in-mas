# Thesis Study Protocol

This document defines the research-grade experimental protocol for the thesis project on coordination failures in LLM-based multi-agent systems.

## 1) Research objective

The primary objective is to induce, observe, and explain coordination failures in controlled multi-agent workflows. The thesis does not treat coordination failures as incidental implementation bugs, but as measurable behavioral outcomes under defined coordination pressure.

Core question:

- Under which controlled task conditions do LLM-based multi-agent systems produce coordination failures, and which interventions reduce them reliably?

## 2) Research questions

- `RQ1`: Which coordination failure modes occur most frequently across the evaluated frameworks?
- `RQ2`: Which task pressures increase specific MAST failure modes?
- `RQ3`: Do mitigation strategies reduce targeted coordination failures without unacceptable trade-offs in latency or success rate?
- `RQ4`: How well does the automated MAST judge align with manual human annotation on a held-out trace sample?

## 3) Hypotheses

- `H1`: Tasks with information asymmetry increase `2.4 Information Withholding` relative to tasks without that pressure.
- `H2`: Tasks with verification pressure increase `3.2 Weak Verification` and `3.3 No or Incorrect Verification`.
- `H3`: Tasks with ambiguous termination conditions increase `3.1 Premature Termination` and `1.5 Unaware of Termination Conditions`.
- `H4`: Structured mitigation strategies reduce the frequency of the task-specific target failure modes relative to the no-mitigation condition.
- `H5`: The automated MAST judge reaches substantial agreement with human annotation on binary task-success labels and moderate-or-better agreement on primary failure-mode assignment.

## 4) Experimental design

### 4.1 Primary benchmark

The primary benchmark is the custom Coordination Suite in [docs/task_design.md](docs/task_design.md#L1) and [docs/coordination_suite_codebook.md](docs/coordination_suite_codebook.md#L1).

Rationale:

- It allows explicit manipulation of coordination pressure.
- It is small enough for repeated controlled trials.
- It aligns directly with the MAST taxonomy.

### 4.2 Secondary benchmark

SWE-Bench Verified and GAIA are secondary external-validity benchmarks. They should only be used after the controlled Coordination Suite baseline and mitigation analyses are stable.

### 4.3 Independent variables

- Framework: LangGraph, AutoGen, CrewAI, MetaGPT.
- Task: fixed Coordination Suite task IDs.
- Coordination pressure: encoded by task design and documented in the codebook.
- Mitigation condition: none, structured output validation, supervisor orchestrator, reflection independent verification.

### 4.4 Dependent variables

- Task success.
- Primary MAST failure modes.
- MAST category distribution.
- Latency.
- Token usage.
- Optional cost estimate if pricing metadata is available or added later.

### 4.5 Controls

- Same model family and temperature across frameworks where possible.
- Same task texts, task order, and benchmark source files.
- Same run count `N` per task-framework-condition cell.
- Same stopping constraints (`max_steps`) across conditions.

### 4.6 Freeze point and reproducibility lock

Before Phase B begins, the following artifacts are frozen and versioned:

- the Coordination Suite source file
- the Coordination Suite codebook
- this study protocol
- framework versions and dependency state
- model name, exposed provider-side model version string when available, temperature, max-steps setting, and seed strategy

The frozen configuration is treated as the internal-validity reference point for the baseline and mitigation phases. Any later change to task wording, codebook logic, framework version, or judge setup must be documented as a protocol deviation.

### 4.7 Mitigation operationalization

The mitigation conditions are not abstract labels; each condition corresponds to an explicit intervention in the coordination workflow.

- `structured_output_validation`: the agent is instructed to produce a structured result with explicit verification fields so that missing constraints, missing evidence, and weak handoffs become easier to detect before termination.
- `supervisor_orchestrator`: an explicit orchestration layer assigns responsibilities, sequences handoffs, and enforces a single coordination authority to reduce role overlap and contradictory outputs.
- `reflection_independent_verification`: the workflow adds a deliberate self-check and an independent verification step before final completion, intended to reduce shallow verification and premature termination.

Each mitigation must be kept implementation-stable within a study phase. If prompts or orchestration logic are changed, the condition must be re-frozen and the deviation reported.

## 5) Study phases

### Phase A: Validation

- Purpose: verify runtime, trace quality, and judge output shape.
- Recommended size: `2-3` tasks with `N=3..5`.

### Phase B: Baseline comparison

- Purpose: estimate framework-specific failure behavior without mitigations.
- Recommended size: all Coordination Suite tasks with `N=30..50` per framework-task cell.
- Minimum acceptable size under compute constraints: `N=30`, with explicit justification in the thesis.

Sampling rationale:

- The target range `N=30..50` is chosen as a pragmatic repetition range that supports stable descriptive estimates, bootstrap confidence intervals, and non-trivial pairwise comparisons without making the study computationally infeasible.
- If an a-priori power analysis is feasible once pilot variances are available, it should be added before the final baseline run.
- If a formal power analysis is not feasible because of provider cost or unstable variance estimates, the thesis must explicitly document this and report a sensitivity-oriented justification using pilot data, confidence intervals, and observed effect sizes.

### Phase C: Judge validation

- Purpose: validate the automated MAST judge against manual annotation.
- Recommendation: annotate a stratified sample of at least `30-60` runs across frameworks and task pressures.
- Use adjudicated labels as the ground-truth target where disagreements are resolved.

Annotation process:

- Human annotators use the Coordination Suite codebook as the primary coding reference.
- At least one primary annotator reviews all sampled traces. Where feasible, a second annotator reviews a subset or all traces before adjudication.
- Inter-annotator agreement should be reported before adjudication when more than one annotator is used.
- Disagreements are resolved in an adjudication pass, and adjudicated labels are treated as the final reference labels for judge evaluation.

### Phase D: Mitigation comparison

- Purpose: test whether interventions reduce targeted failure modes.
- Use the same fixed task set as the baseline.
- Compare each mitigation against the `none` condition within each framework.

### Phase E: Optional external validation

- Purpose: test whether patterns observed in the Coordination Suite transfer to more realistic benchmark tasks.
- Use SWE-Bench Verified and GAIA only after internal validity is established.

## 6) Primary analysis plan

Primary outcomes:

- Success rate by framework and condition.
- Frequency of primary MAST failure modes by framework and task pressure.
- Frequency of MAST categories by framework and condition.

Secondary outcomes:

- Latency.
- Token usage.
- Cost, where available.

Recommended reporting:

- Descriptive statistics with bootstrap confidence intervals.
- Pairwise framework and condition comparisons.
- Effect sizes, not only p-values.
- Clear distinction between internal validity results and external validation results.

Statistical decision rules:

- Success-rate and binary outcome comparisons should use Chi-square where assumptions are met and Fisher's exact test when expected counts are small.
- Continuous outcomes such as latency and token usage should use Welch's t-test when group assumptions are sufficiently plausible and Mann-Whitney U otherwise.
- Distributional comparisons of MAST categories should use Chi-square-style contingency analysis.
- Bootstrap confidence intervals should be reported as the default descriptive uncertainty estimate.
- If multiple pairwise tests are interpreted jointly, the thesis should disclose the correction strategy used for multiple comparisons, or explicitly label the analysis as exploratory where correction is not applied.

Qualitative complement:

- In addition to aggregate statistics, the thesis should include a short qualitative trace analysis of high-value disagreement cases, representative coordination failures, and mitigation successes.
- These qualitative examples are explanatory supplements and do not replace the quantitative comparisons.

Task success operationalization:

- Task success is determined from the task-specific success criteria defined in the Coordination Suite and interpreted through the codebook.
- This ensures that success is tied to explicit task completion requirements rather than generic plausibility of the final answer.

## 7) Judge validation criteria

Minimum reporting requirements:

- Raw agreement for task-success labels.
- Cohen's kappa for task-success labels.
- Exact-match rate for primary failure-mode sets.
- Mean Jaccard overlap between human and judge failure-mode sets.
- Per-mode precision, recall, and F1.
- A disagreement table with trace references for qualitative inspection.

## 8) Threats to validity

- Model drift at the provider level.
- Framework implementations are not equally mature.
- The custom task suite maximizes internal control but may reduce external realism.
- LLM-as-Judge outputs may systematically favor some failure modes.
- Token cost may be incomplete if provider metadata does not expose price information.

Mitigation for model drift:

- Record the provider-returned model identifier whenever available.
- Keep the main baseline and mitigation runs temporally compact where possible.
- Treat substantial provider-side model updates during the study window as a validity threat and report them explicitly.

## 9) Run validity and failure handling

The protocol distinguishes between analytical outcomes and infrastructure failures.

- A valid run is one that produces a trace, a final output or explicit failure output, and a parseable run record.
- A run with task failure is still analytically valid if the trace and judgement artifacts are present.
- Timeouts, crashes, empty traces, or non-parseable outputs are recorded as runtime failures and must not be silently discarded.
- Runtime failures should be reported separately from task-success failures.
- If a MAST judgement is missing or malformed, the run remains part of the runtime accounting but must be marked as missing-judge data in the analysis.
- No imputation is performed for missing traces, missing judgements, or missing success labels in the main analysis.
- Any exclusion from the primary analysis must be rule-based, logged, and reported in the thesis.

## 10) Transparency and AI assistance

- Development of the software artifacts, experiment scripts, and parts of the documentation may be supported by GitHub Copilot or comparable generative AI tools.
- Such assistance must be disclosed transparently in the thesis.
- All final methodological decisions, code review, and interpretation remain the responsibility of the author.

## 11) Execution order

1. Validate runtime and trace quality.
2. Freeze the Coordination Suite and codebook.
3. Run the Coordination Suite baseline.
4. Generate and annotate the judge-validation sample.
5. Evaluate judge-human agreement.
6. Run mitigation experiments on the same fixed tasks.
7. Optionally extend to SWE-Bench Verified and GAIA.