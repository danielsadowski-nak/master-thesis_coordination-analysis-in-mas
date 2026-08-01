# Coordination Task Design (Custom Suite)

The tasks in `data/coordination_tasks/coordination_suite_v1.jsonl` are purpose-built to surface coordination failures in a controlled way.

Transparency note: the initial draft structure and phrasing of tasks were generated with AI assistance and then reviewed/adapted for thesis methodology.

## Language choice

Prompts are written in English to maximize compatibility across framework adapters and model defaults. This reduces language-induced variance unrelated to coordination behavior.

## Design rationale per task

1. **coord-info-asymmetry-constraint**  
   Targets hidden constraints that can be lost in handoff; expected MAST relevance: Information Withholding, Ignored Other Agent Input, Weak Verification.

2. **coord-conflicting-done-criteria**  
   Introduces incompatible completion definitions across roles; expected MAST relevance: Task Derailment, Action-Reasoning Mismatch, Premature Termination.

3. **coord-unclear-handoff-incomplete-artifact**  
   Starts with an under-specified handoff artifact; expected MAST relevance: Fail to Ask for Clarification, Information Withholding, No/Incorrect Verification.

4. **coord-weak-final-verification**  
   Pressures agents toward superficial sign-off; expected MAST relevance: Weak Verification, No/Incorrect Verification, Premature Termination.

5. **coord-role-overlap-duplicate-actions**  
   Uses overlapping ownership to induce duplicated or contradictory outputs; expected MAST relevance: Disobey Role Specification, Step Repetition, Ignored Other Agent Input.

6. **coord-ambiguous-termination-condition**  
   Adds a two-branch stop rule that is easy to misapply; expected MAST relevance: Unaware of Termination Conditions, Premature Termination.

7. **coord-combined-asymmetry-verification**  
   Combines hidden constraints with verification pressure; expected MAST relevance: Information Withholding + Weak Verification interaction.

8. **coord-clarification-before-execution**  
   Tests whether the system asks clarification under missing operational constraints; expected MAST relevance: Fail to Ask for Clarification, Task Compliance failures.

## Methodology usage in thesis

- Validation phase: run a 2-3 task subset with `N=3..5` to confirm runtime behavior and trace quality.
- Baseline phase: keep tasks fixed and increase sample size for framework comparison.
- Mitigation phase: reuse fixed tasks to compare interventions (`structured_output_validation`, `supervisor_orchestrator`, `reflection_independent_verification`).
