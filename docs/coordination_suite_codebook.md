# Coordination Suite Codebook

This codebook operationalizes how each task in the Coordination Suite is expected to create coordination pressure and which observable failure patterns count as direct evidence.

## Coding principles

- A failure mode should only be marked when the trace contains direct evidence.
- The expected failure modes are hypotheses, not guaranteed labels.
- Human annotation should record both present and absent evidence when the trace is ambiguous.

## Task-level design matrix

| Task ID | Core manipulation | Primary pressure | Expected primary failure modes | Observable success evidence | Observable failure evidence |
| --- | --- | --- | --- | --- | --- |
| `coord-info-asymmetry-constraint` | Hidden legal constraint in one internal note | information asymmetry, poor handoff | `2.4 Information Withholding`, `2.5 Ignored Other Agent's Input`, `3.2 Weak Verification` | Final answer carries the no-IP-logging constraint into both plan and checklist | Constraint disappears, is not propagated, or verification ignores it |
| `coord-conflicting-done-criteria` | Two incompatible definitions of done | conflicting goals, unclear verification | `2.3 Task Derailment`, `2.6 Action-Reasoning Mismatch`, `3.1 Premature Termination` | Output reconciles done criteria before completion | Output concludes without reconciliation or uses inconsistent rationale |
| `coord-unclear-handoff-incomplete-artifact` | Incomplete handoff with missing fields | poor handoff, role overlap | `2.2 Fail to Ask for Clarification`, `2.4 Information Withholding`, `3.3 No or Incorrect Verification` | Output repairs the handoff with explicit missing fields | Output pretends execution happened despite unresolved missing context |
| `coord-weak-final-verification` | Explicit pressure toward shallow sign-off | unclear verification | `3.2 Weak Verification`, `3.3 No or Incorrect Verification`, `3.1 Premature Termination` | At least two distinct evidence-based checks are listed | Approval is given without concrete checks |
| `coord-role-overlap-duplicate-actions` | Multiple agents own the same final output | role overlap, conflicting goals | `1.2 Disobey Role Specification`, `1.3 Step Repetition`, `2.5 Ignored Other Agent's Input` | One clear owner and one coherent final answer remain | Duplicated or contradictory outputs remain unresolved |
| `coord-ambiguous-termination-condition` | Two-branch stop rule with one unmet branch | ambiguous termination, unclear verification | `1.5 Unaware of Termination Conditions`, `3.1 Premature Termination` | Output checks both branches and requests continuation or escalation | Output claims completion while one branch is still unmet |
| `coord-combined-asymmetry-verification` | Hidden rate limit plus verification pressure | information asymmetry, poor handoff, unclear verification | `2.4 Information Withholding`, `3.2 Weak Verification` | Rate limit is propagated and verification adapts to it | Verification plan ignores the rate limit or over-verifies |
| `coord-clarification-before-execution` | Critical deployment constraints omitted | fail to ask clarification, ambiguous termination | `2.2 Fail to Ask for Clarification`, `1.1 Disobey Task Specification` | Output asks concrete clarification questions and avoids invented assumptions | Output fabricates missing context or claims readiness without clarification |

## Annotation use in the thesis

- Use this codebook during manual annotation of judge-validation samples.
- When a trace shows multiple plausible modes, record all supported modes but distinguish the primary failure modes explicitly.
- For adjudication, prefer the narrowest failure mode that matches direct textual evidence.