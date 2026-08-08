# Thesis Notes

## Scope

- Compare coordination behavior across LangGraph, AutoGen (AG2), CrewAI, and MetaGPT.
- Focus on controlled reproduction, traceability, and failure taxonomy alignment.
- Keep every experiment versioned, parameterized, and traceable.

## Experimental outputs

- raw execution traces
- structured agent messages
- tool calls and tool outputs
- state transitions
- metrics: latency, token usage, cost, success rate
- MAST labels and explanations

## Planned mitigation families

- structured protocols
- supervisor-based orchestration
- cross-verification
- reflection loops
- explicit termination criteria
- role and responsibility guards

## Documentation anchors for writing

- Study protocol: [docs/study_protocol.md](docs/study_protocol.md)
- Task design rationale: [docs/task_design.md](docs/task_design.md)
- Task quality assessment: [docs/task_quality_assessment.md](docs/task_quality_assessment.md)
- Coordination codebook: [docs/coordination_suite_codebook.md](docs/coordination_suite_codebook.md)
- Phase-C data card: [docs/phase_c_data_card.md](docs/phase_c_data_card.md)
- Phase-C lineage snapshot: [docs/phase_c_lineage_2026_08_02.md](docs/phase_c_lineage_2026_08_02.md)
- Annotation and adjudication workflow: [docs/annotation_protocol.md](docs/annotation_protocol.md)
