# Durable project context

Project context for rag-lab, maintained by the playbook.

## Contents

- `adr/` — architecture decision records, numbered and immutable.
- `system/ingestion.md` — hazards in the extraction and load path, most of
  which fail silently.
- `system/batch-runs.md` — how to run the batch: worker and thread settings,
  resume points, corpus selection.
- `roadmap.md` — pipeline stages, what works and what is open.
- `eval-plan.md` — how retrieval is scored: the three question groups, their
  metrics, and the known weaknesses of each.
- `when-rag-fits.md` — when retrieval is the right tool for a finance question
  and when a report is better; the use cases where RAG is strongest.
