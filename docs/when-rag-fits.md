# When RAG fits in finance, and when a report is better

## A report wins when the answer is already structured

If the questions are known in advance and the data sits in a warehouse, a scheduled report beats retrieval on accuracy, cost, latency and auditability. "FY2018 capex for these 20 companies" is a report. For filed financial statements, SEC XBRL is the authoritative structured source and should be preferred over parsing the PDF.

The failure mode this rules out: letting a generator read a figure out of a retrieved chunk when a lookup could have returned it exactly. That produces something less reliable than SQL, presented as if it were more capable. ADR 0001 exists to prevent it.

## What this system adds

Two things a report cannot do.

It builds the structured layer that does not exist yet — the `facts` table is a small warehouse extracted from PDFs. Reporting tools consume structure, they do not create it.

It answers narrative questions, which have no tabular equivalent: what management said about a risk, how guidance was phrased, which accounting policy changed.

## Where RAG is strongest

The common trait: expensive expert time spent reading bespoke language that will never be normalised into a schema, where the answer must be traceable to specific wording because paraphrasing it wrong has consequences.

| Use case | Why it fits |
|---|---|
| Credit agreements and covenants | Individually negotiated, hundreds of pages, no two alike |
| Insurance policy wording and claims | Long, versioned, cross-referencing exclusions; high query volume |
| Regulatory and accounting standards | IFRS/GAAP, MiFID II, Basel — a corpus nobody has memorised |
| Earnings call transcripts | Pure narrative; value is comparative across quarters |
| ISDA and derivatives documentation | CSA terms, netting opinions, collateral eligibility |
| KYC onboarding and adverse media | Ownership structures, source-of-wealth documents |
| Prospectuses and fund documentation | Advisers querying hundreds of offering documents |
| Lease and contract abstraction (IFRS 16) | Unstructured input, structured output — RAG as extraction, not answering |

The last row is the same shape as this system: retrieval used to produce structure rather than to answer.

## The test

If a SQL query can answer it, that is the answer. RAG earns its place where the document is the authority.
