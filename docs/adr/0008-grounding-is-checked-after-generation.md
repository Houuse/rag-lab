# 0008. Grounding is checked after generation, not requested in the prompt

*2026-08-28*

## Context

The quality bar is an exact figure with a page citation, or an explicit
refusal. A confidently wrong number is worse than no number, because someone
acts on it.

A prompt can ask for citations. It cannot guarantee them, and the failure is
invisible: a fabricated `[F12345]` looks exactly like a real one, and a figure
the model computed looks exactly like a figure it copied. The reader has no way
to tell, which is precisely the property that makes a RAG system worse than a
spreadsheet.

Retrieval reaches the answer for well under half of questions, so the model is
regularly handed a context that does not contain what was asked. What it does
in that situation is the whole ballgame.

## Decision

Check mechanically, after generation, and treat the prompt as a request rather
than a control.

Every context line carries an id — `[F12345]` for a fact, `[C678]` for a chunk.
`ask.verify` then rejects three things: a cited id that was never supplied, an
answer carrying no citations at all, and any figure appearing in no retrieved
fact. The third catches the model calculating or recalling from training, which
ADR 0001 exists to prevent.

Refusal is a first-class output. `INSUFFICIENT EVIDENCE` is a success, and the
system prompt says so explicitly, because a model that treats refusal as
failure will always prefer a plausible guess.

Failures are rendered into the answer the reader sees, not written to a log.
A warning nobody reads is not a warning.

## Consequences

Measured over 56 questions end to end: 43 refusals, 3 wrong answers, 5
ungrounded. The design holds — the system is not fabricating figures — and it
is also refusing three questions in four, which is honest and not yet useful.
Those two facts are the same fact, and separating `refused` from `wrong` in the
eval is what makes it visible. Collapsing them into "incorrect" would hide the
only thing this buys.

One case is worth keeping: an AMCOR question where the model gave the right
value, 2,018, and cited a page that appeared in no supplied fact. A reader
would have accepted it — the number is correct — and never checked the
citation. That is the class of error this catches and nothing else would.

The checks are mechanical and therefore shallow. They verify that an answer
stayed inside its evidence, not that it is correct or that it answered the
question asked. One question was scored `wrong` for answering which activity
brought in the *least* cash when it was asked for the *most*; every citation
was valid. Correctness needs a judge, and a judge needs its own validation
before its scores mean anything.

Two bugs in the checker were found by writing answers designed to fail, not by
reading the code: it scanned the digits inside `[F13028]` as a claimed figure,
and a "not a year" guard skipped every value below 1900 — which exempted 1,577,
the figure most worth checking. A verifier needs adversarial tests more than
the thing it verifies does.
