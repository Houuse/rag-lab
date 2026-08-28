# Answer evaluation

Run 2026-08-28T13:45:56+00:00
model qwen2.5:7b-instruct, 10 facts and 3 passages per question

End to end: route, retrieve, generate, verify — scored against
FinanceBench's answers. `eval.py` measures retrieval; this measures what
actually comes out.

## direct (n=9)

| outcome | n | share |
|---|---|---|
| correct | 1 | 11% |
| ungrounded | 2 | 22% |
| refused | 6 | 67% |

## computed (n=3)

| outcome | n | share |
|---|---|---|
| ungrounded | 2 | 67% |
| refused | 1 | 33% |

Computed answers are ratios and growth rates whose values appear in no
filing. Answering them needs arithmetic, and the model is forbidden to
calculate (ADR 0001). Refusal is therefore the *correct* behaviour here
until an arithmetic step exists in code — and a `correct` in this group
means the model broke the rule and happened to be right.

## narrative (n=5)

| outcome | n | share |
|---|---|---|
| refused | 3 | 60% |
| unscored | 1 | 20% |
| error | 1 | 20% |

`unscored` is honest, not a gap in the harness: grading a prose answer
needs a judge, and a judge needs validating before its scores mean
anything. The trigram overlap in `note` is inspectable, not a grade.

## The number that matters

On the 9 direct questions: **1 correct, 2 wrong,
6 refused**.

Wrong and refused are not interchangeable. A refusal means retrieval failed
and the system said so, which is the designed behaviour and costs a user
nothing but time. A wrong figure with a citation is the failure this whole
design exists to prevent, because someone acts on it.

Median 51s per question, 0.4 hours total.

