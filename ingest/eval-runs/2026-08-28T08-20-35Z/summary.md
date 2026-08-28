# Retrieval evaluation

Run 2026-08-28T08:20:35+00:00

Retrieval only. Two conditions: **oracle** restricts retrieval to the
filing FinanceBench names, **corpus** restricts nothing. The gap is the
cost of entity and period resolution.

## direct (n=70)

| condition | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|
| oracle | 0.086 | 0.143 | 0.200 | 0.286 | 0.121 |
| corpus | 0.029 | 0.129 | 0.129 | 0.157 | 0.063 |

## computed (n=56)

| condition | any@1 | any@5 | any@10 | any@20 | all@1 | all@5 | all@10 | all@20 | mean cov@20 |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.446 | 0.714 | 0.750 | 0.857 | 0.000 | 0.000 | 0.000 | 0.000 | 0.103 |
| corpus | 0.232 | 0.464 | 0.554 | 0.661 | 0.000 | 0.000 | 0.000 | 0.000 | 0.050 |

## narrative (n=24)

| condition | hit@20 (overlap>=0.25) | median overlap | p90 overlap |
|---|---|---|---|
| oracle | 0.500 | 0.230 | 1.000 |
| corpus | 0.417 | 0.058 | 0.767 |

## Measurement validity

**The direct group is mostly not a direct lookup.** A question lands in it
when a number in its answer also appears in the evidence — but most answers
are sentences that happen to contain one. "The consumer segment shrunk by
0.9% organically" is scored by looking for the value 0.9; one answer yields
1.5 from a bond coupon. Value-matching cannot measure those.

| direct/oracle | n | found@20 |
|---|---|---|
| answer is a sentence (>=4 words) | 62 | 15/62 = 0.242 |
| answer is a bare figure | 8 | 5/8 = 0.625 |

The second row is the real numeric-lookup rate. The headline figure is
dominated by the first.

**`all@k` for the computed group is unreachable by arithmetic, not by
retrieval quality.** `evidence_text` holds a median of 88 distinct
numbers (max 217), and 49 of 56 questions need more than the k=20
facts retrieved. A score of 0.000 is the definition biting, not a result.
Read `any@k` and mean coverage; treat `all@k` as broken until the metric
targets the numbers a computation actually needs.

## Read these numbers with the known limits

- A coincidental value match scores as a hit. 1,577 appears on several
  pages of the 3M filing, some occurrences unrelated to capex.
- Some questions have more than one defensible answer — net income 5,363
  including noncontrolling interest against 5,349 attributable to 3M — so
  a miss can be the reference being one of several valid values.
- Computed coverage understates performance: `evidence_text` contains
  numbers the computation does not need.
- The narrative threshold (0.25) is arbitrary, which is why the
  overlap distribution is reported beside the hit rate.

Per-question rows are in `rows.csv`. At n=70 for the largest group a few
points is inside the noise, and the rows are what make a change explicable.
