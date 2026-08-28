"""Answer a question from the filings: route, retrieve, assemble, generate, verify.

    python ask.py "What was 3M's FY2018 capital expenditure?"
    python ask.py "Why did operating cash flow fall in 2018 for 3M?"
    python ask.py "..." --show-context     # what the model was actually given
    python ask.py "..." --model qwen2.5:3b-instruct

Needs a local model served by Ollama:

    ollama serve &
    ollama pull qwen2.5:7b-instruct

The quality bar is an exact figure with a page citation, or an explicit
refusal. A confidently wrong number is worse than no number, because someone
acts on it. Three things enforce that here, and none of them is the prompt:

1. Every figure the model may use is a `facts` row — a value read out of a
   table cell at ingest, with its page. The model selects; it never computes
   and never recalls.
2. Every context line carries an id. The answer must cite them, and `verify()`
   checks each cited id was actually supplied.
3. Any number in the answer that is not in the supplied facts is flagged.
   A model that does arithmetic anyway is caught rather than believed.

None of this makes the model honest. It makes dishonesty visible.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from search import resolve_scope, search, search_facts

OLLAMA = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

REFUSAL = "INSUFFICIENT EVIDENCE"

SYSTEM = f"""You answer questions about SEC filings using ONLY the CONTEXT provided.

Rules, in order of importance:

1. If the CONTEXT does not contain what the question asks for, reply with
   exactly: {REFUSAL}
   followed by one sentence saying what is missing. Refusing is correct and
   expected. A wrong figure is far worse than no figure.
2. Every number in your answer must be copied from a FACT line. Do not
   calculate, convert, sum, or adjust anything. Do not use knowledge from
   outside the CONTEXT, even if you are confident.
3. Cite the id in square brackets after every figure and every claim, like
   [F12345] or [C678]. A sentence with a figure and no citation is invalid.
4. Check the period. A FACT from the wrong year does not answer a question
   about this year — if only the wrong year is present, refuse.
5. Be brief. State the figure, its units and its page. No preamble.
"""

# Questions that want a figure. Deliberately a rule and not the model: routing
# with the model would mean a second call and a second thing to debug, and
# getting this wrong sends a numeric question down the prose path, which is the
# failure ADR 0001 exists to prevent.
NUMERIC_RE = re.compile(
    r"\b(what is|what was|how much|how many|amount|total|value of|"
    r"capex|capital expenditure|revenue|net sales|net income|margin|ratio|"
    r"assets|liabilities|cash flow|ebitda|eps|dividend|inventory|"
    r"\$|usd|millions?|billions?)\b",
    re.I,
)


@dataclass
class Context:
    """What the model is shown, and the only thing it is allowed to use."""

    facts: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    company: str | None = None
    fiscal_year: int | None = None
    kind: str = "numeric"

    @property
    def ids(self) -> set[str]:
        return {f"F{f[6]}" for f in self.facts} | {f"C{c[0]}" for c in self.chunks}

    @property
    def values(self) -> set[float]:
        return {abs(float(f[2])) for f in self.facts}

    @property
    def pages(self) -> set[float]:
        """Page numbers are grounded too — the answer is asked to cite them."""
        return {float(f[4]) for f in self.facts} | {
            float(c[3]) for c in self.chunks if c[3] is not None
        }


def route(question: str) -> tuple[str, str | None, int | None]:
    """What kind of answer is wanted, and which filings can hold it."""
    company, year = resolve_scope(question)
    kind = "numeric" if NUMERIC_RE.search(question) else "narrative"
    return kind, company, year


def retrieve(question: str, kind: str, company, year, k_facts: int, k_chunks: int) -> Context:
    """Facts for figures, chunks for prose — and some of the other either way.

    A numeric question still gets prose, because the sentence around a number
    is what makes it checkable. A narrative question still gets facts, because
    "why did cash flow fall" is usually answered with figures in it.
    """
    ctx = Context(company=company, fiscal_year=year, kind=kind)
    ctx.facts = search_facts(question, k=k_facts, company=company, fiscal_year=year)
    ctx.chunks = search(question, k=k_chunks, company=company, fiscal_year=year)

    # A filter that matches nothing is worse than no filter: the answer becomes
    # unreachable. Fall back rather than confidently return an empty context.
    if not ctx.facts and not ctx.chunks and (company or year):
        ctx.facts = search_facts(question, k=k_facts, company=company)
        ctx.chunks = search(question, k=k_chunks, company=company)
        ctx.fiscal_year = None
    return ctx


def render(question: str, ctx: Context) -> str:
    lines = ["CONTEXT", ""]
    if ctx.facts:
        lines.append("Figures — each is a value read from a table cell in the filing:")
        for score, ftext, signed, scale, page, doc, fid in ctx.facts:
            unit = f" {scale}" if scale else ""
            lines.append(f"  [F{fid}] {doc} p{page} | {ftext} | value={signed}{unit}")
        lines.append("")
    if ctx.chunks:
        lines.append("Passages:")
        for cid, score, kind, page, item, context, doc, text in ctx.chunks:
            head = f"  [C{cid}] {doc} p{page}"
            if context:
                head += f" | {context[:70]}"
            lines.append(head)
            body = " ".join(text.split())[:700]
            lines.append(f"      {body}")
        lines.append("")
    lines += ["QUESTION", question]
    return "\n".join(lines)


def generate(prompt: str, model: str, host: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        # Deterministic: an answer that changes between runs cannot be
        # regression-tested, and this is a regression suite before it is a
        # product.
        "options": {"temperature": 0, "seed": 1},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["message"]["content"].strip()
    except urllib.error.URLError as e:
        sys.exit(
            f"cannot reach Ollama at {host}: {e}\n"
            "  ollama serve &\n"
            f"  ollama pull {model}"
        )


CITE_RE = re.compile(r"\[([FC]\d+)\]")
# Not preceded or followed by a letter: the "3" in "3M" and the "10" in
# "10-K" are parts of names, not figures the model could have invented.
ANSWER_NUM_RE = re.compile(r"(?<![A-Za-z0-9])-?\$?\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])")


def verify(answer: str, ctx: Context) -> list[str]:
    """Everything wrong with this answer that can be checked mechanically.

    Not a judge of correctness — a check that the answer stayed inside its
    evidence. These are the failures that make a RAG system worse than a
    lookup: a citation that points nowhere, and a number that came from the
    model rather than the filing.
    """
    problems = []
    if answer.startswith(REFUSAL):
        return problems

    cited = set(CITE_RE.findall(answer))
    if not cited:
        problems.append("no citations: every figure and claim must cite an id")
    for c in sorted(cited - ctx.ids):
        problems.append(f"cited {c}, which was not in the context — fabricated citation")

    supplied = ctx.values | ctx.pages
    # Strip the citations first: [F13028] is an id, not a figure, and scanning
    # it as one reports the answer's own citation as a hallucinated number.
    prose = CITE_RE.sub(" ", answer)
    for m in ANSWER_NUM_RE.findall(prose):
        raw = m.replace("$", "")
        # A bare four-digit integer in this range is a year. "2,031" and
        # "$2031.5" are figures — the comma and the decimal say so. Do not
        # widen this to a numeric range: 1,577 is the figure we most need to
        # check and it sits below any plausible year cutoff.
        if re.fullmatch(r"\d{4}", raw) and 1900 <= int(raw) <= 2100:
            continue
        try:
            v = abs(float(raw.replace(",", "")))
        except ValueError:
            continue
        if not any(abs(v - s) < 0.01 or abs(v * 1000 - s) < 0.01 or abs(v / 1000 - s) < 1e-6
                   for s in supplied):
            problems.append(f"figure {m} is in no supplied fact — the model computed or recalled it")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=OLLAMA)
    ap.add_argument("--facts", type=int, default=15, help="facts to retrieve")
    ap.add_argument("--chunks", type=int, default=3, help="passages to retrieve")
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--no-generate", action="store_true", help="route and retrieve only")
    args = ap.parse_args()

    kind, company, year = route(args.question)
    print(f"route        {kind}  company={company or '-'}  fy={year or '-'}", file=sys.stderr)

    ctx = retrieve(args.question, kind, company, year, args.facts, args.chunks)
    print(f"retrieved    {len(ctx.facts)} facts, {len(ctx.chunks)} passages"
          + ("" if ctx.fiscal_year == year else "  (year filter dropped: no match)"),
          file=sys.stderr)

    prompt = render(args.question, ctx)
    if args.show_context or args.no_generate:
        print(prompt)
        if args.no_generate:
            return
        print("\n" + "=" * 70 + "\n")

    answer = generate(prompt, args.model, args.host)
    print(answer)

    problems = verify(answer, ctx)
    if problems:
        print("\n--- UNGROUNDED", file=sys.stderr)
        for p in problems:
            print(f"    {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
