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
import contextlib
import io
import json
import logging
import re
import sys
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass, field

# Before importing search, which pulls in transformers and sentence-transformers:
# loading the query embedder prints an HF Hub notice, "All keys matched
# successfully", and a deprecation warning from nomic's remote code. Harmless,
# and checked — but dumped into an interactive session they are just noise, and
# they bury the one line that matters.
warnings.filterwarnings("ignore", message=".*get_extended_attention_mask.*")
for _noisy in ("transformers", "transformers.modeling_utils", "huggingface_hub",
               "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

from search import (  # noqa: E402
    embed_query, hybrid, hybrid_facts, resolve_scope, search, search_facts,
)


OLLAMA = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

REFUSAL = "INSUFFICIENT EVIDENCE"


def warm_up() -> None:
    """Load the query embedder now, quietly.

    It loads lazily on the first question otherwise, which makes that one
    question mysteriously slower than the rest. And it prints an HF Hub
    notice and a "All keys matched successfully" line straight to the
    console rather than through logging, so the only reliable way to keep
    them out of an interactive session is to swallow this one call's output.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        embed_query("warm up")

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
5. Answer in one complete sentence naming the company, the period, the figure
   with its units, and the page. No preamble, no repetition, no bullet list.
   Cite each figure once.

Example of a good answer:
  3M's FY2018 capital expenditure was $1,577 million [F13028], page 126.
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


# How much of each kind of evidence a question gets, by route. Both routes get
# both kinds: the sentence around a number is what makes it checkable, and "why
# did cash flow fall" is usually answered with figures in it. What changes is
# the proportion.
#
# Narrative questions were given ten facts and three chunks and refused seven
# times in ten. "What are the major products AMD sells" is not answerable from
# ten table cells, and three passages is not much of a filing. Numbers below
# are a considered guess, not a measurement — the eval run that would settle
# them has not happened.
BUDGET = {
    "numeric":   {"facts": 10, "chunks": 3},
    "narrative": {"facts": 4, "chunks": 10},
}


def retrieve(question: str, kind: str, company, year,
             k_facts: int | None = None, k_chunks: int | None = None,
             mode: str = "vector") -> Context:
    """Retrieve evidence, weighted towards what this kind of question needs."""
    budget = BUDGET.get(kind, BUDGET["numeric"])
    k_facts = budget["facts"] if k_facts is None else k_facts
    k_chunks = budget["chunks"] if k_chunks is None else k_chunks
    ctx = Context(company=company, fiscal_year=year, kind=kind)
    find_facts = hybrid_facts if mode == "hybrid" else search_facts
    find_chunks = hybrid if mode == "hybrid" else search
    ctx.facts = find_facts(question, k=k_facts, company=company, fiscal_year=year)
    ctx.chunks = find_chunks(question, k=k_chunks, company=company, fiscal_year=year)

    # A filter that matches nothing is worse than no filter: the answer becomes
    # unreachable. Fall back rather than confidently return an empty context.
    if not ctx.facts and not ctx.chunks and (company or year):
        ctx.facts = find_facts(question, k=k_facts, company=company)
        ctx.chunks = find_chunks(question, k=k_chunks, company=company)
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


NUM_THREAD = 6
"""CPU threads for generation.

Left to itself Ollama used about two of eight cores here — 4.5 tok/s.
Measured: 6 threads gives 7.5, 4 gives 6.7, and asking for all 8 gives 4.9,
slower than 6. The same oversubscription result docs/system/batch-runs.md
records for conversion: past the point where the cores are full, more
parallelism costs rather than pays, and something else always wants a core.
"""


def generate(prompt: str, model: str, host: str, stream: bool = True,
             num_thread: int = NUM_THREAD) -> str:
    """Ask the model. Streams by default.

    A minute of silence is indistinguishable from a hang, and the first thing
    anyone does is press a key — which then lands in the next prompt. Printing
    tokens as they arrive costs nothing and turns the wait into progress.
    """
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": stream,
        # Reasoning models put their chain of thought in `message.thinking` and
        # leave `content` empty until it ends. gemma4:26b spent an entire
        # generation thinking and returned an empty answer, which reached
        # verify() as "no citations" — a real failure reported as a formatting
        # one. Nothing here needs deliberation: the task is to copy the right
        # figure out of a supplied list and cite it.
        "think": False,
        # Deterministic: an answer that changes between runs cannot be
        # regression-tested, and this is a regression suite before it is a
        # product.
        "options": {"temperature": 0, "seed": 1, "num_thread": num_thread},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            if stream:
                parts, payload = [], {}
                for line in r:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    piece = payload.get("message", {}).get("content") or ""
                    if piece:
                        parts.append(piece)
                        print(piece, end="", flush=True)
                if parts:
                    print()
                payload = {**payload, "message": {"content": "".join(parts)}}
            else:
                payload = json.loads(r.read())
        message = payload.get("message", {})
        answer = (message.get("content") or "").strip()
        if not answer:
            # Say which failure this is. An empty answer silently becomes "no
            # citations" downstream, which sends you looking at the prompt when
            # the model never spoke at all.
            thinking = len(message.get("thinking") or "")
            reason = payload.get("done_reason", "?")
            sys.exit(
                f"{model} returned an empty answer (done_reason={reason}, "
                f"{thinking} chars of hidden reasoning). If this is a reasoning "
                "model, it may have run out of tokens before it finished thinking."
            )
        return answer
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


def expand(question: str, previous: str | None) -> str:
    """What to search for, when the question alone is not searchable.

    "and 2019?" carries its scope but no topic. Embedded on its own it
    retrieves whatever happens to sit near the word 2019 — divestitures,
    marketable securities — and the follow-up silently answers a different
    question than the one asked.

    So a short follow-up searches with the previous question's words prepended.
    Only the retrieval query is expanded; the model is still shown, and still
    answers, exactly what the user typed.
    """
    if previous and len(question.split()) <= 6:
        return f"{previous} {question}"
    return question


def answer_once(question: str, args, carried: tuple[str | None, int | None],
                previous: str | None = None):
    """One question through the whole pipeline. Returns the scope it used."""
    kind, company, year = route(question)
    query = expand(question, previous)

    # Carry scope across turns so "and 2019?" or "what about net sales?" works.
    # Only ever fills a gap — anything the question states itself wins, because
    # a stale company silently answering about the wrong filer is the worst
    # failure this thing has.
    prev_company, prev_year = carried
    inherited = []
    if company is None and prev_company:
        company, _ = prev_company, inherited.append(f"company={prev_company}")
    if year is None and prev_year:
        year, _ = prev_year, inherited.append(f"fy={prev_year}")

    note = f"  (carried over {', '.join(inherited)})" if inherited else ""
    print(f"route        {kind}  company={company or '-'}  fy={year or '-'}{note}",
          file=sys.stderr)
    if query != question:
        print(f"searching    {query[:88]!r}", file=sys.stderr)

    ctx = retrieve(query, kind, company, year, args.facts, args.chunks,
                   getattr(args, "retrieval", "vector"))
    print(f"retrieved    {len(ctx.facts)} facts, {len(ctx.chunks)} passages"
          + ("" if ctx.fiscal_year == year else "  (year filter dropped: no match)"),
          file=sys.stderr)

    prompt = render(question, ctx)
    if args.show_context:
        print(prompt)
        print("\n" + "=" * 70 + "\n")
    if args.no_generate:
        if not args.show_context:
            print(prompt)
        return company, year

    print("thinking      (Ctrl-C to cancel)", flush=True, file=sys.stderr)
    answer = generate(prompt, args.model, args.host, stream=args.stream)
    if not args.stream:
        print(answer)

    for p in verify(answer, ctx):
        print(f"  !! {p}", file=sys.stderr)
    return company, year


def chat(args) -> None:
    """Ask follow-ups without repeating yourself. Ctrl-D or /quit to leave."""
    print(f"model {args.model}. Answers come from the filings and are")
    print("citation-checked. Ctrl-C cancels an answer; /quit exits.")
    print("/context shows what the model was given, /reset clears company and year.\n")
    carried: tuple[str | None, int | None] = (None, None)
    previous: str | None = None
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q in ("/quit", "/exit"):
            return
        if q == "/context":
            args.show_context = not args.show_context
            print(f"show-context {'on' if args.show_context else 'off'}\n")
            continue
        if q == "/reset":
            carried, previous = (None, None), None
            print("scope cleared\n")
            continue
        try:
            carried = answer_once(q, args, carried, previous)
            previous = q
        except KeyboardInterrupt:
            # Cancel this answer, keep the session. A traceback here is
            # just noise: nothing has gone wrong.
            print("\n  (cancelled)", file=sys.stderr)
        except SystemExit as e:  # a dead Ollama should not kill the session
            print(f"  !! {e}", file=sys.stderr)
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--retrieval", choices=("vector", "hybrid"), default="vector",
                    help="hybrid fuses vector and keyword search; not yet the\n                          default because it is not yet measured end to end")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=OLLAMA)
    ap.add_argument("--facts", type=int, default=None,
                help="facts to retrieve; default depends on the route (see "
                     "BUDGET). Recall keeps climbing past 10, answer quality "
                     "does not: at 50 the 7B stopped citing and rambled")
    ap.add_argument("--chunks", type=int, default=None,
                help="passages to retrieve; default depends on the route")
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--no-generate", action="store_true", help="route and retrieve only")
    ap.add_argument("--chat", action="store_true", help="interactive, follow-ups keep scope")
    ap.add_argument("--no-stream", dest="stream", action="store_false",
                    help="wait for the whole answer instead of printing it as it arrives")
    args = ap.parse_args()

    print("loading...", end="", flush=True, file=sys.stderr)
    warm_up()
    print(" ready", file=sys.stderr)

    if args.chat or not args.question:
        chat(args)
        return

    answer_once(args.question, args, (None, None))


if __name__ == "__main__":
    main()
