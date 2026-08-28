"""Serve the pipeline as an OpenAI-compatible API, so any chat UI can front it.

    python serve.py                      # listens on 127.0.0.1:8642

Then point a client at it. Open WebUI:

    pip install open-webui
    OPENAI_API_BASE_URL=http://localhost:8642/v1 \
    OPENAI_API_KEY=none \
    ENABLE_OLLAMA_API=False \
    open-webui serve

and pick the `rag-lab` model. Any OpenAI-compatible client works — LibreChat,
LM Studio, or curl.

Why a shim rather than pointing the UI at Ollama directly: Ollama cannot reach
the database. A chat UI talking straight to it gets a model answering SEC
questions from memory, confidently and wrongly. Everything that makes an answer
trustworthy here — resolving the company and period, retrieving filed values,
checking every citation — happens in this process, before and after the model
is called.

Open WebUI has its own RAG and it is deliberately not used. It would re-chunk
the PDFs naively, discarding the fact-per-table-cell store that moved capital
expenditure from rank 10 to rank 1, along with the page-accurate extraction and
the grounding checks.

Two endpoints are all a chat client needs: GET /v1/models to discover what to
call, POST /v1/chat/completions to talk to it.
"""

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ask

MODEL_ID = "rag-lab"


def run_pipeline(question: str, args) -> str:
    """One question, all the way through. Returns the text a UI should show."""
    kind, company, year = ask.route(question)
    ctx = ask.retrieve(question, kind, company, year, args.facts, args.chunks,
                       args.retrieval)
    prompt = ask.render(question, ctx)
    answer = ask.generate(prompt, args.model, args.host, stream=False)

    scope = f"{company or 'all companies'} · {year or 'all years'}"
    footer = [
        "",
        "---",
        f"*searched {scope} — {len(ctx.facts)} figures, {len(ctx.chunks)} passages*",
    ]
    # Grounding failures belong in the answer, not in a log the reader will
    # never see. An answer that quietly failed its checks is the thing this
    # project exists to prevent.
    problems = ask.verify(answer, ctx)
    if problems:
        footer.append("")
        footer.append("> **Ungrounded — do not trust this answer:**")
        for p in problems:
            footer.append(f"> - {p}")
    return answer + "\n".join(footer)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    args = None  # set in main()

    def log_message(self, fmt, *a):  # one tidy line instead of two noisy ones
        print(f"  {self.address_string()} {fmt % a}")

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(200, {
                "object": "list",
                "data": [{
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "rag-lab",
                }],
            })
            return
        self._send(404, {"error": {"message": f"no route for {self.path}"}})

    def do_POST(self) -> None:
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            self._send(404, {"error": {"message": f"no route for {self.path}"}})
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._send(400, {"error": {"message": f"bad JSON: {e}"}})
            return

        # Only the latest user turn is used. Follow-ups that depend on earlier
        # turns are a real gap, and pretending otherwise by concatenating the
        # history would retrieve against a blur of several questions.
        messages = [m for m in req.get("messages", []) if m.get("role") == "user"]
        if not messages:
            self._send(400, {"error": {"message": "no user message"}})
            return
        question = (messages[-1].get("content") or "").strip()

        try:
            text = run_pipeline(question, self.args)
        except SystemExit as e:  # generate() exits on an unreachable model
            text = f"**The pipeline failed.**\n\n```\n{e}\n```"
        except Exception as e:
            text = f"**The pipeline raised {type(e).__name__}.**\n\n```\n{e}\n```"

        if req.get("stream"):
            self._stream(text, req.get("model", MODEL_ID))
        else:
            self._send(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", MODEL_ID),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
            })

    def _stream(self, text: str, model: str) -> None:
        """Server-sent events, as the OpenAI streaming API defines them.

        The answer is already complete by the time we get here — the pipeline
        has to finish before its citations can be checked, and shipping tokens
        the verifier might then contradict would be worse than a short wait.
        Chunking it out keeps clients happy and shows the answer arriving.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        def event(delta: dict, finish=None) -> None:
            chunk = {
                "id": cid, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        event({"role": "assistant", "content": ""})
        for i in range(0, len(text), 24):
            event({"content": text[i : i + 24]})
        event({}, finish="stop")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--bind", default="127.0.0.1", help="loopback by default")
    ap.add_argument("--model", default=ask.DEFAULT_MODEL)
    ap.add_argument("--host", default=ask.OLLAMA, help="where Ollama is")
    ap.add_argument("--facts", type=int, default=None)
    ap.add_argument("--chunks", type=int, default=None)
    ap.add_argument("--retrieval", choices=("vector", "hybrid"), default="vector")
    args = ap.parse_args()

    print("loading...", end="", flush=True)
    ask.warm_up()
    print(" ready")

    Handler.args = args
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"\nrag-lab API on http://{args.bind}:{args.port}/v1   (model: {MODEL_ID})")
    print(f"generating with {args.model} via {args.host}\n")
    print("point a chat UI at it, for example:")
    print(f"  OPENAI_API_BASE_URL=http://localhost:{args.port}/v1 \\")
    print("  OPENAI_API_KEY=none ENABLE_OLLAMA_API=False open-webui serve\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
