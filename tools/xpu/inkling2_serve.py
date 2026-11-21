#!/usr/bin/env python3
"""OpenAI-compatible chat server over `xpu-inkling2 MODEL --serve CTX` (one request at a time, every prompt cold).

  inkling2_serve.py --runner BIN --model MODEL.gguf --meta DIR --port N [--ctx 49152]

DIR holds the release's tokenizer.json and chat_template.jinja. Sampling: temperature (default 1.0, the release's
generation_config) and seed; top_p must be 1 and penalties 0, anything else is refused. Reasoning text streams as
`reasoning_content`, the answer as `content`, split on <|content_thinking|> and <|content_text|>.
"""
import argparse
import json
import os
import random
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import PreTrainedTokenizerFast

MESSAGE_MODEL, CONTENT_TEXT, EOS, CONTENT_THINKING, END_MESSAGE = 200001, 200004, 200006, 200008, 200010


class Runner:
    def __init__(self, binary, model, ctx):
        self.p = subprocess.Popen([binary, model, "--serve", str(ctx)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        line = self.p.stdout.readline()
        if line.strip() != "ready":
            raise SystemExit("runner did not start: " + line)
        self.lock = threading.Lock()

    def generate(self, ids, max_new, temp, seed):
        """Yields ("t", id) per token, then ("e", prompt_n, gen_n, prefill_s, decode_s)."""
        self.p.stdin.write(f"{max_new} {temp} {seed} " + " ".join(map(str, ids)) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("runner exited")
            w = line.split()
            if w[0] == "t":
                yield ("t", int(w[1]))
            elif w[0] == "e":
                yield ("e", int(w[1]), int(w[2]), float(w[3]), float(w[4]))
                return
            else:
                raise RuntimeError("runner: " + line.strip())


class Splitter:
    """Token ids to reasoning / content text deltas."""

    def __init__(self, tok):
        self.tok = tok
        self.mode = "content"
        self.ids = {"content": [], "reasoning": []}
        self.text = {"content": "", "reasoning": ""}

    def push(self, t):
        if t == CONTENT_THINKING:
            self.mode = "reasoning"
            return None
        if t == CONTENT_TEXT:
            self.mode = "content"
            return None
        if t >= 199998:  # other control tokens carry no text
            return None
        m = self.mode
        self.ids[m].append(t)
        s = self.tok.decode(self.ids[m], skip_special_tokens=True)
        if s.endswith("�"):  # an incomplete UTF-8 sequence: wait for the next token
            return None
        delta, self.text[m] = s[len(self.text[m]):], s
        return (m, delta) if delta else None


def make_handler(args, runner, tok, template):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            pass

        def reply(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                return self.reply(200, {"object": "list", "data": [{"id": args.name, "object": "model"}]})
            if self.path.startswith("/health"):
                return self.reply(200, {"status": "ok"})
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                return self.reply(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            top_k = body.get("top_k")
            if float(body.get("top_p", 1.0)) != 1.0 or top_k not in (None, 0, -1, 1):
                return self.reply(400, {"error": "only top_p 1, and top_k 1 (greedy) or none, are supported"})
            for k in ("presence_penalty", "frequency_penalty"):
                if float(body.get(k, 0) or 0) != 0:
                    return self.reply(400, {"error": k + " must be 0"})
            if float(body.get("repetition_penalty", 1.0) or 1.0) != 1.0:
                return self.reply(400, {"error": "repetition_penalty must be 1"})
            kw = dict(body.get("chat_template_kwargs") or {})
            kw.pop("enable_thinking", None)  # the release's template takes reasoning_effort; thinking cannot be off
            text = tok.apply_chat_template(body["messages"], chat_template=template, add_generation_prompt=True,
                                           tokenize=False, **kw)
            ids = tok.encode(text, add_special_tokens=False)
            temp = 0.0 if top_k == 1 else float(body.get("temperature", 1.0))
            seed = int(body.get("seed", random.getrandbits(63))) & ((1 << 64) - 1)
            max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096)
            stream = bool(body.get("stream"))
            rid = "chatcmpl-%x" % random.getrandbits(48)
            split = Splitter(tok)
            with runner.lock:
                t0 = time.monotonic()
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                last, end = None, None
                for ev in runner.generate(ids, max_new, temp, seed):
                    if ev[0] == "e":
                        end = ev
                        break
                    last = ev[1]
                    d = split.push(ev[1])
                    if d and stream:
                        key = "content" if d[0] == "content" else "reasoning_content"
                        self.sse({"id": rid, "object": "chat.completion.chunk", "model": args.name,
                                  "choices": [{"index": 0, "delta": {key: d[1]}, "finish_reason": None}]})
                _, pn, gn, ps, ds = end
                usage = {"prompt_tokens": pn, "completion_tokens": gn, "total_tokens": pn + gn}
                timings = {"prompt_n": pn, "prompt_ms": ps * 1e3, "predicted_n": gn, "predicted_ms": ds * 1e3,
                           "predicted_per_second": (gn - 1) / ds if ds > 0 and gn > 1 else None,
                           "wall_ms": (time.monotonic() - t0) * 1e3}
                finish = "stop" if last == EOS else "length"
                if stream:
                    self.sse({"id": rid, "object": "chat.completion.chunk", "model": args.name,
                              "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage,
                              "timings": timings})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    self.close_connection = True
                    return
                msg = {"role": "assistant", "content": split.text["content"],
                       "reasoning_content": split.text["reasoning"]}
                self.reply(200, {"id": rid, "object": "chat.completion", "model": args.name,
                                 "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                 "usage": usage, "timings": timings})

        def sse(self, obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ctx", type=int, default=49152)
    ap.add_argument("--name", default="inkling")
    args = ap.parse_args()
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.meta, "tokenizer.json"))
    template = open(os.path.join(args.meta, "chat_template.jinja")).read()
    runner = Runner(args.runner, args.model, args.ctx)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args, runner, tok, template))
    print(f"listening on 127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
