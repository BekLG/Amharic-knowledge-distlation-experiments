#!/usr/bin/env python3
"""
Generate teacher (Gemma 4 26B) answers for the Amharic distillation pilot.

- Works with a local Ollama server (default) or any OpenAI-compatible server (llama.cpp, vLLM).
- Standard library only.
- Resumable: rows already in the output file are skipped, so you can stop with Ctrl+C
  and start again at any time.
- Eval rows go first, then train rows interleaved across categories, so a partial run
  is still a balanced sample.

Examples
  # Ollama (model name as shown by `ollama list`)
  python gen_teacher.py --train pilot_train.jsonl --eval pilot_eval.jsonl \
      --out teacher_outputs.jsonl --model gemma4:26b --workers 2

  # llama.cpp server started with: llama-server -m gemma-4-26b.gguf -np 4 -c 16384 --port 8080
  python gen_teacher.py ... --backend openai --url http://localhost:8080 --model gemma4 --workers 4

  # Check throughput on 20 prompts first
  python gen_teacher.py ... --limit 20 --out smoke.jsonl
"""
import argparse, json, os, random, signal, sys, threading, time, urllib.request, urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Teacher-only instruction. It steers the teacher's answers and is NOT saved into the
# training conversations, so the student learns to answer this way without seeing it.
DEFAULT_SYSTEM = (
    "You are a helpful, accurate assistant for Ethiopian users. "
    "Answer in Amharic (Ge'ez script) unless the user clearly asks for another language "
    "or the task is a translation into another language. If the user writes in Latin-script "
    "Amharic or mixes English, still answer in natural Amharic, keeping common English "
    "technical terms where people normally use them. "
    "Be clear and helpful, and keep answers reasonably short unless the user asks for detail. "
    "Do not invent facts, names, statistics, phone numbers, prices or links. If you are not sure, "
    "say so. If information is missing or contradictory, point it out. "
    "For self-harm, abuse or emergencies, respond with care, encourage contacting trusted people "
    "and local emergency services or a health professional, and never give harmful instructions. "
    "Decline requests that would cause harm, briefly and politely, and offer a safe alternative."
)

# Output-length caps per category (tokens). Keeps the run short without cutting useful answers.
MAX_TOKENS = {
    "format": 250, "news_classification": 250, "translation_en_am": 500, "translation_am_en": 450,
    "summarization": 400, "knowledge_qa": 450, "math": 650, "creative": 800, "writing": 700,
}
DEFAULT_MAX_TOKENS = 600

stop_flag = threading.Event()
write_lock = threading.Lock()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def done_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for l in f:
                try:
                    r = json.loads(l)
                    if r.get("response") is not None:
                        ids.add(r["id"])
                except json.JSONDecodeError:
                    pass  # half-written last line after a crash
    return ids


def order_rows(eval_rows, train_rows, seed=13):
    """Eval first, then train round-robin across categories (shuffled within each)."""
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for r in train_rows:
        by_cat[r["category"]].append(r)
    for v in by_cat.values():
        rng.shuffle(v)
    cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))
    inter, i = [], 0
    while any(by_cat.values()):
        for c in cats:
            if i < len(by_cat[c]):
                inter.append(by_cat[c][i])
        i += 1
        if all(i >= len(by_cat[c]) for c in cats):
            break
    return list(eval_rows) + inter


def post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_teacher(args, row, system):
    messages = ([{"role": "system", "content": system}] if system else []) + row["messages"]
    max_tok = MAX_TOKENS.get(row.get("category"), DEFAULT_MAX_TOKENS)
    t0 = time.time()
    if args.backend == "ollama":
        payload = {
            "model": args.model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": args.temperature, "top_p": args.top_p,
                        "num_predict": max_tok, "num_ctx": args.num_ctx, "seed": args.seed},
        }
        if args.keep_alive:
            payload["keep_alive"] = args.keep_alive
        try:
            d = post(args.url.rstrip("/") + "/api/chat", payload, args.timeout)
        except urllib.error.HTTPError as e:
            # Older Ollama builds reject "think"; retry without it.
            if e.code == 400 and "think" in e.read().decode(errors="ignore").lower():
                payload.pop("think", None)
                d = post(args.url.rstrip("/") + "/api/chat", payload, args.timeout)
            else:
                raise
        msg = d.get("message", {})
        text = msg.get("content", "")
        thinking = msg.get("thinking")
        finish = d.get("done_reason", "stop")
        out_tok = d.get("eval_count")
        in_tok = d.get("prompt_eval_count")
    else:
        payload = {"model": args.model, "messages": messages, "temperature": args.temperature,
                   "top_p": args.top_p, "max_tokens": max_tok, "seed": args.seed,
                   "chat_template_kwargs": {"enable_thinking": False}}
        d = post(args.url.rstrip("/") + "/v1/chat/completions", payload, args.timeout)
        ch = d["choices"][0]
        text = ch["message"].get("content") or ""
        thinking = ch["message"].get("reasoning_content")
        finish = ch.get("finish_reason", "stop")
        u = d.get("usage", {})
        out_tok, in_tok = u.get("completion_tokens"), u.get("prompt_tokens")
    return {
        "response": text.strip(), "finish_reason": finish, "thinking_present": bool(thinking),
        "output_tokens": out_tok, "prompt_tokens": in_tok, "max_tokens": max_tok,
        "seconds": round(time.time() - t0, 2),
    }


def worker(args, row, system, out_f):
    if stop_flag.is_set():
        return None
    last_err = None
    for attempt in range(args.retries + 1):
        if stop_flag.is_set():
            return None
        try:
            res = call_teacher(args, row, system)
            rec = {
                "id": row["id"], "split": row.get("split"), "category": row.get("category"),
                "subcategory": row.get("subcategory"), "language_mode": row.get("language_mode"),
                "messages": row["messages"], **res,
                "teacher_model": args.model, "backend": args.backend,
                "gen_params": {"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
                               "system_prompt": "default" if system == DEFAULT_SYSTEM else ("none" if not system else "custom")},
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with write_lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                os.fsync(out_f.fileno())
            return rec
        except Exception as e:  # network hiccup, server restart, timeout
            last_err = e
            time.sleep(min(60, 5 * (attempt + 1)))
    with write_lock:
        sys.stderr.write(f"\n[failed] {row['id']}: {last_err}\n")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", default="data/teacher/teacher_outputs.jsonl")
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--workers", type=int, default=2, help="parallel requests; match OLLAMA_NUM_PARALLEL or llama.cpp -np")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per request")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--keep_alive", default="24h", help="Ollama: keep the model loaded between requests")
    ap.add_argument("--system", default="default", help="'default', 'none', or a path to a text file")
    ap.add_argument("--only", choices=["all", "eval", "train"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new rows (for a smoke test)")
    ap.add_argument("--log_every", type=float, default=30, help="seconds between progress lines")
    args = ap.parse_args()

    system = DEFAULT_SYSTEM if args.system == "default" else ("" if args.system == "none" else open(args.system, encoding="utf-8").read().strip())

    ev = load_jsonl(args.eval) if args.only in ("all", "eval") else []
    tr = load_jsonl(args.train) if args.only in ("all", "train") else []
    rows = order_rows(ev, tr)
    done = done_ids(args.out)
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"total {len(rows)} | already done {len(done)} | to do now {len(todo)} | workers {args.workers}", flush=True)
    if not todo:
        return

    def handle_sigint(sig, frame):
        if not stop_flag.is_set():
            print("\nStopping after the requests in flight finish (Ctrl+C again to force)...", flush=True)
            stop_flag.set()
        else:
            os._exit(1)
    signal.signal(signal.SIGINT, handle_sigint)

    start, n_ok, tok, last_print = time.time(), 0, 0, 0.0
    with open(args.out, "a", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, args, r, system, out_f) for r in todo]
        for f in as_completed(futs):
            rec = f.result()
            if not rec:
                continue
            n_ok += 1
            tok += rec.get("output_tokens") or 0
            el = time.time() - start
            if time.time() - last_print >= args.log_every or n_ok == len(todo):
                last_print = time.time()
                rate = n_ok / el if el else 0
                eta_h = (len(todo) - n_ok) / rate / 3600 if rate else 0
                print(f"[{time.strftime('%H:%M:%S')}] {n_ok}/{len(todo)} | {tok/el:.1f} out tok/s | "
                      f"{rate*3600:.0f} rows/h | ETA {eta_h:.1f} h | last: {rec['category']} {rec['output_tokens']} tok, {rec['finish_reason']}",
                      flush=True)
    print(f"\ndone: {n_ok} rows, {tok} output tokens in {(time.time()-start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
