#!/usr/bin/env python3
"""
Check teacher outputs and split them into training-ready and review/reject files.

Inputs
  teacher_outputs.jsonl   from gen_teacher.py
  math_answers.csv        computed answers for the math prompts
  teacher_answer_review_ids.txt   safety / sensitive IDs (always go to human review)

Outputs (in --outdir)
  sft_train.jsonl         accepted train rows as {"id", "category", "messages": [user, assistant]}
  eval_teacher.jsonl      teacher answers for eval rows (reference answers for judging students)
  review_queue.csv        rows a person must read before they are used (safety + flagged)
  rejected.jsonl          rows that failed a hard check, with reasons
  filter_report.md        counts per check and per category

Run
  python filter_teacher.py --teacher teacher_outputs.jsonl --math math_answers.csv \
      --review_ids teacher_answer_review_ids.txt --outdir filtered
"""
import argparse, csv, json, math, os, re
from collections import Counter, defaultdict

GEEZ = re.compile(r"[ሀ-፿ᎀ-᎟ⶀ-⷟꬀-꬯]")
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
THINK = re.compile(r"<think>|</think>|<\|channel\|>|<unused\d+>|^thought\b", re.I | re.M)
REFUSAL = re.compile(r"(I can't|I cannot|I'm sorry, but|I am unable|ልረዳህ አልችልም|ልረዳሽ አልችልም|መርዳት አልችልም|ይቅርታ፣ ግን)", re.I)

# Which language the ANSWER should be in.
EXPECT_EN = {"translation_am_en"}
SKIP_LANG = {"format"}            # JSON / numbers / lists: language check not meaningful
LOW_AM = {"code_switch", "latin_amharic", "english_instruction"}  # Amharic answer, but English terms allowed


def geez_share(text):
    letters = LETTER.findall(text)
    if not letters:
        return None
    return sum(1 for c in letters if GEEZ.match(c)) / len(letters)


def repetition_score(text, n=4):
    """Share of repeated word 4-grams; high values mean looping output."""
    w = text.split()
    if len(w) < 40:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    c = Counter(grams)
    return sum(v - 1 for v in c.values() if v > 1) / len(grams)


def max_line_repeat(text):
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 15]
    return max(Counter(lines).values()) if lines else 0


# ---------- math answer checking ----------
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text):
    out = []
    for m in NUM.findall(text):
        s = m.replace(",", "")
        try:
            out.append(float(s))
        except ValueError:
            pass
    # h:mm times -> decimal hours, so 8:30 matches 8.5
    for h, mnt in re.findall(r"\b(\d{1,2}):(\d{2})\b", text):
        out.append(int(h) + int(mnt) / 60)
        out.append((int(h) % 12) + int(mnt) / 60)
    # simple fractions a/b
    for a, b in re.findall(r"\b(\d+)\s*/\s*(\d+)\b", text):
        if float(b):
            out.append(float(a) / float(b))
    return out


def close(target, cands):
    for c in cands:
        for t in (target, target * 100, target / 100 if target else None):
            if t is None:
                continue
            tol = max(0.011, abs(t) * 0.005)
            if abs(c - t) <= tol:
                return True
        # hours given as h + minutes, e.g. 4.0441 h -> "4 ሰዓት ከ3 ደቂቃ" is checked via minutes
    if 0 < target < 48 and not float(target).is_integer():
        mins = round((target - int(target)) * 60)
        if int(target) in cands and mins in cands:
            return True
        if (target % 12) and int(target % 12) in cands and mins in cands:
            return True
    return False


def parse_answer(raw):
    raw = (raw or "").strip()
    if not raw or raw.lower() == "nan":
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        y, m, d = map(int, raw.split("-"))
        return ("date", y, m, d)
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
    if raw.startswith("(") or ("," in raw and len(nums) > 1):
        return ("tuple", [float(x) for x in nums])
    try:
        return ("num", float(raw.replace(",", "")))
    except ValueError:
        return ("text", raw)


def check_math(resp, ans_row):
    status = ans_row.get("status", "ok")
    if status != "ok":
        return "manual", f"{status}: check the model points it out"
    parsed = parse_answer(ans_row.get("computed_answer"))
    if parsed is None or parsed[0] == "text":
        return "manual", "non-numeric expected answer"
    cands = numbers_in(resp)
    # Look mainly at the end of the answer, where the final result usually is.
    tail = numbers_in(resp[-400:])
    if parsed[0] == "num":
        if close(parsed[1], tail) or close(parsed[1], cands):
            return "pass", ""
        return "fail", f"expected {parsed[1]:g}"
    if parsed[0] == "tuple":
        missing = [v for v in parsed[1] if not close(v, cands)]
        return ("pass", "") if not missing else ("fail", f"missing {missing}")
    if parsed[0] == "date":
        _, y, m, d = parsed
        ok = (y in cands or float(y) in cands) and float(d) in cands
        return ("pass", "") if ok else ("fail", f"expected date {y}-{m:02d}-{d:02d}")
    return "manual", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--math", required=True)
    ap.add_argument("--review_ids", required=True)
    ap.add_argument("--outdir", default="data/sft")
    ap.add_argument("--min_am", type=float, default=0.70, help="min Ge'ez share of letters for Amharic answers")
    ap.add_argument("--min_am_low", type=float, default=0.45, help="same, for code-switch / Latin / English-instruction prompts")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = {}
    with open(args.teacher, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                try:
                    r = json.loads(l)
                    rows[r["id"]] = r  # last write wins if a row was regenerated
                except json.JSONDecodeError:
                    pass
    math_ans = {}
    with open(args.math, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            math_ans[r["id"]] = r
    review_ids = {l.strip() for l in open(args.review_ids, encoding="utf-8") if l.strip()}

    accepted, rejected, review = [], [], []
    stats = Counter()
    per_cat = defaultdict(Counter)

    for rid, r in rows.items():
        resp = (r.get("response") or "").strip()
        cat = r.get("category")
        hard, soft = [], []

        if not resp:
            hard.append("empty")
        if r.get("finish_reason") in ("length", "max_tokens"):
            hard.append("truncated")
        if THINK.search(resp):
            hard.append("thinking_leak")
        if re.search(r"</?(i|b|u|br|span|div|p)\s*/?>", resp):
            hard.append("html_tag")
        if re.search(r"\b(1[5-9]\d\d|20[0-3]\d)\s*ዓ\.?\s*ም", resp):
            soft.append("check_calendar_label")
        rep = repetition_score(resp)
        if rep > 0.25 or max_line_repeat(resp) >= 3:
            hard.append(f"repetition({rep:.2f})")

        share = geez_share(resp)
        if cat not in SKIP_LANG and share is not None:
            if cat in EXPECT_EN:
                if share > 0.05:
                    hard.append(f"lang:expected_en(geez={share:.2f})")
            else:
                need = args.min_am_low if cat in LOW_AM else args.min_am
                if share < need:
                    hard.append(f"lang:expected_am(geez={share:.2f})")

        if REFUSAL.search(resp[:300]) and cat != "safety":
            soft.append("possible_refusal")
        if re.search(r"(\+251|\b0[79]\d{8}\b|https?://|www\.)", resp):
            soft.append("phone_or_link_in_answer")  # teacher may have invented one
        if len(resp) < 15 and cat not in ("format", "math", "news_classification"):
            soft.append("very_short")

        math_result = ""
        if cat == "math" and rid in math_ans:
            res, note = check_math(resp, math_ans[rid])
            math_result = res
            if res == "pass":
                stats["math_pass"] += 1
            if res == "fail":
                hard.append(f"math_wrong({note})")
            elif res == "manual":
                soft.append(f"math_manual({note})")

        needs_person = rid in review_ids or bool(soft)
        rec_reasons = hard + soft
        for x in rec_reasons:
            stats[x.split("(")[0]] += 1

        if hard:
            rejected.append({**r, "reject_reasons": hard + soft})
            per_cat[cat]["rejected"] += 1
            continue
        if needs_person:
            review.append({"id": rid, "split": r.get("split"), "category": cat,
                           "reason": "safety/sensitive list" if rid in review_ids else "; ".join(soft),
                           "prompt": r["messages"][-1]["content"], "response": resp,
                           "math_expected": math_ans.get(rid, {}).get("computed_answer", ""),
                           "decision_keep_yes_no": "", "fixed_response": "", "notes": ""})
            per_cat[cat]["review"] += 1
            continue
        accepted.append(r)
        per_cat[cat]["accepted"] += 1

    def sft(r, resp=None):
        return {"id": r["id"], "category": r.get("category"), "language_mode": r.get("language_mode"),
                "messages": r["messages"] + [{"role": "assistant", "content": resp or r["response"]}]}

    with open(os.path.join(args.outdir, "sft_train.jsonl"), "w", encoding="utf-8") as f:
        for r in accepted:
            if r.get("split") == "train":
                f.write(json.dumps(sft(r), ensure_ascii=False) + "\n")
    with open(os.path.join(args.outdir, "eval_teacher.jsonl"), "w", encoding="utf-8") as f:
        for r in accepted:
            if r.get("split") == "eval":
                f.write(json.dumps(sft(r), ensure_ascii=False) + "\n")
    with open(os.path.join(args.outdir, "rejected.jsonl"), "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["id", "split", "category", "reason", "prompt", "response", "math_expected", "decision_keep_yes_no", "fixed_response", "notes"]
    with open(os.path.join(args.outdir, "review_queue.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(review)

    n = len(rows)
    lines = ["# Teacher output filter report", "",
             f"Rows read: **{n}** · accepted: **{len(accepted)}** · to review: **{len(review)}** · rejected: **{len(rejected)}**", "",
             "## Flags (a row can have several)", "", "| flag | rows |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in stats.most_common()]
    lines += ["", "## By category", "", "| category | accepted | review | rejected |", "|---|---|---|---|"]
    for c in sorted(per_cat):
        p = per_cat[c]
        lines.append(f"| {c} | {p['accepted']} | {p['review']} | {p['rejected']} |")
    lines += ["", "After review: rows marked `yes` in review_queue.csv (with `fixed_response` if you edited the answer) "
              "can be merged into sft_train.jsonl / eval_teacher.jsonl with `merge_review.py`."]
    open(os.path.join(args.outdir, "filter_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
