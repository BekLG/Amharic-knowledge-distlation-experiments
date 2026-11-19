"""Step 14: validation checks (all must pass). Writes data/work/validation.json."""
import json
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

from common import (OUT, WORK, RAW, CATEGORIES, LANG_MODES, SOURCES, TRAIN_TARGETS, EVAL_TARGETS,
                    read_csv, token_estimate)
from dedup import norm_key, near_dup_pairs, shareable_lines

FIELDS = ["id", "category", "subcategory", "language_mode", "messages", "source", "origin_id",
          "needs_human_review", "split"]
ID_RE = re.compile(r"^pilot_[0-9a-f]{24}$")
PHONE_RE = re.compile(r"(?<!\d)(\+?251[\s-]?|0)9\d{2}[\s-]?\d{3}[\s-]?\d{3}(?!\d)|(?<!\d)\d{3}[\s-]\d{3}[\s-]\d{4}(?!\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
URL_RE = re.compile(r"https?://|www\.|\b[\w-]+\.(com|org|net|et|io)\b", re.I)
TECH_URL_TOPIC = re.compile(r"URL|ድረ ገጽ|website|ሊንክ|link", re.I)
# company names that look like domains inside encyclopedic English passages (not links)
COMPANY_DOMAINS = re.compile(r"\b(TheStreet\.com|[xX]\.com)\b")
REQ_RE = re.compile(r"Amharic|አማርኛ")
DONT_RE = re.compile("አታባብስ|ምርመራ አታድርግ|ዋጋ አትገምት|ሳትፈጥር|አትፍጠር|አትገምት|የማታውቀውን ጠይቀኝ|እንደማታውቅ አስታውስ|"
                     "አትጨምር|አታቅርብ|እንደሆነ አታስብ")


def load(path):
    rows, errors = [], []
    for i, line in enumerate(open(path, encoding="utf-8")):
        try:
            rows.append(json.loads(line))
        except Exception as e:  # noqa
            errors.append(f"line {i}: {e}")
    return rows, errors


def main():
    res = {}
    train, e1 = load(OUT / "pilot_train.jsonl")
    evalr, e2 = load(OUT / "pilot_eval.jsonl")
    allr = train + evalr
    P = lambda r: r["messages"][0]["content"]  # noqa

    # 1 schema
    bad = []
    for r in allr:
        if list(r.keys()) != FIELDS:
            bad.append((r.get("id"), "fields"))
            continue
        if r["category"] not in CATEGORIES or r["language_mode"] not in LANG_MODES or r["source"] not in SOURCES:
            bad.append((r["id"], "value"))
        if not (isinstance(r["messages"], list) and len(r["messages"]) == 1 and r["messages"][0]["role"] == "user"
                and r["messages"][0]["content"].strip()):
            bad.append((r["id"], "messages"))
        if unicodedata.normalize("NFC", P(r)) != P(r):
            bad.append((r["id"], "not NFC"))
        if not isinstance(r["needs_human_review"], bool) or r["split"] not in ("train", "eval"):
            bad.append((r["id"], "flag/split"))
        if r["origin_id"] is not None and not ID_RE.match(r["origin_id"]):
            bad.append((r["id"], "origin_id"))
        if not r["subcategory"]:
            bad.append((r["id"], "subcategory"))
    bad += [(None, x) for x in e1 + e2]
    bad += [(r["id"], "split mismatch") for r in train if r["split"] != "train"]
    bad += [(r["id"], "split mismatch") for r in evalr if r["split"] != "eval"]
    res["1_schema"] = {"pass": not bad, "problems": bad[:10]}

    # 2 unique ids
    ids = Counter(r["id"] for r in allr)
    dup = [k for k, v in ids.items() if v > 1]
    badfmt = [k for k in ids if not ID_RE.match(k)]
    res["2_unique_ids"] = {"pass": not dup and not badfmt, "duplicates": dup[:5], "bad_format": badfmt[:5]}

    # 3 counts
    tc = Counter(r["category"] for r in train)
    out_rng = {c: tc[c] for c, (lo, hi) in TRAIN_TARGETS.items() if not lo <= tc[c] <= hi}
    ec = Counter(r["category"] for r in evalr)
    eval_bad = {}
    for k, v in EVAL_TARGETS.items():
        n = sum(ec[c] for c in (k if isinstance(k, tuple) else (k,)))
        if n != v:
            eval_bad[str(k)] = n
    latin_eval = sum(1 for r in evalr if r["language_mode"] == "latin_am")
    res["3_counts"] = {"pass": 4950 <= len(train) <= 5050 and len(evalr) == 300 and not out_rng and not eval_bad
                       and latin_eval >= 10,
                       "train": len(train), "eval": len(evalr), "out_of_range": out_rng, "eval_mismatch": eval_bad,
                       "eval_latin_am": latin_eval}

    # 4 duplicates / leakage
    keys = Counter(norm_key(P(r)) for r in allr)
    exact = sum(v - 1 for v in keys.values() if v > 1)
    nd_train = near_dup_pairs([P(r) for r in train])
    nd_cross = near_dup_pairs([P(r) for r in train], [P(r) for r in evalr])
    nd_eval = near_dup_pairs([P(r) for r in evalr])
    tl = {l for r in train for l in shareable_lines(P(r))}
    shared = [(r["id"], l) for r in evalr for l in shareable_lines(P(r)) if l in tl]
    res["4_duplicates"] = {"pass": exact == 0 and not nd_train and not nd_cross and not shared,
                           "exact_duplicates": exact, "near_dup_train_pairs": len(nd_train),
                           "near_dup_train_eval_pairs": len(nd_cross), "near_dup_within_eval": len(nd_eval),
                           "eval_lines_shared_with_train": shared[:5]}

    # 5 repetition
    line_c = Counter()
    for r in train:
        ls = [l.strip() for l in P(r).split("\n") if l.strip()]
        if len(ls) >= 2:
            for l in ls:
                if 10 <= len(l) <= 220 and l not in ("Context:", "ጽሑፍ፦"):
                    line_c[l] += 1
    max_line = line_c.most_common(1)[0] if line_c else ("", 0)
    open_c = Counter(" ".join(P(r).split()[:5]) for r in train)
    max_open = open_c.most_common(1)[0]
    gen = [r for r in train if r["source"] in ("v2_generated", "v2_rewritten")]
    dont_share = sum(1 for r in gen if DONT_RE.search(P(r))) / max(1, len(gen))
    res["5_repetition"] = {"pass": max_line[1] <= 8 and max_open[1] <= 0.01 * len(train),
                           "max_line_uses": max_line[1], "max_line": max_line[0][:80],
                           "max_five_word_opening": max_open[1], "opening": max_open[0],
                           "limit_opening": int(0.01 * len(train)),
                           "generated_with_dont_constraint_share": round(dont_share, 3)}

    # 6 English prompts ask for an Amharic answer
    bad_en = [r["id"] for r in allr if r["language_mode"] == "en" and
              r["category"] not in ("translation_en_am", "translation_am_en") and not REQ_RE.search(P(r))]
    res["6_english_requests"] = {"pass": not bad_en, "missing": bad_en[:10]}

    # 7 length
    toks = [token_estimate(P(r)) for r in train]
    one_line = sum(1 for r in train if "\n" not in P(r).strip() and len(P(r).split()) <= 9) / len(train)
    long600 = sum(1 for t in toks if t > 600) / len(train)
    over = [r["id"] for r in allr if token_estimate(P(r)) > 1500]
    res["7_length"] = {"pass": not over and long600 <= 0.10 and 0.25 <= one_line <= 0.40,
                       "max_tokens": round(max(token_estimate(P(r)) for r in allr), 1), "over_1500": over[:5],
                       "share_over_600": round(long600, 3), "share_one_liners": round(one_line, 3),
                       "one_liner_definition": "single line of at most 9 words (about one phone-screen line in Amharic)",
                       "share_single_line_any_length": round(sum(1 for r in train if "\n" not in P(r).strip()) / len(train), 3)}

    # 8 personal data
    pii = []
    for r in allr:
        p = P(r)
        if PHONE_RE.search(p) or EMAIL_RE.search(p):
            pii.append(r["id"])
        elif URL_RE.search(COMPANY_DOMAINS.sub("", p)) and not TECH_URL_TOPIC.search(p):
            pii.append(r["id"])
    res["8_no_pii_urls"] = {"pass": not pii, "rows": pii[:10]}

    # 9 language mix (train)
    lm = Counter(r["language_mode"] for r in train)
    n = len(train)
    am, en, mx = lm["am"] / n, lm["en"] / n, (lm["mixed"] + lm["latin_am"]) / n
    res["9_language_mix"] = {"pass": am >= 0.75 and 0.05 <= en <= 0.10 and 0.06 <= mx <= 0.09,
                             "am": round(am, 4), "en": round(en, 4), "mixed_plus_latin": round(mx, 4),
                             "counts": dict(lm)}

    # 10 math answers
    ans = pd.read_csv(WORK / "math_answers.csv", keep_default_na=False)
    have = {r["id"] for _, r in ans.iterrows() if r["computed_answer"] or r["status"] != "ok"}
    miss = [r["id"] for r in allr if r["category"] == "math" and r["id"] not in have]
    res["10_math_answers"] = {"pass": not miss, "missing": miss[:5],
                              "math_rows": sum(1 for r in allr if r["category"] == "math")}

    # 11 safety ids listed
    listed = set((OUT / "teacher_answer_review_ids.txt").read_text().split())
    unl = [r["id"] for r in allr if r["category"] == "safety" and r["id"] not in listed]
    res["11_safety_listed"] = {"pass": not unl, "unlisted": unl[:5], "listed_total": len(listed)}

    # extra: every input row is accounted for (kept / changed / removed)
    inputs = set(read_csv(RAW / "pilot_train_5000.csv").id) | set(read_csv(RAW / "pilot_eval_300.csv").id)
    removed = set(pd.read_csv(OUT / "removed.csv").id)
    covered = {r["id"] for r in allr} | {r["origin_id"] for r in allr if r["origin_id"]} | removed
    lost = sorted(inputs - covered)
    res["x_all_inputs_accounted"] = {"pass": not lost, "unaccounted": lost[:10], "inputs": len(inputs)}

    ok = all(v["pass"] for v in res.values())
    res["all_pass"] = ok
    (WORK / "validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for k, v in res.items():
        if k != "all_pass":
            print(("PASS " if v["pass"] else "FAIL ") + k, {a: b for a, b in v.items() if a != "pass"})
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
