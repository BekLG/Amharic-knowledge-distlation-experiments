"""Step 12: exact duplicates, near-duplicates (char 5-gram Jaccard >= 0.80) within train,
train rows near-duplicating eval rows, and train rows sharing a situation / instruction line
with eval. Writes data/work/train_dedup.jsonl, eval_final.jsonl, step12_removed.jsonl."""
import json
from collections import Counter, defaultdict

from common import WORK, TRAIN_TARGETS, read_jsonl, write_jsonl, dropped
from dedup import norm_key, near_dup_pairs, shareable_lines

SOURCE_RANK = {"aya_dataset": 0, "addisgpt": 0, "amharic_qa": 0, "news": 0, "wiki_split": 0, "dolly": 1,
               "nq_open": 1, "tatoeba": 0, "v2_rewritten": 2, "v2_generated": 3}


def quality_key(r):
    """Lower is better: human-written sources first, then longer (more specific) prompts."""
    return (SOURCE_RANK.get(r["source"], 3), -len(r["prompt"]))


def main():
    train = read_jsonl(WORK / "train_assembled.jsonl") + read_jsonl(WORK / "eval_to_train.jsonl")
    evalr = read_jsonl(WORK / "eval_built.jsonl")
    removed, stats = [], Counter()

    def drop(r, reason):
        removed.append(dropped(r["id"], r.get("original_prompt") or r["prompt"], reason, "step12_dedup"))

    # 1. exact duplicates after normalisation (NFC, whitespace, punctuation)
    seen = {}
    keep = []
    for r in sorted(train, key=quality_key):
        k = norm_key(r["prompt"])
        if k in seen:
            drop(r, f"exact duplicate (after normalisation) of {seen[k]}")
            stats["exact_dup_train"] += 1
        else:
            seen[k] = r["id"]
            keep.append(r)
    eval_keys = {norm_key(r["prompt"]): r["id"] for r in evalr}
    assert len(eval_keys) == len(evalr), "exact duplicate inside eval"
    t2 = []
    for r in keep:
        if norm_key(r["prompt"]) in eval_keys:
            drop(r, f"exact duplicate of eval row {eval_keys[norm_key(r['prompt'])]}")
            stats["exact_dup_eval"] += 1
        else:
            t2.append(r)
    train = t2

    # 2. near duplicates within train
    pairs = near_dup_pairs([r["prompt"] for r in train])
    gone = set()
    for i, j, sim in pairs:
        if i in gone or j in gone:
            continue
        a, b = train[i], train[j]
        worse = b if quality_key(a) <= quality_key(b) else a
        better = a if worse is b else b
        gone.add(i if worse is a else j)
        drop(worse, f"near-duplicate of {better['id']} (char 5-gram Jaccard {sim})")
        stats["near_dup_train"] += 1
    train = [r for k, r in enumerate(train) if k not in gone]

    # 3. train rows near-duplicating an eval row
    pairs = near_dup_pairs([r["prompt"] for r in train], [r["prompt"] for r in evalr])
    gone = set()
    for i, j, sim in pairs:
        if i not in gone:
            gone.add(i)
            drop(train[i], f"near-duplicate of eval row {evalr[j]['id']} (Jaccard {sim})")
            stats["near_dup_eval"] += 1
    train = [r for k, r in enumerate(train) if k not in gone]

    # 4. shared situation / instruction lines between eval and train
    eval_lines = defaultdict(list)
    for r in evalr:
        for l in shareable_lines(r["prompt"]):
            eval_lines[l].append(r["id"])
    t2 = []
    for r in train:
        hit = [l for l in shareable_lines(r["prompt"]) if l in eval_lines]
        if hit:
            drop(r, f"shares a line with eval row {eval_lines[hit[0]][0]}: {hit[0][:60]}")
            stats["shared_line_eval"] += 1
        else:
            t2.append(r)
    train = t2

    counts = Counter(r["category"] for r in train)
    write_jsonl(WORK / "train_dedup.jsonl", train)
    write_jsonl(WORK / "eval_final.jsonl", evalr)
    write_jsonl(WORK / "step12_removed.jsonl", removed)
    out = {"train": len(train), "eval": len(evalr), **stats,
           "out_of_range": {c: counts[c] for c, (lo, hi) in TRAIN_TARGETS.items() if not lo <= counts[c] <= hi},
           "by_category": dict(counts)}
    (WORK / "step12_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
