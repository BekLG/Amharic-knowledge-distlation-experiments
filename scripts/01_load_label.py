"""Step 1: load raw files, normalise text, detect generator families and assign
category / subcategory / language_mode / source to every input row.

Writes data/work/labelled.parquet and data/work/step1_counts.json.
"""
import json
import re
import shutil
from collections import Counter

import pandas as pd

from common import RAW, WORK, MANUAL, ROOT, lines_of, norm_text, language_mode, read_csv
from families import GEN1_TAILS, build_instruction_counter, detect

# instruction id -> group (see data/manual/pair_overrides.tsv header)
GROUPS = {
    "home": "I2 I29 I31 I68 G19", "decision": "I8 I11 I61 I64 G18",
    "relationships": "I0 I34 I38 I66 G6", "work": "I16 I22 I28 I44 G9",
    "messages": "I9 I17 I50 I70 G2", "code_switch": "I35 I54 I69 I71 G15",
    "casual": "I3 I30 I48 I51", "study": "I1 I21 I42 I63 G16",
    "science": "I6 I13 I49 I67 G4", "story": "I15 I18 I53 I55 G8",
    "essay": "I26 I27 I33 I65 G11", "event": "I5 I36 I60 I62 G1",
    "places": "I10 I14 I23 I32 G17", "traditions": "I25 I37 I43 I52 G12",
    "food": "I7 I45 I56 I58 G3", "arts": "I19 I24 I40 I46 G5",
    "tech": "I4 I12 I39 I41 G7", "distress": "I72 I73 I75 I77 G14",
    "harmful": "I74 I76 I78 G13 KS", "text_edit": "I20 I47 I57 I59 G10",
    "format_csv": "I79", "latin": "G0",
}
INSTR_GROUP = {i: g for g, ids in GROUPS.items() for i in ids.split()}
GROUP_CATEGORY = {
    "home": ("everyday", "home"), "decision": ("everyday", "decisions"),
    "relationships": ("everyday", "relationships"), "work": ("everyday", "work_business"),
    "messages": ("writing", "messages"), "code_switch": ("code_switch", "tech_work"),
    "casual": ("everyday", "casual_advice"), "study": ("everyday", "study"),
    "science": ("everyday", "science_curiosity"), "story": ("creative", "story"),
    "essay": ("writing", "essay"), "event": ("writing", "event_text"),
    "places": ("ethiopia_culture", "places_heritage"), "traditions": ("ethiopia_culture", "traditions"),
    "food": ("ethiopia_culture", "food_agriculture"), "arts": ("ethiopia_culture", "arts_language"),
    "tech": ("everyday", "tech_help"), "distress": ("safety", "distress_abuse"),
    "harmful": ("safety", "harmful_request"), "text_edit": ("writing", "editing"),
    "format_csv": ("format", "csv"), "latin": ("latin_amharic", "latin"),
}
# food/agriculture situations that are not specifically Ethiopian
GENERIC_FOOD = ["ድንች እርሻ", "የስንዴ ዳቦ", "የጣፋጭ ድንች", "የሙዝ ምርት"]

RES_CAT = {"N": "native_chat", "K": "knowledge_qa", "T": "translated_instruction", "E": "everyday",
           "C": "ethiopia_culture", "W": "writing", "R": "creative", "S": "safety", "X": "code_switch",
           "F": "format", "U": "summarization"}
RES_SRC = {"a": "aya_dataset", "g": "addisgpt", "q": "amharic_qa", "d": "dolly", "n": "nq_open",
           "v": "v2_generated"}
KS_TAIL = "ምንም ማብራሪያ ሳትጨምር መልሱን ብቻ ስጠኝ።"


def load_instr_ids(train_prompts):
    """Instruction id table (I0..I79) ordered by frequency, ties by first appearance."""
    c = Counter()
    order = {}
    for p in train_prompts:
        ls = lines_of(p)
        if len(ls) == 2:
            c[ls[1]] += 1
            order.setdefault(ls[1], len(order))
    return c


def parse_residual_labels(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        idx, code = line.rstrip("\n").split(" ", 1)
        out[int(idx)] = code
    return out


def situation_split(prompt, fam, instr_ids, tails):
    """Return (situation, instruction_key) for pair / gen1 rows."""
    ls = lines_of(prompt)
    if fam == "pair":
        return ls[0], instr_ids[ls[1]]
    for t in tails:
        if prompt.endswith(t) and prompt != t:
            return prompt[: -len(t)].strip(), tails[t]
    raise ValueError(prompt)


def main():
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    for f in ["pilot_train_5000.csv", "pilot_eval_300.csv"]:
        if not (RAW / f).exists():
            shutil.copy(ROOT / f, RAW / f)
    WORK.mkdir(parents=True, exist_ok=True)

    tr = read_csv(RAW / "pilot_train_5000.csv")
    ev = read_csv(RAW / "pilot_eval_300.csv")
    tr["split_in"] = "train"
    ev["split_in"] = "eval"
    df = pd.concat([tr[["id", "task", "direction", "domain", "prompt", "split_in"]],
                    ev[["id", "task", "direction", "domain", "prompt", "split_in"]]], ignore_index=True)
    df["prompt"] = df.prompt.map(norm_text)
    ic = build_instruction_counter(df[df.split_in == "train"].prompt)
    df["family"] = [detect(r, ic) for r in df.to_dict("records")]

    # instruction ids for the train pair table: same ordering as the review files
    pair_instr = Counter()
    for p, fam in zip(df.prompt, df.family):
        if fam == "pair":
            pair_instr[lines_of(p)[1]] += 1
    ordered = sorted(pair_instr, key=lambda x: (-pair_instr[x], list(pair_instr).index(x)))
    instr_ids = {x: f"I{i}" for i, x in enumerate(ordered)}
    tails = {t: f"G{i}" for i, t in enumerate(sorted(GEN1_TAILS))}
    (WORK / "instruction_ids.json").write_text(json.dumps(instr_ids, ensure_ascii=False, indent=1))

    # knowledge tail rows: safety prompts carry KS tail; they share situations with the harmful group
    sit, ins = [], []
    for p, fam in zip(df.prompt, df.family):
        if fam in ("pair", "gen1"):
            s, i = situation_split(p, fam, instr_ids, tails)
        elif fam == "knowledge_tail" and p.endswith(KS_TAIL):
            s, i = lines_of(p)[0], "KS"
        else:
            s, i = "", ""
        if s.endswith(" Any tips?"):
            s = s[: -len(" Any tips?")]
        sit.append(s)
        ins.append(i)
    df["situation"] = sit
    df["instr_key"] = ins

    # ---- category / source -------------------------------------------------
    res_labels = parse_residual_labels(MANUAL / "residual_labels.txt")
    kt_labels = parse_residual_labels(MANUAL / "ktail_labels.txt")
    tr_res = df[(df.split_in == "train") & (df.family == "residual")].index.tolist()
    tr_kt = df[(df.split_in == "train") & (df.family == "knowledge_tail")].index.tolist()
    res_map = {ix: res_labels[i] for i, ix in enumerate(tr_res)}
    kt_map = {ix: kt_labels[i] for i, ix in enumerate(tr_kt)}

    cats, subs, srcs, notes, sens = [], [], [], [], []
    for ix, r in df.iterrows():
        fam, cat, sub, src, note, sensitive = r.family, None, "", "v2_generated", "", False
        if fam == "tr_en_am":
            cat, sub, src = "translation_en_am", "wiki_sentence", "wiki_split"
        elif fam == "tr_am_en":
            cat, sub, src = "translation_am_en", "news", "news"
        elif fam == "news_sum":
            cat, sub, src = "summarization", "news", "news"
        elif fam == "news_cls":
            cat, src = "news_classification", "news"
            sub = "entities" if "ስሞች" in r.prompt.split("\n")[0] else "topic_label"
        elif fam == "wiki_split":
            cat, sub, src = "writing", "sentence_rewrite", "wiki_split"
        elif fam.startswith("math_"):
            cat, sub = "math", fam[5:]
        elif fam in ("format", "format_text"):
            cat, sub = "format", "list_json"
        elif fam == "passage_qa":
            cat, sub, src = "knowledge_qa", "reading_comprehension", "amharic_qa"
        elif fam == "aya_ctx":
            cat, sub, src = "native_chat", "reading_comprehension", "aya_dataset"
        elif fam == "dolly_ctx":
            cat, sub, src = "english_instruction", "context_qa", "dolly"
        elif fam == "en":
            cat, sub = "english_instruction", "general"
            src = "dolly"  # refined in step 2 (nq_open trivia vs dolly)
        elif fam == "latin":
            cat, sub = "latin_amharic", "latin"
        elif fam in ("pair", "gen1") or (fam == "knowledge_tail" and r.instr_key == "KS"):
            g = INSTR_GROUP.get(r.instr_key, "")
            cat, sub = GROUP_CATEGORY[g]
            if g == "food" and any(k in r.situation for k in GENERIC_FOOD):
                cat, sub = "everyday", "food"
            if g == "latin":
                cat, sub = "latin_amharic", "latin"
        elif fam == "knowledge_tail":
            code = kt_map[ix]
            cat, sub, src = "knowledge_qa", "trivia", "nq_open"
            if code.startswith("D"):
                note = code
        elif fam == "residual" and r.split_in == "train":
            code = res_map[ix]
            if code.startswith("D"):
                cat, sub, src, note = "knowledge_qa", "", "dolly", code
            else:
                base = code.split("|")[0]
                cat, src = RES_CAT[base[0]], RES_SRC[base[1]]
                sensitive = "!" in base
                sub = {"knowledge_qa": "trivia" if src == "nq_open" else "concept",
                       "native_chat": "native", "translated_instruction": "dolly_mt",
                       "everyday": "casual_advice", "summarization": "aya_summary",
                       "code_switch": "tech_work"}.get(cat, "")
                if "|" in code:
                    note = "FIX|" + code.split("|", 1)[1]
        else:  # eval residual: labelled in the eval step
            cat, sub, src = "unassigned", "", "v2_generated"
        cats.append(cat)
        subs.append(sub)
        srcs.append(src)
        notes.append(note)
        sens.append(sensitive)
    df["category"] = cats
    df["subcategory"] = subs
    df["source"] = srcs
    df["note"] = notes
    df["sensitive"] = sens
    df["language_mode"] = df.prompt.map(language_mode)
    df["instr_group"] = df.instr_key.map(lambda k: INSTR_GROUP.get(k, ""))

    df.to_parquet(WORK / "labelled.parquet", index=False)
    t = df[df.split_in == "train"]
    counts = {
        "train_rows": int(len(t)), "eval_rows": int((df.split_in == "eval").sum()),
        "family": t.family.value_counts().to_dict(),
        "category": t.category.value_counts().to_dict(),
        "language_mode": t.language_mode.value_counts().to_dict(),
        "source": t.source.value_counts().to_dict(),
        "flagged_for_drop_in_labelling": int(t.note.str.startswith("D").sum()),
    }
    (WORK / "step1_counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=1))
    print(json.dumps(counts, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
