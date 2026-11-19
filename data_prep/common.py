"""Shared helpers for the pilot-fix pipeline."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
WORK = ROOT / "data" / "work"
OUT = ROOT / "data" / "prompts"
MANUAL = ROOT / "data_prep" / "manual"  # hand-authored judgements and prompts

CATEGORIES = [
    "native_chat", "english_instruction", "translated_instruction", "summarization",
    "translation_en_am", "translation_am_en", "news_classification", "knowledge_qa",
    "everyday", "ethiopia_culture", "writing", "math", "format", "code_switch",
    "latin_amharic", "creative", "safety",
]
LANG_MODES = ["am", "en", "mixed", "latin_am"]
SOURCES = ["aya_dataset", "addisgpt", "amharic_qa", "dolly", "nq_open", "news",
           "wiki_split", "tatoeba", "v2_generated", "v2_rewritten"]

TRAIN_TARGETS = {
    "native_chat": (400, 520), "translated_instruction": (180, 220),
    "english_instruction": (150, 200), "summarization": (280, 320),
    "translation_en_am": (230, 250), "translation_am_en": (190, 200),
    "news_classification": (90, 110), "knowledge_qa": (250, 350),
    "everyday": (700, 850), "ethiopia_culture": (400, 480), "writing": (400, 480),
    "math": (240, 260), "format": (100, 120), "code_switch": (180, 220),
    "latin_amharic": (120, 150), "creative": (100, 130), "safety": (120, 150),
}
EVAL_TARGETS = {
    "native_chat": 50, "everyday": 30, "ethiopia_culture": 25, "writing": 25, "math": 25,
    ("code_switch", "latin_amharic"): 25, "safety": 20, "creative": 15, "format": 15,
    "knowledge_qa": 20, "summarization": 20, ("translation_en_am", "translation_am_en"): 20,
    ("english_instruction", "translated_instruction"): 10,
}


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s.replace("﻿", "").replace("\r\n", "\n"))
    s = "\n".join(l.rstrip() for l in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def is_geez(c):
    return "ሀ" <= c <= "፿" or "ᎀ" <= c <= "᎟" or "ⶀ" <= c <= "⷟"


def is_ethiopic_letter(c):
    # letters only (exclude Ethiopic punctuation ፡።፣፤፥፦፧፨ and numerals ፩-፼)
    return is_geez(c) and not ("፠" <= c <= "፼")


ROMAN_AM_WORDS = set("""
ena min alegn alegñ efeligalehu efelgalehu efelegalehu new nigerign negergn asredagn asredañ yelegnim yeleñim
asibalehu asbalehu alawekum alawkm ye be le lemin lemn endet endemin bezu gin sira sra bet chemir chemr asayegn asayeñ
timikir tmkr selam zare timhirt temari ketero lijemir ljemr yemikerew mekoyet altechegnim kalehu gize tinish tnsh
bemin endemigebagn mejemeriya hulunim atigemit achir melekekt guwadenyaye guadenyaye guadeñaye meftihewoch meftihe
sost akerareb kechalke misale lematkel ayinet atkilt kelal kelalu wechi bezubignal mekoteb birr lemkeyer bota
yet endejemer altesmamam bezih guday wetat nibab kibeb lemekefet asredagni alchalkum endet lemadreg yemiyaskefa
ahun lela neger yelem ishi bicha tiru wede kene keza yihe yih ante anchi enem ene
""".split())


def latin_am_score(s: str) -> float:
    w = re.findall(r"[a-zñ']+", s.lower())
    if not w:
        return 0.0
    return sum(x in ROMAN_AM_WORDS for x in w) / len(w)


AMHARIC_REQUESTS = ["መልሱን በአማርኛ ስጠኝ።", "በአማርኛ አስረዳኝ እባክህ።", "በአማርኛ መልስልኝ።", "እባክህ መልሱን በአማርኛ ጻፍልኝ።",
                    "አማርኛ ላይ ብታስረዳኝ ይሻለኛል።", "በአማርኛ ቢሆን ደስ ይለኛል።"]
# programming / file-format tokens are not counted as English words when measuring the mix
TECH_TOKENS = {"python", "javascript", "markdown", "yaml", "java", "excel"}


def _letters_for_mode(s):
    for phrase in AMHARIC_REQUESTS:  # an Amharic answer request on an English prompt keeps it English
        s = s.replace(phrase, " ")

    def keep(m):
        run = m.group(0)
        if (run.isupper() and len(run) >= 2) or "_" in run or any(ch.isdigit() for ch in run) \
                or run.lower() in TECH_TOKENS:
            return " "
        return run
    s = re.sub(r"[A-Za-z_0-9]+", keep, s)
    return [c for c in s if c.isalpha()]


def language_mode(s: str) -> str:
    """am: >95% of letters Ge'ez; en: no Ge'ez letters and not romanised Amharic (an appended Amharic
    answer request does not count); latin_am: romanised Amharic; mixed: Ge'ez text with English words.
    ALL-CAPS acronyms, snake_case keys and programming-language names are not counted as English words."""
    letters = _letters_for_mode(s)
    if not letters:
        return "am"
    ge = sum(1 for c in letters if is_ethiopic_letter(c))
    if ge == 0:
        return "latin_am" if latin_am_score(s) >= 0.12 else "en"
    if ge / len(letters) > 0.95:
        return "am"
    return "mixed"


def new_id(seed: str) -> str:
    return "pilot_" + hashlib.sha1(("v2|" + seed).encode("utf-8")).hexdigest()[:24]


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, dtype=str)


def lines_of(s):
    return [l.strip() for l in s.split("\n") if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def token_estimate(s: str) -> float:
    ge = sum(1 for c in s if is_geez(c))
    return ge / 2.2 + (len(s) - ge) / 4


def row(rid, category, subcategory, prompt, source, origin_id, needs_review, split="train",
        status="kept", original_prompt=None, change_type="", reason="", language_mode_=None, **extra):
    """Uniform intermediate record used by every step."""
    from common import language_mode as _lm
    r = {
        "id": rid, "category": category, "subcategory": subcategory,
        "language_mode": language_mode_ or _lm(prompt), "prompt": norm_text(prompt),
        "source": source, "origin_id": origin_id, "needs_human_review": bool(needs_review),
        "split": split, "status": status, "original_prompt": original_prompt,
        "change_type": change_type, "reason": reason,
    }
    r.update(extra)
    return r


def dropped(rid, original_prompt, reason, step):
    return {"id": rid, "original_prompt": original_prompt, "reason": reason, "step": step}


def parse_tsv(path, ncols):
    out = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        parts += [""] * (ncols - len(parts))
        out.append(parts[:ncols])
    return out


AMHARIC_REQUEST_RE = r"Amharic|አማርኛ"


_URLISH = re.compile(r"https?://|www\.|t\.me/|\b[\w-]+\.(com|net|org|et|io)\b", re.I)


def strip_credits(text):
    """Drop source-credit / promo sentences (ምንጭ …, URLs, 'subscribe to our Telegram') from a passage."""
    sents = re.split(r"(?<=[።!?])\s*", text)
    keep = [x for x in sents if x and not _URLISH.search(x) and not x.strip().startswith("ምንጭ")
            and "ሰብስክራይብ" not in x and "ሰብስክራይ ያድርጉ" not in x]
    out = " ".join(keep).strip()
    return out if out else text
