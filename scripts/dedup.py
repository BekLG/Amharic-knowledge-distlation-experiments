"""Shared normalisation / near-duplicate helpers (used by step 12 and validate.py)."""
import re
import unicodedata

from datasketch import MinHash, MinHashLSH

PUNCT = re.compile(r"[\s፡-፨!-/:-@\[-`{-~«»“”‘’…—–]+")


def norm_key(s):
    s = unicodedata.normalize("NFC", s).lower()
    return PUNCT.sub("", s)


def shingles(s, n=5):
    t = norm_key(s)
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def near_dup_pairs(texts_a, texts_b=None, thr=0.8, num_perm=128):
    """Return [(i, j, sim)] with char-5gram Jaccard >= thr. Within texts_a if texts_b is None,
    else between texts_a (i) and texts_b (j)."""
    sh_a = [shingles(t) for t in texts_a]
    sh_b = sh_a if texts_b is None else [shingles(t) for t in texts_b]
    lsh = MinHashLSH(threshold=thr - 0.15, num_perm=num_perm)
    mh_b = []
    for j, s in enumerate(sh_b):
        m = MinHash(num_perm=num_perm)
        for x in s:
            m.update(x.encode("utf-8"))
        mh_b.append(m)
        lsh.insert(str(j), m)
    out = set()
    for i, s in enumerate(sh_a):
        if texts_b is None:
            m = mh_b[i]
        else:
            m = MinHash(num_perm=num_perm)
            for x in s:
                m.update(x.encode("utf-8"))
        for j in map(int, lsh.query(m)):
            if texts_b is None and j <= i:
                continue
            sim = jaccard(s, sh_b[j])
            if sim >= thr:
                out.add((i, j, round(sim, 3)))
    return sorted(out)


def shareable_lines(p):
    """Lines that count as situation / instruction lines (skip passages and tiny labels)."""
    out = []
    for l in p.split("\n"):
        l = l.strip()
        if 12 <= len(l) <= 220 and l not in ("Context:", "ጽሑፍ፦"):
            out.append(l)
    return out
