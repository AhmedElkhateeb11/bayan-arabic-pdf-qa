"""Bayan preprocessing: direct PDF text extraction (no OCR) and Arabic text cleaning.

Pipeline (each step is a separate, testable function):
  Page level (frozen in corpus v1, defines passage boundaries)
    1. nfkc()            Unicode NFKC: Arabic presentation forms -> standard letters
    2. remove_tatweel()  delete kashida (U+0640) used for justification
  Passage level (clean view, used for reading / answer generation)
    3. boilerplate          drop running headers/footers/side tabs and page-number lines
    4. remove_glyph_names() drop font glyph names leaked into the text (/uni0627, two.fitted)
    5. unify_letters()      Persian/Urdu look-alike code points -> Arabic letters
    6. repair_spacing()     re-attach diacritics separated by a space, collapse spaces
    7. split_glued()        split two words glued by extraction (conservative, lexicon-checked)
  Retrieval views (never applied to the clean text)
    8. normalize_for_retrieval()  diacritics, alef/ya/ta-marbuta variants, digits, punctuation
    9. tfidf_tokens()    tokenisation, stop-word removal + light stemming (TF-IDF / BM25 view only)
  Segmentation
   10. sentences()       sentence split on Arabic/Latin end punctuation
"""
from __future__ import annotations

import collections
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- characters
DIACRITICS = re.compile(r"[ً-ْٰ]")          # harakat, tanween, shadda, sukun, dagger alef
TATWEEL = "ـ"
PRESENTATION_FORMS = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
ARABIC_LETTER = re.compile(r"[ء-ي]")
ARABIC_WORD = re.compile(r"^[ء-ي]+$")
LATIN = re.compile(r"[A-Za-z]")
ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
STRIP_PUNCT = ".,:;()«»\"،؛؟!-–—[]{}/'*•…"


# ---------------------------------------------------------------- extraction
def extract_pages(pdf_path: str) -> List[str]:
    """Embedded-text extraction with pypdf (no OCR, no rendering). One string per PDF page."""
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    out = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:  # a malformed page should not stop the corpus build
            out.append("")
    return out


# ---------------------------------------------------------------- steps 1-2
def nfkc(text: str) -> str:
    """Step 1. Map presentation forms (e.g. U+FEDF) to standard Arabic letters; also unifies
    compatibility characters. Does not change letters that are already standard."""
    return unicodedata.normalize("NFKC", text)


def remove_tatweel(text: str) -> str:
    """Step 2. Kashida is a visual stretching mark; it splits identical words into different strings."""
    return text.replace(TATWEEL, "")


def page_text(raw: str) -> str:
    """Steps 1+2: the page text from which corpus-v1 passages were cut."""
    return remove_tatweel(nfkc(raw))


# ---------------------------------------------------------------- passages (corpus v1 rule)
PASSAGE_WORDS = 120


def chunk_page(text: str, target: int = PASSAGE_WORDS) -> List[Tuple[int, int]]:
    """Word ranges [start, end) of page-bounded passages: k = round(n/target) (min 1) equal parts."""
    n = len(text.split())
    if n == 0:
        return []
    k = max(1, round(n / target))
    size = -(-n // k)
    return [(j * size, min(n, (j + 1) * size)) for j in range(k) if j * size < n]


def passage_id(file: str, page: int, j: int) -> str:
    return f"{file[6:-4]}_p{page:04d}_{j}"  # strip 'bayan_' and '.pdf'


# ---------------------------------------------------------------- glyph-name artefacts
GLYPH_NAME = re.compile(
    r"/?(?:uni[0-9A-Fa-f]{4,}|u[0-9A-F]{4,6})(?:\.[A-Za-z0-9_]+)*"
    r"|/?\b(?:zero|one|two|three|four|five|six|seven|eight|nine|period|comma|colon|percent|hyphen|space"
    r"|parenleft|parenright|slash)(?:\.[A-Za-z0-9_]+)+")


def remove_glyph_names(text: str) -> Tuple[str, int]:
    """When a font has no Unicode mapping, some extractors emit the glyph's internal name instead
    ('/uni0627', '/uniFEEB', 'two.fitted'). These strings are never words; in this corpus they come
    from chart labels. Remove them and report how many were removed."""
    found = GLYPH_NAME.findall(text)
    if not found:
        return text, 0
    return re.sub(r"\s+", " ", GLYPH_NAME.sub(" ", text)).strip(), len(found)


# ---------------------------------------------------------------- step 3a letter unification
_FARSI_YEH_FINAL = re.compile(r"ی(?![ء-يٮ-ۿ])")


def unify_letters(text: str) -> Tuple[str, int]:
    """Some PDFs encode Arabic text with Persian/Urdu code points that look identical on the page:
    U+06CC (Farsi yeh), U+06BE (heh doachashmee), U+06A9 (keheh). NFKC does not change them, so
    'التقییم' and 'التقييم' would never match. Map them to Arabic letters, following what the
    glyph shows: a word-final U+06CC is drawn without dots, i.e. as ى; elsewhere it is ي."""
    n = len(re.findall(r"[یھک]", text))
    t = _FARSI_YEH_FINAL.sub("ى", text)
    t = t.replace("ی", "ي").replace("ھ", "ه").replace("ک", "ك")
    return t, n


# ---------------------------------------------------------------- step 3b boilerplate
def _line_key(line: str) -> str:
    return re.sub(r"[0-9٠-٩]+", "#", re.sub(r"\s+", " ", line)).strip()


def find_boilerplate(pages: Sequence[str], min_share: float = 0.2, min_pages: int = 3) -> set:
    """Line patterns (digits masked) repeated on >= min_share of a document's pages:
    running headers, footers, side tabs. Lines need >= 2 words and >= 8 Arabic letters, so a
    single repeated word (e.g. a country name in a table) is never treated as boilerplate."""
    counts = collections.Counter()
    for t in pages:
        keys = {_line_key(l) for l in t.split("\n")}
        for k in keys:
            if len(k.split()) >= 2 and len(ARABIC_LETTER.findall(k)) >= 8:
                counts[k] += 1
    thr = max(min_pages, min_share * len(pages))
    return {k for k, c in counts.items() if c >= thr}


def word_flags(text: str, boiler: set) -> List[bool]:
    """For each whitespace token of the page (same order as text.split()), True if it belongs to a
    boilerplate line or to a page-number line (digits only, first or last non-empty line)."""
    lines = text.split("\n")
    nonempty = [i for i, l in enumerate(lines) if l.strip()]
    edge = {nonempty[0], nonempty[-1]} if nonempty else set()
    flags = []
    for i, l in enumerate(lines):
        drop = _line_key(l) in boiler or (i in edge and re.fullmatch(r"\s*[0-9٠-٩]{1,4}\s*", l) is not None)
        flags.extend([drop] * len(l.split()))
    return flags


# ---------------------------------------------------------------- step 4 spacing
def repair_spacing(text: str) -> Tuple[str, int]:
    """Re-attach a diacritic that extraction separated from its letter ('الحُم َّى' -> 'الحُمَّى')
    and collapse runs of whitespace. Returns (text, number of re-attached diacritics)."""
    fixed, n = re.subn(r"\s+(?=[ً-ْٰ])", "", text)
    return re.sub(r"\s+", " ", fixed).strip(), n


# ---------------------------------------------------------------- step 5 glued words
_zcache: Dict[str, float] = {}
_FUNCTION_2 = {"في", "من", "عن", "أو", "أن", "إن", "لا", "ما", "مع"}
_SUFFIX_LIKE = {"ان", "ين", "تين", "تان", "ون", "ات", "هما", "هم", "هن", "ها", "ية", "يا", "وا",
                "كم", "نا", "تهما", "تها", "ته", "وها", "وه", "وهم"}


def _zipf(w: str) -> float:
    if w not in _zcache:
        from wordfreq import zipf_frequency
        _zcache[w] = zipf_frequency(w, "ar")
    return _zcache[w]


def split_candidate(word: str) -> Optional[Tuple[str, str, str]]:
    """Return (left, right, rule) when an unknown token is two known words glued together.
    R1: ta marbuta or alef maqsura inside a token (they only occur word-finally in Arabic).
    R2: a frequent word followed by a frequent 'ال' word.  R3: two very frequent words.
    R4: a frequent word and a two-letter function word (في، من، عن ...) on either side.
    Known words (wordfreq zipf > 0) are never split."""
    w = word
    if len(w) < 5 or not ARABIC_WORD.match(w) or _zipf(w) > 0:
        return None
    for i, ch in enumerate(w[:-1]):
        if ch in "ةى" and i >= 2:
            a, b = w[: i + 1], w[i + 1:]
            if len(b) >= 2 and _zipf(a) >= 3 and _zipf(b) >= 3:
                return a, b, "R1"
    if w[-2:] in _FUNCTION_2 and len(w) >= 6 and _zipf(w[:-2]) >= 4.0:
        return w[:-2], w[-2:], "R4"
    if w[:2] in _FUNCTION_2 and w[2:4] == "ال" and len(w) >= 7 and _zipf(w[2:]) >= 4.0:
        return w[:2], w[2:], "R4"
    if len(w) < 6:
        return None
    best = None
    for i in range(3, len(w) - 2):
        a, b = w[:i], w[i:]
        if b in _SUFFIX_LIKE:
            continue
        if b.startswith("ال") and len(b) >= 4 and _zipf(a) >= 3.5 and _zipf(b) >= 3.5:
            score = (2, min(_zipf(a), _zipf(b)), "R2")
        elif _zipf(a) >= 4.0 and _zipf(b) >= 4.0:
            score = (1, min(_zipf(a), _zipf(b)), "R3")
        else:
            continue
        if best is None or score > best[0]:
            best = (score, a, b)
    return (best[1], best[2], best[0][2]) if best else None


def split_glued(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Step 5. Split glued tokens; punctuation around a token is kept. Returns (text, [(before, after)])."""
    out, changes = [], []
    for tok in text.split():
        core = tok.strip(STRIP_PUNCT)
        bare = DIACRITICS.sub("", core)
        cand = split_candidate(bare) if bare == core else None  # only undiacritised tokens
        if cand:
            a, b, _ = cand
            new = tok.replace(core, f"{a} {b}", 1)
            changes.append((tok, new))
            out.append(new)
        else:
            out.append(tok)
    return " ".join(out), changes


# ---------------------------------------------------------------- clean view
def clean_passage(words: Sequence[str], flags: Sequence[bool]) -> Tuple[str, Dict[str, int]]:
    """Steps 3-7 for one passage, given its page words and boilerplate flags."""
    kept = [w for w, f in zip(words, flags) if not f]
    stats = {"boilerplate_words": len(words) - len(kept)}
    text, stats["glyph_names_removed"] = remove_glyph_names(" ".join(kept))
    text, stats["letters_unified"] = unify_letters(text)
    text, stats["diacritics_reattached"] = repair_spacing(text)
    text, changes = split_glued(text)
    stats["glued_split"] = len(changes)
    return text, stats


# ---------------------------------------------------------------- step 6 retrieval view
def normalize_for_retrieval(text: str) -> str:
    """Orthographic normalisation for matching only: remove diacritics; unify أ إ آ ٱ -> ا,
    ى -> ي, ة -> ه, ؤ -> و, ئ -> ي; Arabic-Indic digits -> 0-9; punctuation -> space; lower-case Latin."""
    t = DIACRITICS.sub("", remove_tatweel(text))
    t = re.sub("[أإآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = t.translate(ARABIC_INDIC_DIGITS).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[ء-ي]+|[a-z]+|\d+(?:[.,]\d+)?", text)


# ---------------------------------------------------------------- step 7 TF-IDF view
_stop: Optional[set] = None
_stemmer = None


def _resources():
    global _stop, _stemmer
    if _stop is None:
        import nltk
        try:
            words = nltk.corpus.stopwords.words("arabic")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = nltk.corpus.stopwords.words("arabic")
        _stop = {normalize_for_retrieval(w) for w in words}
        from nltk.stem.arlstem import ARLSTem
        _stemmer = ARLSTem()
    return _stop, _stemmer


def tfidf_tokens(clean_text: str, stem: bool = True) -> List[str]:
    """TF-IDF/BM25 view of the clean text: NLTK Arabic stop words removed, ARLSTem light stemming
    (prefixes such as و/ال/لل and suffixes such as ات/ين/ة), then the same orthographic
    normalisation as the retrieval view. Stemming runs before ة->ه so the stemmer sees real endings."""
    stop, stemmer = _resources()
    light = DIACRITICS.sub("", remove_tatweel(clean_text)).translate(ARABIC_INDIC_DIGITS).lower()
    light = re.sub(r"[^\w\s]", " ", light)
    out = []
    for t in tokenize(light):
        n = normalize_for_retrieval(t)
        if n in stop or len(n) < 2:
            continue
        out.append(normalize_for_retrieval(stemmer.stem(t)) if (stem and ARABIC_WORD.match(t)) else n)
    return out


# ---------------------------------------------------------------- step 8 sentences
def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!؟?؛])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 1]


# ---------------------------------------------------------------- quality measures (EDA)
def damaged_word_rate(text: str) -> Tuple[int, int]:
    """(Arabic tokens, tokens unknown to the wordfreq Arabic lexicon)."""
    toks = [DIACRITICS.sub("", t).strip(STRIP_PUNCT) for t in text.split()]
    ar = [t for t in toks if ARABIC_WORD.match(t)]
    return len(ar), sum(_zipf(t) == 0 for t in ar)


def build_corpus(pdf_dir: str, kept_pages: Dict[str, set],
                 raw_pages: Optional[Dict[str, List[str]]] = None) -> Tuple[List[dict], Dict[str, List[str]]]:
    """Re-create corpus-v1 passages from the PDFs and attach the clean view.
    kept_pages: {file: {pdf_page, ...}} from the page-level screening.
    raw_pages: optional cache {file: [raw page text]} to skip re-extraction."""
    import os
    passages, raw_pages = [], dict(raw_pages or {})
    for file in sorted(kept_pages):
        if file not in raw_pages:
            path = os.path.join(pdf_dir, file)
            if not os.path.exists(path):
                continue
            raw_pages[file] = extract_pages(path)
        raws = raw_pages[file]
        texts = {i + 1: page_text(r) for i, r in enumerate(raws)}
        boiler = find_boilerplate([texts[p] for p in sorted(kept_pages[file])])
        for p in sorted(kept_pages[file]):
            t = texts[p]
            words, flags = t.split(), word_flags(t, boiler)
            for j, (a, b) in enumerate(chunk_page(t)):
                clean, st = clean_passage(words[a:b], flags[a:b])
                passages.append(dict(passage_id=passage_id(file, p, j), file=file, pdf_page=p,
                                     text_v1=" ".join(words[a:b]), clean_text=clean, **st))
    return passages, raw_pages
