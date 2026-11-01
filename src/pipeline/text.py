"""Text helpers: accents, prices, pack sizes, clean names, brands, search-term matching."""

import math
import re
import unicodedata

from src.pipeline.units import _ALIAS, to_base

PRIVATE_LABELS = [
    "hacendado", "deliplus", "bosque verde", "compy", "dia", "milsani", "aldi", "el horno",
    "lacura", "auchan", "alcampo", "alipende", "bm", "consum",
]

# Common manufacturer brands, used only when a chain gives no brand column
KNOWN_BRANDS = [
    "pascual", "puleva", "central lechera asturiana", "kaiku", "lauki", "danone", "activia",
    "president", "carbonell", "la masia", "koipe", "brillante", "sos", "la fallera", "gallo",
    "barilla", "bimbo", "campofrio", "calvo", "isabel", "albo", "don simon", "granini", "font vella",
    "lanjaron", "bezoya", "solan de cabras", "ariel", "skip", "dixan", "colhogar", "scottex",
    "renova", "el pozo", "nestle", "hochland", "garcia baquero",
]

_UNIT_WORDS = sorted(_ALIAS, key=len, reverse=True)
_UNIT_RE = "|".join(re.escape(u.lower()) for u in _UNIT_WORDS)
_NUM = r"\d+(?:[.,]\d+)?"
PACK_MULTI = re.compile(rf"(\d+)\s*(?:[a-z]+\s+){{0,2}}x\s*({_NUM})\s*({_UNIT_RE})\b")
PACK_SINGLE = re.compile(rf"({_NUM})\s*({_UNIT_RE})\b")
PACK_PLUS = re.compile(rf"(\d+)\s*\+\s*(\d+)\s*({_UNIT_RE})\b")
PACK_FRACTION = re.compile(rf"(\d+)\s*/\s*(\d+)\s*({_UNIT_RE})\b")
PACK_COUNT = re.compile(r"\bpack\s*(?:de\s*)?(\d+)\b(?!\s*x)")


def strip_accents(text):
    """Lowercase and remove accents: 'Plátano' → 'platano'."""
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _is_blank(x):
    return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and not x.strip())


def parse_price(x):
    """'0,83 €' → 0.83, 'Precio 1.234,50 €' → 1234.5, 1.03 → 1.03, blank → None."""
    if _is_blank(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    match = re.search(r"\d{1,3}(?:\.\d{3})+,\d+|\d+(?:[.,]\d+)?", str(x))
    if not match:
        return None
    token = match.group(0)
    if "," in token:
        token = token.replace(".", "").replace(",", ".")
    return float(token)


def _to_float(s):
    return float(s.replace(",", "."))


def parse_pack(text):
    """'6 briks x 1 Litro' → (6, 1.0, 'LITRO'); '340 g' → (1, 0.34, 'KILO'); nothing found → None."""
    if _is_blank(text):
        return None
    clean = strip_accents(text)
    if match := PACK_MULTI.search(clean):  # "6 x 1 l"
        count, amount, unit = int(match.group(1)), _to_float(match.group(2)), match.group(3)
    elif match := PACK_PLUS.search(clean):  # "4+2 rollos" = 6 rolls
        count, amount, unit = int(match.group(1)) + int(match.group(2)), 1.0, match.group(3)
    elif match := PACK_FRACTION.search(clean):  # "1/2 docena" = 6 units
        count, amount, unit = 1, int(match.group(1)) / int(match.group(2)), match.group(3)
    elif match := PACK_SINGLE.search(clean):  # "1,5 l", optionally followed by "pack 6"
        count, amount, unit = 1, _to_float(match.group(1)), match.group(2)
        if trailing := PACK_COUNT.search(clean, match.end()):
            count = int(trailing.group(1))
    else:
        return None
    quantity, base = to_base(amount, unit)
    return count, float(quantity), base


def _remove_words(text, words):
    for word in sorted(words, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(word)}\b", " ", text)
    return text


def clean_name(name, brand=None):
    """Lowercase, no accents, no brand / own-brand words / sizes / punctuation."""
    text = strip_accents(name)
    text = PACK_MULTI.sub(" ", text)
    text = PACK_SINGLE.sub(" ", text)
    text = text.replace("producto alcampo", " ")
    words = list(PRIVATE_LABELS)
    if not _is_blank(brand):
        words.append(strip_accents(str(brand)).replace("®", "").replace("™", "").strip())
    text = re.sub(r"[^\w\s]", " ", text)
    text = _remove_words(text, words)
    return re.sub(r"\s+", " ", text).strip()


def extract_brand(name, brand_col=None):
    """Brand from the chain's column if given, else from the known lists; None if not found."""
    if not _is_blank(brand_col):
        brand = str(brand_col).replace("®", "").replace("™", "").strip()
        return brand.title() if brand else None
    clean = strip_accents(name)
    for brand in sorted(PRIVATE_LABELS + KNOWN_BRANDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(brand)}\b", clean):
            return brand.title()
    return None


def is_private_label(brand, name):
    """True for the chains' own brands (Hacendado, Dia, Milsani...)."""
    if not _is_blank(brand):
        return strip_accents(str(brand)).replace("®", "").strip() in PRIVATE_LABELS
    clean = strip_accents(name)
    return any(re.search(rf"\b{re.escape(p)}\b", clean) for p in PRIVATE_LABELS)


def match_terms(name, terms):
    """Search terms found in a name as whole words (plural s/es allowed), in `terms` order."""
    clean = strip_accents(name)
    return [
        term for term in terms
        if re.search(r"\b" + re.escape(strip_accents(term)) + r"(?:s|es)?\b", clean)
    ]
