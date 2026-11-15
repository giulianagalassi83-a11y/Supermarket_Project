"""Rule-based product attributes (brand, size, pack, variant flags) and the match veto.

A veto only fires on a *known* conflict: missing sizes or brands never reject a pair on their own.
"""

import re
from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd

from src.pipeline import config
from src.pipeline.text import is_private_label, parse_pack, strip_accents


# flag -> regex on the accent-free, lowercase name. Order matters inside a group (first hit wins).
_VARIANT_PATTERNS = [
    ("semidesnatada", r"\bsemi\s*-?\s*desnatad[ao]s?\b|\bsemi\b"),
    ("desnatada", r"\bdesnatad[ao]s?\b"),
    ("entera", r"\benter[ao]s?\b"),
    ("virgen_extra", r"\bvirgen\s+extra\b"),
    ("orujo", r"\borujo\b"),
    ("refinado", r"\brefinad[ao]\b|\bsuave\b|\bintenso\b|\b0[,.]4\s*o?\b|\b1\s*o\b"),
    ("virgen", r"\bvirgen\b"),
    ("sin_lactosa", r"\bsin\s+lactosa\b"),
    ("integral", r"\bintegral(es)?\b"),
    ("zero", r"\bzero\b"),  # "0 %" is left out: it tags fat/sugar claims, not a product variant
    ("light", r"\blight\b"),
]

# Mutually exclusive groups: one flag per group, two different flags = conflict.
_GROUPS = {
    "semidesnatada": "milk_fat", "desnatada": "milk_fat", "entera": "milk_fat",
    "virgen_extra": "oil_grade", "virgen": "oil_grade", "refinado": "oil_grade", "orujo": "oil_grade",
    "zero": "reduced", "light": "reduced",
}
# Present on one side and absent on the other = conflict.
_PRESENCE = {"sin_lactosa", "integral", "zero"}


@dataclass(frozen=True)
class Attrs:
    brand: str | None
    own_brand_chain: str | None
    pack_count: int | None
    unit_quantity: float | None
    base_unit: str | None
    variants: frozenset
    name: str = ""  # accent-free lowercase name, for brand mentions
    brand_generic: bool = False  # brand is an everyday word ('pan', 'blanco'): its mention in a name proves nothing
    size_from_db: bool = False  # size/pack came from the products table, not the name: noisy (drained vs net weight)


def _norm_brand(brand):
    if brand is None or (isinstance(brand, float)) or not re.search(r"\w", str(brand)):
        return None  # '-' and other placeholders mean "no brand"
    return strip_accents(str(brand)).replace("®", "").replace("™", "").strip() or None


_STOP = {"la", "el", "los", "las", "de", "del", "the", "and", "y"}


def _tokens(brand):
    return [t for t in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", brand)) if t not in _STOP]


def _one_edit(s, t):
    """True when t is s with one letter inserted, deleted or changed."""
    if abs(len(s) - len(t)) > 1:
        return False
    if len(s) > len(t):
        s, t = t, s
    i = next((k for k in range(len(s)) if s[k] != t[k]), len(s))
    return s[i:] == t[i + 1:] or (len(s) == len(t) and s[i + 1:] == t[i + 1:])


def _is_subsequence(s, t):
    """The letters of s appear in t in order ('fco' in 'francisco')."""
    rest = iter(t)
    return all(c in rest for c in s)


def _token_match(t, u, t_last=False):
    """How well brand word t (from the brand being covered) matches word u: 2 = strong (same word, or t is u cut off
    at the end of a truncated brand: 'gourm' ~ 'gourmet'), 1 = short word or abbreviation ('g' ~ 'gallina',
    'fco' ~ 'francisco'), 0.5 = one-letter typo ('mama' ~ 'mamma'), 0 = no match."""
    if t == u:
        return 2 if len(t) >= 4 else 1
    if t_last and len(t) >= 4 and u.startswith(t):
        return 2
    short, long_ = sorted((t, u), key=len)
    if len(t) <= 3 and t[0] == u[0] and _is_subsequence(t, u):
        return 1
    if len(short) >= 4 and _one_edit(t, u):
        return 0.5
    return 0


def _covers(tx, ty, may_be_cut=False):
    """Every token of tx matches a later token of ty, in order, with at least one strong match."""
    j, scores = 0, []
    for i, t in enumerate(tx):
        last = may_be_cut and i == len(tx) - 1
        while j < len(ty) and not _token_match(t, ty[j], last):
            j += 1
        if j == len(ty):
            return False
        scores.append(_token_match(t, ty[j], last))
        j += 1
    return 2 in scores


def brands_compatible(x, y):
    """Same brand under each chain's spelling.

    'Vaca Que Ríe' ~ 'La Vaca Que Rie', 'Johnson's' ~ 'Johnson' (same letters); 'Central Lechera Asturiana' ~
    'Asturiana' (one is the start or end of the other); 'G.Blanca' ~ 'Gallina Blanca', 'Granja S.Fco' ~
    'La Granja San Francisco' (abbreviations); 'Garden Gourm' ~ 'Garden Gourmet' (cut off). A shared common word
    alone ('Casa Tarradellas' vs 'Casa Di Mamma', 'Purina One' vs 'Purina Felix') or a near spelling ('Calvo' vs
    'Calve', 'Barilla' vs 'Marcilla') is not enough.
    """
    tx, ty = _tokens(x), _tokens(y)
    cx, cy = "".join(tx), "".join(ty)
    if not cx or not cy:
        return True
    if cx == cy:
        return True
    (short, ts), (long_, tl) = sorted([(cx, tx), (cy, ty)], key=lambda p: len(p[0]))
    # One brand is the start or end of the other: on a word boundary ('Asturiana' ~ 'Central Lechera Asturiana'),
    # or inside a long word ('Babybel' ~ 'Minibabybel'); short pieces inside a word don't count ('Monte' ≁ 'Monterde').
    starts = {len("".join(tl[:i])) for i in range(len(tl) + 1)}
    if long_.startswith(short) and (len(short) in starts or len(short) >= 7):
        return True
    if long_.endswith(short) and (len(long_) - len(short) in starts or len(short) >= 7):
        return True
    # Chains cut long brand fields at ~12 characters ("Garden Gourm"); only a long brand can end in a cut-off word.
    return _covers(tx, ty, len(x) >= 10) or _covers(ty, tx, len(y) >= 10)


def generic_brands(names, brands, max_other_brands=2):
    """Brands whose text shows up in the names of more than `max_other_brands` unrelated brands' products.

    Such a brand is an everyday word ('pan', 'blanco', 'gourmet'), so finding it in a product name says nothing about
    the maker. Real makers stay distinctive: 'danone' or 'pascual' appear in at most 1–2 other brands' names
    (their sub-brands). The flagged set shrinks as the cutoff rises (19 brands at 2, 11 at 5, 2 at 20), but
    held-out BM↔Consum results are identical for cutoffs 2–20 (results/generic_sensitivity.csv, 15 Nov).
    """
    index = {}
    for i, name in enumerate(names):
        for w in set(re.findall(r"[a-z0-9]+", name)):
            index.setdefault(w, set()).add(i)
    out = set()
    for brand in {b for b in brands if b}:
        words = re.findall(r"[a-z0-9]+", brand)
        if not words:
            continue
        rows = set.intersection(*(index.get(w, set()) for w in words))
        pattern = re.compile(rf"\b{re.escape(brand)}\b")
        others = {brands[i] for i in rows
                  if brands[i] and not brands_compatible(brand, brands[i]) and pattern.search(names[i])}
        if len(others) > max_other_brands:
            out.add(brand)
    return out


def _variants(name):
    text = strip_accents(name)
    flags, groups_seen = set(), set()
    for flag, pattern in _VARIANT_PATTERNS:
        group = _GROUPS.get(flag)
        if group in groups_seen:
            continue
        if re.search(pattern, text):
            flags.add(flag)
            if group:
                groups_seen.add(group)
    return frozenset(flags)


def _db_pack(db_size):
    """(pack_count, unit_quantity, quantity_unit) from the products table → parse_pack's tuple, or None if unknown."""
    if db_size is None:
        return None
    count, quantity, unit = db_size
    if pd.isna(quantity) or not quantity or pd.isna(unit) or not unit:
        return None
    return (int(count) if pd.notna(count) and count else 1), float(quantity), str(unit)


def extract(name: str, brand: str | None, chain_id: str, db_size=None) -> Attrs:
    """Attributes of one product from its name, brand column and chain.

    Size comes from the name; when the name has none, from `db_size` = the products table's (pack_count,
    unit_quantity, quantity_unit) — Consum, Mercadona and Aldi names almost never carry a size (fixed 15 Nov)."""
    pack = parse_pack(name)
    from_db = pack is None and _db_pack(db_size) is not None
    pack = pack or _db_pack(db_size)
    count, quantity, unit = pack if pack else (None, None, None)
    if unit == "UNIDAD" and quantity and float(quantity).is_integer():
        # "Pack de 4 unidades" / "4 rollos": a count of items of unknown size
        count, quantity, unit = int(count * quantity), None, None
    own = chain_id if is_private_label(brand, name) else None
    return Attrs(_norm_brand(brand), own, count, quantity, unit, _variants(name), strip_accents(name),
                 size_from_db=from_db)


def extract_all(products, keys=None, max_other_brands=2) -> dict:
    """product_key → Attrs for `keys` (default: all rows), with brand_generic learned from the whole frame.

    `products` needs product_key, name, brand, chain_id; pass the full catalogue so word counts are reliable.
    """
    by_key = products.drop_duplicates("product_key").set_index("product_key")
    sizes = by_key.reindex(columns=["pack_count", "unit_quantity", "quantity_unit"])
    raw = {k: extract(r["name"], r["brand"], r["chain_id"], tuple(sizes.loc[k])) for k, r in by_key.iterrows()}
    generic = generic_brands([a.name for a in raw.values()], [a.brand for a in raw.values()], max_other_brands)
    keys = by_key.index if keys is None else keys
    return {k: replace(raw[k], brand_generic=raw[k].brand in generic) for k in keys}


def _variant_conflict(a, b):
    for flag in _PRESENCE:
        if (flag in a.variants) != (flag in b.variants):
            return True
    groups_a = {_GROUPS[f]: f for f in a.variants if f in _GROUPS}
    groups_b = {_GROUPS[f]: f for f in b.variants if f in _GROUPS}
    return any(groups_b.get(g, f) != f for g, f in groups_a.items())


def _same_brand(a, b):
    """Compatible brand spellings, or one brand named in the other's product (sub-brand vs maker: 'Activia' in a
    'Danone' product's name). A generic-word brand ('Pan', 'Blanco') named in a name does not count."""
    if brands_compatible(a.brand, b.brand):
        return True
    mentioned = lambda x, other: not x.brand_generic and bool(re.search(rf"\b{re.escape(x.brand)}\b", other.name))
    return mentioned(a, b) or mentioned(b, a)


def _ratio(x, y):
    return min(x, y) / max(x, y) if max(x, y) > 0 else 1.0


def conflict(a: Attrs, b: Attrs, level: Literal["same", "equivalent"]) -> str | None:
    """The reason the pair can't be a match at this level ('variant', 'size', 'pack', 'brand', 'own_brand'), or None."""
    if _variant_conflict(a, b):
        return "variant"
    if level == "same":
        if a.own_brand_chain and b.own_brand_chain and a.own_brand_chain != b.own_brand_chain:
            return "own_brand"
        if bool(a.own_brand_chain) != bool(b.own_brand_chain):
            return "brand"
        if a.brand and b.brand and not _same_brand(a, b):
            return "brand"
        db_pack = a.size_from_db or b.size_from_db
        if a.pack_count and b.pack_count and a.pack_count != b.pack_count and (config.SIM["db_pack_veto"] or not db_pack):
            return "pack"
    # A DB size is noisy (drained vs net weight, rounding): it only vetoes at the equivalent tolerance (2×).
    strict = level == "same" and not (a.size_from_db or b.size_from_db)
    tol = config.SIM["same_size_tol"] if strict else config.SIM["equiv_size_tol"]
    if a.unit_quantity and b.unit_quantity:
        if a.base_unit != b.base_unit or _ratio(a.unit_quantity, b.unit_quantity) < 1 - tol:
            return "size"
    return None


def equivalence_tier(a: Attrs, b: Attrs) -> str:
    """Tier of a pair already accepted as 'equivalent': 'strict' (per-item size within strict_equiv_size_tol, fit
    for basket swaps), 'loose' (up to equiv_size_tol, fit for €/unit comparison) or 'size_unknown'.

    Derived from the measured sizes, not labelled, so the judge and the audit keep three labels."""
    if not (a.unit_quantity and b.unit_quantity and a.base_unit == b.base_unit):
        return "size_unknown"
    strict = _ratio(a.unit_quantity, b.unit_quantity) >= 1 - config.SIM["strict_equiv_size_tol"]
    return "strict" if strict else "loose"


def features(a: Attrs, b: Attrs) -> dict:
    """Attribute agreement features for the decision model."""
    size_known = bool(a.unit_quantity and b.unit_quantity and a.base_unit == b.base_unit)
    total_a = (a.pack_count or 1) * (a.unit_quantity or 0)
    total_b = (b.pack_count or 1) * (b.unit_quantity or 0)
    if a.brand and b.brand:
        brand_match = float(_same_brand(a, b))
    else:
        brand_match = 0.5
    return {
        "brand_match": brand_match,
        "size_ratio": _ratio(total_a, total_b) if size_known else 1.0,
        "size_known": float(size_known),
        "variant_agree": float(not _variant_conflict(a, b)),
        "pack_match": float(not (a.pack_count and b.pack_count and a.pack_count != b.pack_count)),
    }
