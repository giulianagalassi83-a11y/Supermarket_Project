"""Rule-based product attributes (brand, size, pack, variant flags) and the match veto.

A veto only fires on a *known* conflict: missing sizes or brands never reject a pair on their own.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

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


def _norm_brand(brand):
    if brand is None or (isinstance(brand, float)) or not str(brand).strip():
        return None
    return strip_accents(str(brand)).replace("®", "").replace("™", "").strip() or None


_STOP = {"la", "el", "los", "las", "de", "del", "the", "and", "y"}


def brands_compatible(x, y):
    """Loose brand equality across chains' spellings: 'La Vaca Que Ríe' ~ 'Vaca Que Rie', 'Johnson'S' ~ 'Johnson'."""
    tokens = lambda b: [t for t in re.findall(r"[a-z0-9]+", b) if t not in _STOP]
    tx, ty = tokens(x), tokens(y)
    cx, cy = "".join(tx), "".join(ty)
    if not cx or not cy:
        return True
    if cx in cy or cy in cx:
        return True
    if {t for t in tx if len(t) >= 4} & {t for t in ty if len(t) >= 4}:
        return True
    return SequenceMatcher(None, cx, cy).ratio() >= 0.8


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


def extract(name: str, brand: str | None, chain_id: str) -> Attrs:
    """Attributes of one product from its name, brand column and chain."""
    pack = parse_pack(name)
    count, quantity, unit = pack if pack else (None, None, None)
    if unit == "UNIDAD" and quantity and float(quantity).is_integer():
        # "Pack de 4 unidades" / "4 rollos": a count of items of unknown size
        count, quantity, unit = int(count * quantity), None, None
    own = chain_id if is_private_label(brand, name) else None
    return Attrs(_norm_brand(brand), own, count, quantity, unit, _variants(name), strip_accents(name))


def _variant_conflict(a, b):
    for flag in _PRESENCE:
        if (flag in a.variants) != (flag in b.variants):
            return True
    groups_a = {_GROUPS[f]: f for f in a.variants if f in _GROUPS}
    groups_b = {_GROUPS[f]: f for f in b.variants if f in _GROUPS}
    return any(groups_b.get(g, f) != f for g, f in groups_a.items())


def _same_brand(a, b):
    """Compatible brand spellings, or one brand named in the other's product (sub-brand vs maker)."""
    if brands_compatible(a.brand, b.brand):
        return True
    mentioned = lambda brand, name: bool(re.search(rf"\b{re.escape(brand)}\b", name))
    return mentioned(a.brand, b.name) or mentioned(b.brand, a.name)


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
        if a.pack_count and b.pack_count and a.pack_count != b.pack_count:
            return "pack"
    tol = config.SIM["same_size_tol"] if level == "same" else config.SIM["equiv_size_tol"]
    if a.unit_quantity and b.unit_quantity:
        if a.base_unit != b.base_unit or _ratio(a.unit_quantity, b.unit_quantity) < 1 - tol:
            return "size"
    return None


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
