"""Unit names and conversions. Base units: KILO, LITRO, UNIDAD, LAVADO (others kept as-is)."""

import re

UNIT_ALIASES = {
    "KILO": ["KG", "KGS", "KILO", "KILOS", "KILOGRAMO", "KILOGRAMOS"],
    "LITRO": ["L", "LT", "LTR", "LITRO", "LITROS"],
    "UNIDAD": ["U", "UD", "UDS", "UNIDAD", "UNIDADES", "UNIDAD(ES)", "ROLLO", "ROLLOS"],
    "LAVADO": ["LV", "LAVADO", "LAVADOS"],
    "DOCENA": ["DC", "DZ", "DOCENA", "DOCENAS"],
    "METRO": ["M", "METRO", "METROS", "M-(METRO)"],
    "GRAMO": ["G", "GR", "GRS", "GRAMO", "GRAMOS"],
    "MILILITRO": ["ML", "MILILITRO", "MILILITROS"],
    "CENTILITRO": ["CL", "CENTILITRO", "CENTILITROS"],
}
_ALIAS = {alias: unit for unit, aliases in UNIT_ALIASES.items() for alias in aliases}

# Smaller units → (factor, base unit)
_TO_BASE = {
    "GRAMO": (0.001, "KILO"),
    "MILILITRO": (0.001, "LITRO"),
    "CENTILITRO": (0.01, "LITRO"),
    "DOCENA": (12, "UNIDAD"),
}


def canonical_unit(raw):
    """'litros' → 'LITRO', 'g' → 'GRAMO'; unknown units are kept uppercased; blank → None."""
    if raw is None or not isinstance(raw, str) or not raw.strip():
        return None
    key = raw.strip().upper().rstrip(".")
    return _ALIAS.get(key, key)


def to_base(amount, unit):
    """(340, 'g') → (0.34, 'KILO'); (1, 'docena') → (12, 'UNIDAD')."""
    unit = canonical_unit(unit)
    if unit is None:
        return None
    factor, base = _TO_BASE.get(unit, (1, unit))
    return round(amount * factor, 6), base


def parse_unit_price_label(label):
    """The 'per what' of a price per unit: '100 GR' → (0.1, 'KILO'), '1 Lv' → (1, 'LAVADO')."""
    if label is None or not isinstance(label, str) or not label.strip():
        return None
    text = label.strip().lower()
    match = re.match(r"^(\d+(?:[.,]\d+)?)?[\s-]*(.+?)\.?$", text)
    if not match:
        return None
    amount = float(match.group(1).replace(",", ".")) if match.group(1) else 1.0
    return to_base(amount, match.group(2))


def per_base(price_per, label):
    """Convert a price per '<label>' into a price per base unit: (2.0, '100 GR') → (20.0, 'KILO')."""
    parsed = parse_unit_price_label(label)
    if parsed is None or price_per is None or parsed[0] == 0:
        return None
    amount, unit = parsed
    return round(price_per / amount, 6), unit
