"""Adapters for the Selenium-scraped chains: AhorraMás, Alcampo, DIA, ALDI, Mercadona (Selenium)."""

import numpy as np
import pandas as pd

from src.pipeline.adapters.common import finish, hash_id, prices, read_raw
from src.pipeline.text import parse_pack
from src.pipeline.units import canonical_unit, per_base


def _site_ppu(df):
    """Keep the collector's €/unit only when the site showed one (price_per_unit_raw not blank)."""
    shown = df["price_per_unit_raw"].notna()
    return prices(df["price_per_unit"]).where(shown), df["unit"].where(shown)


def load_ahorramas(path):
    df = read_raw(path)
    out = df.rename(columns={"searched_item": "search_term"})
    # product_id is always empty in the collector (regex bug) → take it from the URL
    out["native_id"] = df["url"].str.extract(r"-(\d+)\.html$")[0]
    # The collector's `price` is the shelf's reference price (€/kg, €/l, €/roll, €/wash, eggs €/dozen),
    # not the item price: "Bonito 80g" = 46.88 €/kg, "12 rollos" = 0.38 €/roll, "Huevos 6u" = 5.40 €/dozen.
    packs = df["name"].map(parse_pack)
    out["price_per_unit"] = prices(df["price"])
    out["ppu_label"] = [
        _ahorramas_label(pack, canonical_unit(unit), str(term).lower())
        for pack, unit, term in zip(packs, df["unit"], df["searched_item"])
    ]
    # Item price = € per base unit × size, when both are in the same unit
    item_prices = []
    for ppu, label, pack in zip(out["price_per_unit"], out["ppu_label"], packs):
        converted = per_base(ppu, label) if pd.notna(ppu) and isinstance(label, str) else None
        same_unit = converted and pack and converted[1] == pack[2]
        item_prices.append(round(converted[0] * pack[0] * pack[1], 2) if same_unit else np.nan)
    out["price"] = item_prices
    return finish(out, "ahorramas")


def _ahorramas_label(pack, collector_unit, term):
    """Unit of AhorraMás' reference price: kg/l from the collector, else the size in the name (eggs: dozen)."""
    if collector_unit in ("KILO", "LITRO"):
        return collector_unit
    if pack:
        return "DOCENA" if pack[2] == "UNIDAD" and term == "huevos" else pack[2]
    return collector_unit


def load_alcampo(path):
    df = read_raw(path)
    out = df.rename(columns={"searched_item": "search_term", "product_id": "native_id"})
    out["price"] = prices(df["price"])
    out["price_per_unit"], out["ppu_label"] = _site_ppu(df)
    return finish(out, "alcampo")


def load_dia(path):
    df = read_raw(path)
    out = df.rename(columns={"searched_item": "search_term"})
    url = df["url"].where(df["url"] != "about:blank")
    out["url"] = url.where(~url.str.startswith("/", na=False), "https://www.dia.es" + url)
    out["native_id"] = df["product_id"].fillna(
        df.apply(lambda r: hash_id(r["name"], r.get("quantity_raw")), axis=1)
    )
    out["price"] = prices(df["price"])
    out["price_per_unit"], out["ppu_label"] = _site_ppu(df)
    return finish(out, "dia")


def load_aldi(path):
    df = read_raw(path)
    out = df.rename(columns={"searched_item": "search_term", "product_id": "native_id"})
    out["price"] = prices(df["price"])
    # "kg = 2.71" → label "kg", value 2.71; "= 24.75" has no label → no usable €/unit
    parts = df["price_per_unit_raw"].str.extract(r"^\s*(?P<label>[^=]*?)\s*=\s*(?P<value>[\d.,]+)")
    has_label = parts["label"].fillna("").str.strip() != ""
    # ALDI writes "g = 12.45" for prices that are per kg (2.49 € / 200 g = 12.45 €/kg)
    out["ppu_label"] = parts["label"].where(has_label).replace({"g": "kg"})
    out["price_per_unit"] = prices(parts["value"]).where(has_label, np.nan)
    return finish(out, "aldi")


def load_mercadona_selenium(path):
    df = read_raw(path)
    out = df.rename(columns={"searched_item": "search_term"})
    # No product ID and no €/unit in this collector: ID from name + size, €/unit computed later
    out["native_id"] = [hash_id(n, q) for n, q in zip(df["name"], df["quantity_raw"])]
    out["price"] = prices(df["price"])
    return finish(out, "mercadona")
