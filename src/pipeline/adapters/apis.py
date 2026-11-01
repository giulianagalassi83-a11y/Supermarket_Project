"""Adapters for the API collectors: BM and Consum (same Aktios platform) and Mercadona's API."""

import numpy as np

from src.pipeline import config
from src.pipeline.adapters.common import finish, prices, read_raw
from src.pipeline.text import match_terms


def load_aktios(path, chain_id):
    df = read_raw(path)
    out = df.rename(columns={
        "searched_item": "search_term", "product_id": "native_id", "category": "chain_category",
    })
    regular, offer = prices(df["regular_price"]), prices(df["offer_price"])
    out["regular_price"], out["offer_price"] = regular, offer
    out["price"] = offer.fillna(regular)
    out["price_per_unit"] = prices(df["offer_unit_price"]).fillna(prices(df["regular_unit_price"]))
    out["ppu_label"] = df["unit_price_unit_type"]
    out["quantity_raw"] = df["description"]
    # The collector checks availability == 1, which never matches (False on every row) → unknown
    out["available"] = np.nan
    return finish(out, chain_id)


def load_mercadona_api(path):
    df = read_raw(path)
    out = df.rename(columns={"product_id": "native_id"})
    out["price"] = prices(df["unit_price"])
    out["regular_price"] = prices(df["previous_unit_price"]).fillna(out["price"])
    out["price_per_unit"] = prices(df["reference_price"])
    out["ppu_label"] = df["reference_format"]
    out["chain_category"] = df[["top_category", "category", "section"]].apply(
        lambda r: " > ".join(r.dropna()), axis=1
    )
    # The API has no search: every whole-word match of a search term is a hit
    out["search_term"] = df["name"].map(lambda n: match_terms(n, config.SEARCH_TERMS) or [np.nan])
    out = out.explode("search_term")
    return finish(out, "mercadona")


def load_bm(path):
    return load_aktios(path, "bm")


def load_consum(path):
    return load_aktios(path, "consum")
