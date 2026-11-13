
"""Adapters for API collectors: BM, Consum (Aktios), and Mercadona."""

import numpy as np

from src.pipeline import config
from src.pipeline.adapters.common import finish, prices, read_raw
from src.pipeline.text import match_terms


# --------------------------------------------------
# BM AND CONSUM (AKTIOS API)
# --------------------------------------------------

def load_aktios(path, chain_id):
    df = read_raw(path)

    out = df.rename(
        columns={
            "searched_item": "search_term",
            "product_id": "native_id",
            "category": "chain_category",
        }
    ).copy()

    # Read regular and promotional selling prices
    regular = prices(df["regular_price"])
    offer = prices(df["offer_price"])

    # A promotional price is usable when it is positive
    # and lower than the regular price, if known.
    valid_offer = (
        (offer > 0) &
        (regular.isna() | (offer < regular))
    )

    out["regular_price"] = regular
    out["offer_price"] = offer.where(valid_offer)

    # Use the promotional price when valid;
    # otherwise, use the regular selling price.
    out["price"] = out["offer_price"].fillna(
        regular.where(regular > 0)
    )

    # Read website-provided reference prices
    regular_ppu = prices(df["regular_unit_price"])
    offer_ppu = prices(df["offer_unit_price"])

    # Reference prices must correspond to the
    # selling price selected above.
    out["price_per_unit"] = regular_ppu.where(
        regular_ppu > 0
    )

    out.loc[valid_offer, "price_per_unit"] = (
        offer_ppu.where(offer_ppu > 0)
    ).loc[valid_offer]

    out["ppu_label"] = df["unit_price_unit_type"]

    # Preserve product descriptions for quantity parsing
    out["quantity_raw"] = df["description"]

    # Availability cannot be reliably inferred
    # from the collector's current field.
    out["available"] = np.nan

    return finish(out, chain_id)


# --------------------------------------------------
# MERCADONA API
# --------------------------------------------------

def load_mercadona_api(path):
    df = read_raw(path)

    out = df.rename(
        columns={"product_id": "native_id"}
    ).copy()

    # Mercadona's unit_price is the selling price
    # of the product, not necessarily its price per kg/L.
    current_price = prices(df["unit_price"])
    previous_price = prices(df["previous_unit_price"])

    out["price"] = current_price

    # Keep the previous price only when it represents
    # a valid higher historical selling price.
    valid_previous = (
        (previous_price > 0) &
        (current_price > 0) &
        (previous_price > current_price)
    )

    out["regular_price"] = previous_price.where(
        valid_previous,
        current_price
    )

    # Use the reference price and unit reported by the API
    out["price_per_unit"] = prices(
        df["reference_price"]
    )

    out["ppu_label"] = df["reference_format"]

    # Preserve the original category hierarchy
    out["chain_category"] = df[
        ["top_category", "category", "section"]
    ].apply(
        lambda r: " > ".join(
            r.dropna().astype(str)
        ),
        axis=1
    )

    # Mercadona API data is classified into search
    # terms through whole-word matching.
    out["search_term"] = df["name"].map(
        lambda n: match_terms(
            n,
            config.SEARCH_TERMS
        ) or [np.nan]
    )

    out = out.explode("search_term")

    return finish(out, "mercadona")


# --------------------------------------------------
# CHAIN-SPECIFIC ENTRY POINTS
# --------------------------------------------------

def load_bm(path):
    return load_aktios(path, "bm")


def load_consum(path):
    return load_aktios(path, "consum")
