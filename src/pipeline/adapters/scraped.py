
"""Adapters for Selenium-scraped supermarkets: AhorraMás, Alcampo, DIA, and ALDI."""

import numpy as np
import pandas as pd

from src.pipeline.adapters.common import finish, hash_id, prices, read_raw


# --------------------------------------------------
# COMMON HELPERS
# --------------------------------------------------

def _site_ppu(df):
    """
    Use the price per unit only when it was explicitly
    provided by the supermarket website.
    """
    shown = df["price_per_unit_raw"].notna()

    return (
        prices(df["price_per_unit"]).where(shown),
        df["unit"].where(shown)
    )


# --------------------------------------------------
# AHORRAMÁS
# --------------------------------------------------

def load_ahorramas(path):
    df = read_raw(path)

    out = df.rename(
        columns={"searched_item": "search_term"}
    ).copy()

    # Extract the product ID from the product URL
    url_ids = df["url"].str.extract(r"-(\d+)\.html(?:\?.*)?$")[0]

    # Prefer the original product ID when available
    if "product_id" in df.columns:
        out["native_id"] = df["product_id"].replace("", np.nan)
        out["native_id"] = out["native_id"].fillna(url_ids)
    else:
        out["native_id"] = url_ids

    # Generate a fallback ID when no identifier is available
    missing_id = out["native_id"].isna()

    if missing_id.any():
        out.loc[missing_id, "native_id"] = [
            hash_id(name, url)
            for name, url in zip(
                df.loc[missing_id, "name"],
                df.loc[missing_id, "url"]
            )
        ]

    # The scraper provides the actual selling price
    out["price"] = prices(df["price"])

    # Use website-displayed reference prices only when available
    out["price_per_unit"], out["ppu_label"] = _site_ppu(df)

    return finish(out, "ahorramas")


# --------------------------------------------------
# ALCAMPO
# --------------------------------------------------

def load_alcampo(path):
    df = read_raw(path)

    out = df.rename(
        columns={
            "searched_item": "search_term",
            "product_id": "native_id"
        }
    ).copy()

    # Extract the actual selling price
    out["price"] = prices(df["price"])

    # Keep reference prices explicitly provided by the website
    out["price_per_unit"], out["ppu_label"] = _site_ppu(df)

    return finish(out, "alcampo")


# --------------------------------------------------
# DIA
# --------------------------------------------------

def load_dia(path):
    df = read_raw(path)

    out = df.rename(
        columns={"searched_item": "search_term"}
    ).copy()

    # Convert relative product URLs into absolute URLs
    url = df["url"].where(df["url"] != "about:blank")

    out["url"] = url.where(
        ~url.str.startswith("/", na=False),
        "https://www.dia.es" + url
    )

    # Use the original product ID or generate a fallback
    out["native_id"] = df["product_id"].fillna(
        df.apply(
            lambda r: hash_id(
                r["name"],
                r.get("quantity_raw")
            ),
            axis=1
        )
    )

    # Extract the actual selling price
    out["price"] = prices(df["price"])

    # Keep reference prices explicitly provided by the website
    out["price_per_unit"], out["ppu_label"] = _site_ppu(df)

    return finish(out, "dia")


# --------------------------------------------------
# ALDI
# --------------------------------------------------

def load_aldi(path):
    df = read_raw(path)

    out = df.rename(
        columns={
            "searched_item": "search_term",
            "product_id": "native_id"
        }
    ).copy()

    # Extract the actual selling price
    out["price"] = prices(df["price"])

    # Extract reference price and unit from the original text
    # Example: "kg = 2.71" -> unit "kg", price 2.71
    parts = df["price_per_unit_raw"].str.extract(
        r"^\s*(?P<label>[^=]*?)\s*=\s*(?P<value>[\d.,]+)"
    )

    has_label = (
        parts["label"].fillna("").str.strip() != ""
    )

    # ALDI sometimes uses "g" for prices expressed per kg
    out["ppu_label"] = (
        parts["label"]
        .where(has_label)
        .replace({"g": "kg"})
    )

    out["price_per_unit"] = (
        prices(parts["value"])
        .where(has_label, np.nan)
    )

    return finish(out, "aldi")
