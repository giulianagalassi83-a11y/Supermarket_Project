"""The numbers behind the dashboard's Analysis page, all computed over products_normalized.csv.

Every function filters with prices.usable() first. Prices compare the same product type across chains;
each result carries product counts so a page can show how much data sits behind it.
"""

import numpy as np
import pandas as pd

from src.analysis import optimizer as opt
from src.analysis import prices
from src.pipeline import config

MIN_PRODUCTS = 5  # below this, a chain's price for a category is shown as "too little data"
MIN_CHAINS = 4  # a type must be sold by this many chains to count in the category comparison

CATEGORY_NAMES = {
    "leche": "Milk", "huevos": "Eggs", "manzana": "Apples", "plátano": "Bananas", "tomate": "Tomatoes",
    "arroz": "Rice", "pasta": "Pasta", "pan": "Bread", "pollo": "Chicken", "carne picada": "Minced meat",
    "salmón": "Salmon", "yogur": "Yoghurt", "queso": "Cheese", "agua": "Water", "zumo": "Juice",
    "atún": "Tuna", "garbanzos": "Chickpeas", "aceite": "Oil", "detergente": "Detergent",
    "papel higiénico": "Toilet paper",
}


def category_name(term):
    return CATEGORY_NAMES.get(term, term)


def coverage(df):
    """Long table: comparable products per category and chain, zeros included."""
    counts = prices.usable(df).groupby(["searched_item", "supermarket"]).size().unstack(fill_value=0)
    return counts.rename_axis(index="category", columns="chain").stack().rename("products").reset_index()


def basket_index(df):
    """Index per chain on the types every chain sells (Mercadona = 100), by median and by cheapest product."""
    usable = prices.usable(df)
    groups = usable.groupby(["product_type", "supermarket"])["price_per_unit"]
    median, cheapest = groups.median().unstack(), groups.min().unstack()
    common = median.dropna().index
    index = lambda table: 100 * table.loc[common].sum() / table.loc[common, "Mercadona"].sum()
    out = pd.DataFrame({"median": index(median), "cheapest": index(cheapest)}).rename_axis("chain").reset_index()
    return out.sort_values("median", ascending=False).reset_index(drop=True), len(common)


def category_prices(df, stat="median"):
    """Long table per category and chain: price against the cheapest chain (1.0 = cheapest).

    Each type sold by at least MIN_CHAINS chains is compared on its own, then a chain's types are
    averaged with a median. `low_data` marks chains with fewer than MIN_PRODUCTS products there, which
    can't win the category.
    """
    usable = prices.usable(df)
    rows = []
    for category, group in usable.groupby("searched_item"):
        table = group.groupby(["product_type", "supermarket"])["price_per_unit"].agg(prices.STATS[stat]).unstack()
        table = table[table.notna().sum(axis=1) >= MIN_CHAINS]
        if table.empty:
            continue
        relative = table.div(table.min(axis=1), axis=0).median()
        counts = group[group["product_type"].isin(table.index)].groupby("supermarket").size()
        for chain, value in relative.dropna().items():
            rows.append({"category": category, "chain": chain, "relative": value, "products": int(counts.get(chain, 0))})
    out = pd.DataFrame(rows)
    out["low_data"] = out["products"] < MIN_PRODUCTS
    return out


def category_winners(df):
    """One row per category: the cheapest chain with enough data, by median and by cheapest product."""
    winners = {}
    for stat in ("median", "min"):
        table = category_prices(df, stat)
        table = table[~table["low_data"]]
        winners[stat] = table.loc[table.groupby("category")["relative"].idxmin()].set_index("category")["chain"]
    out = pd.DataFrame(winners).dropna().rename_axis("category").reset_index()
    out["agree"] = out["median"] == out["min"]
    return out


def own_brand_gap(df):
    """Per chain: how much cheaper own brand is than national brands within the same type.

    Only types where the chain sells both count; the result is the median across those types.
    """
    usable = prices.usable(df)
    table = usable.groupby(["supermarket", "product_type", "is_private_label"])["price_per_unit"].median().unstack()
    table = table.dropna(subset=[0, 1])
    table["gap"] = 100 * (1 - table[1] / table[0])
    out = table.groupby("supermarket")["gap"].agg(gap_pct="median", types="size").rename_axis("chain").reset_index()
    share = usable.groupby("supermarket")["is_private_label"].mean().mul(100).rename("own_share_pct")
    return out.merge(share.rename_axis("chain").reset_index(), on="chain").sort_values("gap_pct", ascending=False)


def price_spread(df):
    """Per category: the spread between products of a chain against the spread between chains.

    Within a chain: p90 / p10 of a type's products, for chain-type pairs with at least MIN_PRODUCTS.
    Between chains: dearest / cheapest chain median of a type. Both are medians across the category.
    """
    usable = prices.usable(df)
    groups = usable.groupby(["searched_item", "product_type", "supermarket"])["price_per_unit"]
    stats = groups.agg(p10=lambda s: s.quantile(0.1), p90=lambda s: s.quantile(0.9), median="median", n="size").reset_index()
    enough = stats[stats["n"] >= MIN_PRODUCTS]
    within = (enough["p90"] / enough["p10"]).groupby(enough["searched_item"]).median()
    by_type = stats.groupby(["searched_item", "product_type"])["median"].agg(["max", "min", "size"])
    by_type = by_type[by_type["size"] >= MIN_CHAINS]
    between = (by_type["max"] / by_type["min"]).groupby("searched_item").median()
    out = pd.DataFrame({"within": within, "between": between}).dropna().rename_axis("category").reset_index()
    return out.sort_values("within", ascending=False).reset_index(drop=True)


def same_barcode(data_dir=config.DATA_DIR):
    """BM and Consum prices for products with the same barcode: the one exact comparison we can make today."""
    columns = ["ean", "name", "price"]
    read = lambda f: pd.read_csv(data_dir / "processed" / f, dtype={"ean": str}, encoding="utf-8")[columns]
    bm, consum = (read(f).dropna(subset=["ean"]).drop_duplicates("ean") for f in ("bm_products.csv", "consum_products.csv"))
    pairs = bm.merge(consum, on="ean", suffixes=("_bm", "_consum"))
    pairs = pairs[(pairs["price_bm"] > 0) & (pairs["price_consum"] > 0)].copy()
    pairs["gap_pct"] = 100 * (pairs["price_bm"] / pairs["price_consum"] - 1)
    return pairs


def type_gap(df, a="Bm", b="Consum"):
    """Dearer-by % of chain a over chain b on the types every chain sells, using type medians."""
    table = prices.usable(df).groupby(["product_type", "supermarket"])["price_per_unit"].median().unstack().dropna()
    return 100 * (table[a].sum() / table[b].sum() - 1)


def greedy_vs_exact(df, stores, home, cost_per_km, cost_per_stop, baskets=30, items=5, seed=config.SEED):
    """Random baskets near `home`: how much the exact optimizer saves over shopping from a price list.

    Baskets take `items` types sold by at least five chains, one unit of each, at median prices.
    """
    usable = prices.usable(df)
    chains_per_type = usable.groupby("product_type")["supermarket"].nunique()
    pool = sorted(chains_per_type[chains_per_type >= 5].index)
    table = prices.price_table(df, pool)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(baskets):
        cost = table.loc[list(rng.choice(pool, size=items, replace=False))]
        cost = cost[[c for c in cost.columns if c in set(stores["supermarket"])]]
        exact, _ = opt.optimize(cost, stores, home, cost_per_km, cost_per_stop)
        greedy = opt.greedy_plan(cost, stores, home, cost_per_km, cost_per_stop)
        if exact.empty or greedy is None:
            continue
        best = exact.iloc[0]
        rows.append({"exact_eur": best["total_eur"], "greedy_eur": greedy["total_eur"], "exact_stops": best["n_stores"],
                     "greedy_stops": greedy["n_stores"], "saving_eur": greedy["total_eur"] - best["total_eur"]})
    return pd.DataFrame(rows)
