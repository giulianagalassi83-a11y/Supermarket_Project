"""Basket optimizer: which stores to visit, and in what order, to buy a basket as cheaply as possible.

This is a small Travelling Purchaser Problem (TPP): pick a set of stores, buy every item where it is
cheapest within that set, and walk home -> stores -> home. The TPP is NP-hard in general, but here the
candidates are a handful of nearby stores (k per chain, at most one store per chain, at most 3 stops),
so trying every store set and every visiting order is exact and still instant.

Plans are enumerated once (prices and route lengths don't depend on the user's costs) and then scored
with `score()`, so changing €/km or the cost per stop, or running a sensitivity sweep, doesn't redo the
search.
"""

from itertools import combinations, permutations

import numpy as np
import pandas as pd

EARTH_KM = 6371.0
DETOUR = 1.3  # real streets are ~20-40 % longer than a straight line
HOME = "Home"


def haversine(lat1, lon1, lat2, lon2):
    """Straight-line km between points (works on scalars or arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def candidate_stores(stores, home, per_chain=3, max_km=None):
    """The `per_chain` nearest stores of each chain (km from home, straight line), closest first."""
    stores = stores.copy()
    stores["distance_km"] = haversine(home[0], home[1], stores["lat"], stores["lon"])
    if max_km is not None:
        stores = stores[stores["distance_km"] <= max_km]
    return (
        stores.sort_values("distance_km")
        .groupby("supermarket").head(per_chain)
        .reset_index(drop=True)
    )


def best_route(home, points, detour=DETOUR):
    """Shortest home -> every point -> home over all visiting orders.

    Returns (km, order, legs): `order` indexes `points`, `legs` holds the km of each leg in that order
    (first leg from home, last leg back home). Distances include the detour factor.
    """
    best = None
    for order in permutations(range(len(points))):
        path = [home, *(points[i] for i in order), home]
        legs = [detour * haversine(*path[i], *path[i + 1]) for i in range(len(path) - 1)]
        if best is None or sum(legs) < best[0]:
            best = (sum(legs), list(order), legs)
    return best


def enumerate_plans(cost, stores, home, max_stores=3, detour=DETOUR):
    """Every useful store set that covers the basket, with its product cost and its shortest route.

    cost: items x chains DataFrame of € for the wanted quantity of each item (NaN = chain lacks it).
    stores: candidate stores with supermarket, name, address, lat, lon (e.g. from candidate_stores).
    A set has at most one store per chain, and every store in it must be the cheapest place for at
    least one item (otherwise the same plan without it is cheaper).
    """
    stores = stores[stores["supermarket"].isin(cost.columns)].reset_index(drop=True)
    chains = list(cost.columns)
    matrix = cost.to_numpy(dtype=float)
    col = {chain: i for i, chain in enumerate(chains)}

    rows = []
    for k in range(1, max_stores + 1):
        for combo in combinations(range(len(stores)), k):
            combo_chains = [stores.at[i, "supermarket"] for i in combo]
            if len(set(combo_chains)) < k:
                continue
            sub = matrix[:, [col[c] for c in combo_chains]]
            if np.isnan(sub).all(axis=1).any():  # some item has no price in any of these stores
                continue
            pick = np.nanargmin(np.where(np.isnan(sub), np.inf, sub), axis=1)
            if len(set(pick)) < k:  # a store buys nothing
                continue
            points = [(stores.at[i, "lat"], stores.at[i, "lon"]) for i in combo]
            km, order, legs = best_route(home, points, detour)
            route = [combo[i] for i in order]
            rows.append({
                "store_ids": route,
                "chains": [stores.at[i, "supermarket"] for i in route],
                "n_stores": k,
                "products_eur": float(sub[np.arange(len(sub)), pick].sum()),
                "route_km": km,
                "legs_km": legs,
                "buy_at": dict(zip(cost.index, (combo_chains[j] for j in pick))),
            })
    return pd.DataFrame(rows), stores


def score(plans, cost_per_km, cost_per_stop=0.0):
    """Add travel, extra-stop and total cost; cheapest plan first. The first stop is free."""
    if plans.empty:
        return plans
    plans = plans.copy()
    plans["travel_eur"] = plans["route_km"] * cost_per_km
    plans["stops_eur"] = (plans["n_stores"] - 1) * cost_per_stop
    plans["total_eur"] = plans["products_eur"] + plans["travel_eur"] + plans["stops_eur"]
    return plans.sort_values(["total_eur", "n_stores"]).reset_index(drop=True)


def optimize(cost, stores, home, cost_per_km, cost_per_stop=0.0, max_stores=3, detour=DETOUR):
    """enumerate_plans + score in one call. Returns (ranked plans, stores the plan ids refer to)."""
    plans, stores = enumerate_plans(cost, stores, home, max_stores, detour)
    return score(plans, cost_per_km, cost_per_stop), stores


def describe_route(plan, stores):
    """[(from, to, km), ...] for a plan, with home as HOME."""
    names = [HOME, *(stores.at[i, "supermarket"] for i in plan["store_ids"]), HOME]
    return [(names[i], names[i + 1], km) for i, km in enumerate(plan["legs_km"])]


def greedy_plan(cost, stores, home, cost_per_km, cost_per_stop=0.0, max_stores=3, detour=DETOUR):
    """Baseline heuristic: buy each item at its cheapest chain, then visit the nearest store of each.

    Ignores travel when choosing where to buy, so it is what a shopper with a price list would do.
    With more than `max_stores` chains, it keeps those selling the most items and rebuys the rest there.
    Compare it with optimize() to show what the exact search adds. Returns one scored plan (a Series).
    """
    stores = stores[stores["supermarket"].isin(cost.columns)]
    cost = cost[[c for c in cost.columns if c in set(stores["supermarket"])]]
    if cost.isna().all(axis=1).any():
        return None
    buy_at = cost.idxmin(axis=1)
    if buy_at.nunique() > max_stores:
        cost = cost[list(buy_at.value_counts().index[:max_stores])]
        if cost.isna().all(axis=1).any():
            return None
        buy_at = cost.idxmin(axis=1)
    nearest = stores.sort_values("distance_km").drop_duplicates("supermarket").set_index("supermarket")
    chains = list(dict.fromkeys(buy_at))
    points = [(nearest.at[c, "lat"], nearest.at[c, "lon"]) for c in chains]
    km, order, legs = best_route(home, points, detour)
    plan = pd.DataFrame([{
        "chains": [chains[i] for i in order],
        "n_stores": len(chains),
        "products_eur": float(cost.min(axis=1).sum()),
        "route_km": km,
        "legs_km": legs,
        "buy_at": buy_at.to_dict(),
    }])
    return score(plan, cost_per_km, cost_per_stop).iloc[0]


def sensitivity(plans, km_costs, cost_per_stop=0.0):
    """For each €/km: the best single-store total, the best overall total and how many stores it uses.

    Shows from which travel cost splitting the basket stops paying off.
    """
    rows = []
    singles = plans[plans["n_stores"] == 1]
    for c in km_costs:
        best = score(plans, c, cost_per_stop).iloc[0]
        rows.append({
            "cost_per_km": c,
            "best_total": best["total_eur"],
            "best_n_stores": best["n_stores"],
            "single_total": score(singles, c).iloc[0]["total_eur"] if len(singles) else np.nan,
        })
    return pd.DataFrame(rows)
