"""Emergent categories (spec §6.4): UMAP + HDBSCAN over every product vector, named by the LLM judge.

A discovery and validation tool for the report. It never drives dashboard prices.

    python -m src.similarity.cluster
        → results/clusters.csv           cluster_id, name, size, top hand type, share, no_hand_type
          results/cluster_agreement.csv  ARI / NMI vs the hand taxonomy, per encoder
          results/untyped_suggestions.csv  untyped Mercadona rows → majority hand type of their cluster (share ≥ .6)
        cluster names cached in data/similarity/cluster_names.csv (nothing is sent twice)
"""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.pipeline import config

NAMES_PATH = config.SIM_DIR / "cluster_names.csv"
MIN_SHARE = 0.6

NAME_PROMPT = """These products from Spanish supermarkets were grouped together by similarity:

{titles}

Give the group a short Spanish name (2–5 words) for the kind of product they share, as a shopper would say it \
(e.g. "leche semidesnatada", "comida húmeda para gatos"). One-sentence rationale."""
NAME_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "rationale": {"type": "string"}},
               "required": ["name", "rationale"]}


def cluster(vectors, seed=config.SEED, min_cluster_size=15):
    """Cluster id per row (−1 = noise): UMAP to 10 dims (cosine), then HDBSCAN."""
    import umap
    from sklearn.cluster import HDBSCAN

    reduced = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine",
                        random_state=seed).fit_transform(vectors)
    return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(reduced)


def agreement(labels, type_ids):
    """ARI / NMI between clusters and hand types, over typed rows that are not noise."""
    labels = np.asarray(labels)
    types = pd.Series(type_ids).reset_index(drop=True)
    keep = (labels != -1) & types.notna().to_numpy()
    a, t = labels[keep], types[keep].to_numpy()
    return {"n": int(keep.sum()), "ari": round(float(adjusted_rand_score(t, a)), 4),
            "nmi": round(float(normalized_mutual_info_score(t, a)), 4)}


def cluster_table(products, labels):
    """One row per cluster: size, typed members, top hand type and its share among typed members."""
    d = products.assign(cluster_id=np.asarray(labels))
    d = d[d["cluster_id"] != -1]
    rows = []
    for cid, g in d.groupby("cluster_id"):
        typed = g["type_id"].dropna()
        top = typed.value_counts()
        rows.append({"cluster_id": int(cid), "size": len(g), "n_typed": len(typed),
                     "top_type_id": top.index[0] if len(top) else None,
                     "share": round(top.iloc[0] / len(typed), 3) if len(top) else 0.0,
                     "chains": g["chain_id"].nunique()})
    t = pd.DataFrame(rows)
    t["no_hand_type"] = t["top_type_id"].isna() | (t["top_type_id"] == config.NO_RELEVANTE)
    return t


def suggest_types(products, labels, min_share=MIN_SHARE, chain="mercadona"):
    """Untyped rows of `chain` → the majority hand type of their cluster when it holds ≥ min_share of typed members."""
    t = cluster_table(products, labels)
    t = t[(t["share"] >= min_share) & t["top_type_id"].notna()]
    d = products.assign(cluster_id=np.asarray(labels))
    d = d[(d["chain_id"] == chain) & d["type_id"].isna()]
    out = d.merge(t[["cluster_id", "top_type_id", "share"]], on="cluster_id")
    return out.rename(columns={"top_type_id": "suggested_type_id"})[
        ["product_key", "cluster_id", "suggested_type_id", "share"] + [c for c in ["name"] if c in out]]


def _sample_key(keys):
    return hashlib.sha1("|".join(sorted(keys)).encode("utf-8")).hexdigest()[:16]


def _ask_name(client, titles):
    from google.genai import types

    try:
        response = client.models.generate_content(
            model=config.SIM["judge_model"], contents=NAME_PROMPT.format(titles="\n".join(f"- {t}" for t in titles)),
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_json_schema=NAME_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_level="low"), temperature=0))
        reply = json.loads(response.text or "")
        return reply.get("name"), reply.get("rationale", "")
    except Exception as exc:  # leave unnamed; the next run retries it
        return None, f"error: {type(exc).__name__}: {str(exc)[:120]}"


def name_clusters(products, labels, client=None, per_cluster=10, seed=config.SEED, cache_path=NAMES_PATH):
    """cluster_id, name, rationale; the name is asked once per distinct sample of titles (cached)."""
    from concurrent.futures import ThreadPoolExecutor

    from src.similarity.judge import _client

    d = products.assign(cluster_id=np.asarray(labels))
    d = d[d["cluster_id"] != -1]
    samples = {int(cid): g.sample(min(per_cluster, len(g)), random_state=seed) for cid, g in d.groupby("cluster_id")}
    todo = pd.DataFrame({"cluster_id": list(samples),
                         "sample_key": [_sample_key(s["product_key"]) for s in samples.values()]})
    cache = (pd.read_csv(cache_path, dtype=str) if cache_path.exists()
             else pd.DataFrame(columns=["sample_key", "name", "rationale", "model"]))
    cache = cache[cache["name"].notna()]
    pending = todo[~todo["sample_key"].isin(cache["sample_key"])]
    if len(pending):
        client = _client(client)
        with ThreadPoolExecutor(max_workers=8) as pool:
            answers = list(pool.map(lambda c: _ask_name(client, list(samples[c]["name"])), pending["cluster_id"]))
        new = pending[["sample_key"]].assign(name=[a[0] for a in answers], rationale=[a[1] for a in answers],
                                             model=config.SIM["judge_model"])
        cache = pd.concat([cache, new], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(cache_path, index=False)
    return todo.merge(cache[["sample_key", "name", "rationale"]], on="sample_key", how="left").drop(columns="sample_key")


def _vectors(which, texts):
    import gc

    from src.similarity.encoder import load_encoder
    from src.similarity.silver import encode_gently

    enc = load_encoder(which)
    encode_gently(enc, texts)
    vectors = np.asarray(enc.encode(texts, "query"), dtype=np.float32)
    del enc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    return vectors


def main(encoders=("base", "finetuned")):
    import time

    from src.similarity.classify import load_all_products

    t0 = time.time()
    products, _ = load_all_products()
    saved = pd.read_csv(config.SIM_ASSIGNMENTS_PATH, dtype=str)[["product_key", "type_id"]]
    products = products.merge(saved, on="product_key", how="left")
    texts = list(products["text"])

    rows, labelings = [], {}
    for which in encoders:
        labels = cluster(_vectors(which, texts))
        labelings[which] = labels
        real = products["type_id"].where(products["type_id"] != config.NO_RELEVANTE)  # no_relevante = one mixed bag
        rows.append({"encoder": which, "clusters": int(labels.max() + 1), "noise_share": round(float((labels == -1).mean()), 3),
                     **agreement(labels, products["type_id"]),
                     **{f"{k}_real_types": v for k, v in agreement(labels, real).items()}})
        print(rows[-1], f"{time.time() - t0:.0f} s", flush=True)
    agree = pd.DataFrame(rows)
    pick = agree.loc[agree["nmi"].idxmax(), "encoder"]
    agree["picked"] = agree["encoder"] == pick
    labels = labelings[pick]

    table = cluster_table(products, labels)
    names = name_clusters(products, labels)
    table = names[["cluster_id", "name"]].merge(table, on="cluster_id").sort_values("size", ascending=False)
    sugg = suggest_types(products, labels)

    config.SIM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agree.to_csv(config.SIM_RESULTS_DIR / "cluster_agreement.csv", index=False)
    table.assign(encoder=pick).to_csv(config.SIM_RESULTS_DIR / "clusters.csv", index=False)
    sugg.to_csv(config.SIM_RESULTS_DIR / "untyped_suggestions.csv", index=False)
    products[["product_key"]].assign(cluster_id=labels).to_csv(config.SIM_DIR / "cluster_members.csv", index=False)
    untyped = int(((products["chain_id"] == "mercadona") & products["type_id"].isna()).sum())
    print(agree.to_string(index=False))
    print(f"\npicked {pick}: {len(table)} clusters, {int(table['no_hand_type'].sum())} with no hand type; "
          f"{len(sugg)} of {untyped} untyped Mercadona rows get a suggestion — {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
