"""Task 9 Step 5: silver labels from the Gemini judge for the chains with no barcode gold.

    python -m src.similarity.silver            # candidates + judge ~300 pairs and ~300 products
    python -m src.similarity.silver --dry-run  # candidates only, no API calls

Pairs come from three sources (column `source`):
- other_chains: best candidate of random queries from the 5 non-barcode chains, stratified by decider probability
- barcode_error: BM→Consum held-out matches accepted at the balanced point (P≥.90 rule) whose barcode disagrees (the precision
  lower bound: is the engine wrong, or are these the same product under another EAN?)
The decider is fit on held-out (fold-model) BM↔Consum candidates, not on the final encoder's own training pairs,
so its cosine scale matches unseen products like the other chains'.
"""

import argparse
import time

import numpy as np
import pandas as pd

from src.pipeline import config
from src.similarity import attributes, judge
from src.similarity.evaluate import _VARIANT_FEATURES, _oof_candidates, grouped_folds
from src.similarity.match import add_features, apply_veto, best_per_query, candidates, choose_threshold, fit_decider, predict

HIGH_T = config.SIM["high_confidence_t"]  # balanced point = the precision_target rule, fit on held-out probs
BANDS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.9), (0.9, 1.01)]  # equal share each, so the audit gets every region
CANDS_PATH = config.SIM_DIR / "silver_candidates.csv"
FEATURES = _VARIANT_FEATURES["finetuned+attrs"]


def judge_text(r):
    size = ""
    if pd.notna(r["unit_quantity"]) and r["unit_quantity"]:
        pack = f"{int(r['pack_count'])} × " if pd.notna(r["pack_count"]) and r["pack_count"] > 1 else ""
        size = f"{pack}{r['unit_quantity']:g} {r['quantity_unit'] or ''}".strip()
    return f"{r['name']} | brand: {r['brand'] or '-'} | size: {size or 'unknown'} | chain: {r['chain_id']}"


def heldout_decider(products, pairs, attrs):
    """Decider + high-confidence threshold from held-out BM↔Consum candidates; also the barcode-error pairs."""
    folds = grouped_folds(pairs)
    cands = _oof_candidates("finetuned", products, pairs, folds, config.SIM["top_k"])
    cands = add_features(apply_veto(cands, attrs, "same"), {}, attrs)
    errors, oof = [], np.zeros(len(cands))
    for i in sorted(cands["fold"].unique()):  # per-fold models → honest probabilities on every row
        test = (cands["fold"] == i).to_numpy()
        oof[test] = predict(fit_decider(cands[~test], cands.loc[~test, "y"], FEATURES), cands[test])
    cands["prob"] = oof
    top = cands.sort_values("prob", ascending=False, kind="stable").drop_duplicates("key_a")
    balanced_t = choose_threshold(top["prob"], top["y"], config.SIM["precision_target"])
    best = best_per_query(cands, balanced_t)
    errors = best[best["y"] == 0][["key_a", "key_b", "prob", "cosine"]].assign(source="barcode_error")
    return fit_decider(cands, cands["y"], FEATURES), balanced_t, errors


def final_encoder():
    from sentence_transformers import SentenceTransformer

    from src.similarity.encoder import Encoder, folder_revision

    path = config.SIM["finetuned_path"]
    return Encoder(str(path), folder_revision(path), model=SentenceTransformer(str(path), device="cuda"),
                   prefix_id=config.SIM["encoder_id"])


def encode_gently(enc, texts, chunk=2048):
    """Fill the cache in chunks with a pause and a temperature guard between them (laptop GPU)."""
    from src.similarity.train import COOL_C, HOT_C, cool_down, gpu_temp

    todo = list(dict.fromkeys(texts))
    for s in range(0, len(todo), chunk):
        enc.encode(todo[s:s + chunk], "query")
        if (t := gpu_temp()) is not None and t >= HOT_C:
            cool_down(COOL_C)
        time.sleep(1)
    enc.save_cache()


def other_chain_candidates(products, attrs, model, enc, per_pair, seed):
    """Best equivalent-level candidate for `per_pair` random queries of every (other chain → any chain) pair."""
    rng = np.random.default_rng(seed)
    chains = sorted(products["chain_id"].unique())
    frames = []
    for a in [c for c in chains if c not in ("bm", "consum")]:
        pool = products[products["chain_id"] == a]
        for b in [c for c in chains if c != a]:
            q = pool.iloc[rng.choice(len(pool), size=min(per_pair, len(pool)), replace=False)]
            c = candidates(q, products[products["chain_id"] == b], enc)
            c = add_features(apply_veto(c, attrs, "equivalent"), {}, attrs)
            c["prob"] = predict(model, c)
            frames.append(best_per_query(c, -np.inf)[["key_a", "key_b", "prob", "cosine"]].assign(chain_b=b))
    out = pd.concat(frames, ignore_index=True)
    out["chain_a"] = out["key_a"].map(products.set_index("product_key")["chain_id"])
    return out


def stratified(cands, n, seed):
    """n rows, an equal share per probability band, round-robin over (chain_a, chain_b) inside each band."""
    d = cands.sample(frac=1, random_state=seed)
    d["band"] = pd.cut(d["prob"], [lo for lo, _ in BANDS] + [BANDS[-1][1]], right=False, labels=False)
    d["turn"] = d.groupby(["band", "chain_a", "chain_b"]).cumcount()
    d = d.sort_values("turn", kind="stable")
    per = n // len(BANDS)
    return pd.concat([d[d["band"] == i].head(per) for i in range(len(BANDS))]).drop(columns="turn")


def type_sample(products, n, seed):
    """n products for threshold tuning, spread over chains, never one of the hand-labelled test products."""
    hand = set(pd.read_csv(config.LABELS_PATH, dtype=str)["product_key"]) if config.LABELS_PATH.exists() else set()
    d = products[products["search_term"].notna() & ~products["product_key"].isin(hand)].sample(frac=1, random_state=seed)
    d["turn"] = d.groupby("chain_id").cumcount()
    return d.sort_values("turn", kind="stable").head(n)


def main(n_pairs=300, n_products=300, n_errors=40, per_pair=150, dry_run=False, seed=config.SEED):
    import sqlite3

    from src.pipeline.classify import load_taxonomy
    from src.similarity.data import barcode_pairs, load_products

    products = load_products()
    pairs = barcode_pairs(products)
    attrs = attributes.extract_all(products)
    model, balanced_t, errors = heldout_decider(products, pairs, attrs)
    print(f"decider: high-confidence {HIGH_T}, balanced (P≥{config.SIM['precision_target']} rule) {balanced_t:.3f}; "
          f"barcode errors at balanced point: {len(errors)}", flush=True)

    enc = final_encoder()
    encode_gently(enc, list(products["mtext"]))
    other = other_chain_candidates(products, attrs, model, enc, per_pair, seed)
    del enc  # free the GPU
    pick = stratified(other, n_pairs, seed).assign(source="other_chains")
    errs = errors.sample(min(n_errors, len(errors)), random_state=seed)
    errs = errs.assign(chain_a="bm", chain_b="consum", band=np.nan)
    silver = pd.concat([pick, errs], ignore_index=True)
    by_key = products.set_index("product_key")
    silver["tier"] = [attributes.equivalence_tier(attrs[a], attrs[b]) for a, b in zip(silver["key_a"], silver["key_b"])]
    silver["accepted"] = np.where(silver["prob"] >= HIGH_T, "high", np.where(silver["prob"] >= balanced_t, "balanced", "no"))
    silver["name_a"], silver["name_b"] = by_key.loc[silver["key_a"], "name"].to_numpy(), by_key.loc[silver["key_b"], "name"].to_numpy()
    CANDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    silver.to_csv(CANDS_PATH, index=False)
    print(silver.groupby(["source", "accepted"]).size().to_string(), flush=True)
    print(silver.groupby(["chain_a"]).size().to_string(), flush=True)
    if dry_run:
        return

    texts = {k: judge_text(r) for k, r in by_key.loc[pd.unique(pd.concat([silver["key_a"], silver["key_b"]]))].iterrows()}
    labelled = judge.judge_pairs(silver[["key_a", "key_b"]], texts)
    print("pair labels:", labelled["label"].value_counts(dropna=False).to_dict(), flush=True)

    with sqlite3.connect(config.DB_PATH) as con:
        hits = pd.read_sql_query("SELECT product_key, search_term FROM search_hits", con)
    types = judge.judge_types(type_sample(products, n_products, seed), hits, load_taxonomy(config.TAXONOMY_PATH))
    print("type labels:", types["type_id"].notna().sum(), "of", len(types), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--products", type=int, default=300)
    ap.add_argument("--per-pair", type=int, default=150, help="random queries per (chain_a → chain_b)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.pairs, a.products, per_pair=a.per_pair, dry_run=a.dry_run)
