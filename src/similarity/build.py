"""The laptop-only similarity build: classification + cross-chain matches + canonical products.

    python -m src.similarity.build

Writes (committed, read by src.pipeline.build / the dashboard without any ML package):
    data/similarity/assignments.csv         one type per product (keyed by product_key + text_hash)
    data/similarity/matches.csv             cross-chain pairs: level same / equivalent (+ equiv_tier, confidence)
    data/similarity/canonical_products.csv  groups of 'same' matches, at most one product per chain

Matching, per unordered chain pair (recomputed from scratch every run):
  1. top-k candidates both ways (fine-tuned encoder, same type only), veto at 'equivalent', decider probability
  2. 'same': rows that also pass the 'same' veto, best per query ≥ balanced point, mutual best
  3. 'equivalent': only products with no 'same' match, best per query ≥ balanced point, mutual best
"""

import gc
import time

import numpy as np
import pandas as pd

from src.pipeline import config
from src.pipeline.classify import TfidfFallback, _hash, load_taxonomy
from src.similarity import attributes, equivalent
from src.similarity.match import add_features, apply_veto, best_per_query, candidates, group, mutual_best, predict

ASSIGNMENT_COLUMNS = ["product_key", "text_hash", "type_id", "score", "runner_up_type_id", "margin",
                      "low_confidence", "method", "model_name"]
MATCH_COLUMNS = ["product_key_a", "product_key_b", "chain_a", "chain_b", "level", "equiv_tier", "confidence",
                 "score", "model_revision"]


def pick_matches(ab, ba, threshold, equiv_threshold=None):
    """'same' first (rows with same_ok), then 'equivalent' among products left unmatched; both mutual-best.

    ab: candidates from chain A queries (key_a in A), ba: from chain B queries; columns key_a, key_b, prob, same_ok.
    With equiv_threshold, the 'equivalent' step ranks by the equivalent-level decider (column prob_equiv) instead
    of the same-product one; the output `prob` is always the score of the level picked.
    """
    same = mutual_best(best_per_query(ab[ab["same_ok"]], threshold), best_per_query(ba[ba["same_ok"]], threshold))
    taken = set(same["key_a"]) | set(same["key_b"])
    ab2 = ab[~ab["key_a"].isin(taken) & ~ab["key_b"].isin(taken)]
    ba2 = ba[~ba["key_a"].isin(taken) & ~ba["key_b"].isin(taken)]
    if equiv_threshold is not None:
        ab2, ba2 = (d.drop(columns="prob").rename(columns={"prob_equiv": "prob"}) for d in (ab2, ba2))
        threshold = equiv_threshold
    equiv = mutual_best(best_per_query(ab2, threshold), best_per_query(ba2, threshold))
    same, equiv = (d.drop(columns="prob_equiv", errors="ignore") for d in (same, equiv))
    return pd.concat([same.assign(level="same"), equiv.assign(level="equivalent")], ignore_index=True)


def _scored(queries, targets, encoder, attrs, decider, equiv_decider=None):
    c = candidates(queries, targets, encoder)
    c = add_features(apply_veto(c, attrs, "equivalent"), {}, attrs)
    c["prob"] = predict(decider, c)
    if equiv_decider is not None:
        c["prob_equiv"] = predict(equiv_decider, c)
    c["same_ok"] = [attributes.conflict(attrs[a], attrs[b], "same") is None for a, b in zip(c["key_a"], c["key_b"])]
    return c


def match_all(products, encoder, attrs, decider, threshold, equiv_decider=None, equiv_threshold=None):
    """Matches for every chain pair; products need product_key, chain_id, mtext, type_id (real types only).

    With equiv_decider, the 'equivalent' level is ranked and thresholded by it (src.similarity.equivalent).
    """
    chains = sorted(products["chain_id"].unique())
    frames = []
    for i, a in enumerate(chains):
        for b in chains[i + 1:]:
            pa, pb = products[products["chain_id"] == a], products[products["chain_id"] == b]
            m = pick_matches(_scored(pa, pb, encoder, attrs, decider, equiv_decider),
                             _scored(pb, pa, encoder, attrs, decider, equiv_decider), threshold, equiv_threshold)
            print(f"  {a:>9} ↔ {b:<9} same {int((m['level'] == 'same').sum()):>5}  "
                  f"equivalent {int((m['level'] == 'equivalent').sum()):>5}", flush=True)
            frames.append(m.assign(chain_a=a, chain_b=b))
    m = pd.concat(frames, ignore_index=True)
    m["equiv_tier"] = [attributes.equivalence_tier(attrs[a], attrs[b]) if lvl == "equivalent" else None
                       for a, b, lvl in zip(m["key_a"], m["key_b"], m["level"])]
    m["confidence"] = np.where(m["prob"] >= config.SIM["high_confidence_t"], "high", "balanced")
    m["model_revision"] = encoder.key
    return m.rename(columns={"key_a": "product_key_a", "key_b": "product_key_b", "prob": "score"})[MATCH_COLUMNS]


def _free_gpu():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def classify_all(products, hits, taxonomy):
    """Assignments for every product with the configured classifier (config.SIM['classifier'])."""
    from src.similarity.classify import EngineVectorizer
    from src.similarity.encoder import load_encoder
    from src.similarity.silver import encode_gently
    from src.pipeline.classify import decide

    which = config.SIM["classifier"]
    thresholds = config.SIM["classifier_thresholds"][which]
    if which == "tfidf":
        vectorizer = TfidfFallback(list(products["text"]) + list(taxonomy["description"]))
    else:
        enc = load_encoder(which)
        encode_gently(enc, list(products["text"]) + list(taxonomy["description"]))
        vectorizer = EngineVectorizer(enc)
    out = decide(products, hits, taxonomy, vectorizer, thresholds)
    out = out.merge(products[["product_key", "text"]], on="product_key")
    out["text_hash"] = [_hash(t) for t in out.pop("text")]
    print(f"classified {len(out)} products with {which} (τ={thresholds['tau']}, δ={thresholds['delta']}): "
          + ", ".join(f"{k} {v}" for k, v in out["method"].value_counts().items()), flush=True)
    return out[ASSIGNMENT_COLUMNS]


def main():
    from src.similarity.classify import load_all_products
    from src.similarity.data import barcode_pairs, load_products
    from src.similarity.encoder import load_encoder
    from src.similarity.silver import encode_gently, heldout_decider

    t0 = time.time()
    taxonomy = load_taxonomy(config.TAXONOMY_PATH)
    everything, hits = load_all_products()
    assignments = classify_all(everything, hits, taxonomy)
    _free_gpu()

    products = load_products().drop(columns="type_id")  # current products only are matched
    products = products.merge(assignments[["product_key", "type_id"]], on="product_key", how="left")
    attrs = attributes.extract_all(products)
    decider, balanced_t, _ = heldout_decider(products, barcode_pairs(products), attrs)
    print(f"decider: balanced (P≥{config.SIM['precision_target']} rule) {balanced_t:.3f}, "
          f"high {config.SIM['high_confidence_t']}", flush=True)
    equiv_decider, equiv_t = equivalent.fit(attrs)
    print(f"equivalent decider (Gemini silver, OOF P≥{config.SIM['equiv_precision_target']} rule): {equiv_t:.3f}", flush=True)

    typed = products[products["type_id"].notna() & (products["type_id"] != config.NO_RELEVANTE)].reset_index(drop=True)
    enc = load_encoder("finetuned")
    encode_gently(enc, list(products["mtext"]))  # every current product: dashboard search reads these from the cache
    print(f"matching {len(typed)} typed current products of {len(products)}", flush=True)
    matches = match_all(typed, enc, attrs, decider, balanced_t, equiv_decider, equiv_t)
    del enc
    _free_gpu()

    chain_of = typed.set_index("product_key")["chain_id"]
    same = matches[matches["level"] == "same"].rename(columns={"product_key_a": "key_a", "product_key_b": "key_b",
                                                                 "score": "prob"})
    canonical = group(same, chain_of.get)

    config.SIM_DIR.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(config.SIM_ASSIGNMENTS_PATH, index=False)
    matches.to_csv(config.SIM_MATCHES_PATH, index=False)
    canonical.to_csv(config.SIM_CANONICAL_PATH, index=False)
    print(f"\nwrote {len(assignments)} assignments, {len(matches)} matches "
          f"({matches['level'].value_counts().to_dict()}), {canonical['canonical_id'].nunique()} canonical products "
          f"covering {len(canonical)} products — {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
