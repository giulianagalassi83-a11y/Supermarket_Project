"""Evaluation for the similarity engine: retrieval metrics, grouped folds, baselines and result tables."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from src.pipeline import config


def retrieval_metrics(ranked_keys, gold):
    """top-1, top-5 and MRR of each row's gold key in its ranking (beyond the ranking → reciprocal rank 0)."""
    ranked_keys = np.asarray(ranked_keys, dtype=object)
    hits = ranked_keys == np.asarray(gold, dtype=object)[:, None]
    found = hits.any(axis=1)
    rank = np.where(found, hits.argmax(axis=1) + 1, np.inf)
    return {
        "top1": float(np.mean(rank <= 1)),
        "top5": float(np.mean(rank <= 5)),
        "mrr": float(np.mean(np.where(found, 1 / rank, 0.0))),
    }


def grouped_folds(pairs, n=config.SIM["n_folds"]):
    """Deterministic GroupKFold on the BM product's search_term: a category is never in train and test."""
    return list(GroupKFold(n_splits=n).split(pairs, groups=pairs["search_term"]))


def tfidf_rank(query_texts, target_texts, target_keys, k):
    """Baseline ranking: char_wb 3–5 TF-IDF cosine (as the pipeline's fallback)."""
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(list(query_texts) + list(target_texts))
    scores = (tfidf.transform(query_texts) @ tfidf.transform(target_texts).T).toarray()
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return np.asarray(target_keys, dtype=object)[order]


def write_result(table, row):
    """Append one row to data/similarity/results/<table>.csv."""
    path = config.SIM_RESULTS_DIR / f"{table}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def prf(accepted, gold):
    """Precision = correct accepted / accepted; recall = correct accepted / all queries. Both Series by query."""
    accepted = accepted.reindex(gold.index)
    made = accepted.notna()
    correct = made & (accepted == gold)
    precision = float(correct.sum() / made.sum()) if made.any() else float("nan")
    recall = float(correct.sum() / len(gold)) if len(gold) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if made.any() and (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accepted": int(made.sum())}


# ---------------------------------------------------------------- matching cross-validation (spec §8, Task 8)


def tradeoff(heldout, thresholds=np.round(np.arange(0.30, 1.0, 0.01), 2)):
    """Precision / recall / F1 / F0.5 per fold and threshold from held-out best candidates, plus a 'mean' fold.

    Gold is barcode-only, so precision is a lower bound (same product under another EAN counts as wrong)."""
    rows = []
    for fold, g in heldout.groupby("fold"):
        for t in thresholds:
            made = g["prob"] >= t
            correct = (made & g["correct"]).sum()
            p = correct / made.sum() if made.any() else np.nan
            r = correct / len(g)
            f = lambda b: (1 + b * b) * p * r / (b * b * p + r) if made.any() and p + r else 0.0
            rows.append({"fold": fold, "threshold": t, "precision": p, "recall": r, "f1": f(1), "f05": f(0.5),
                         "accepted_share": made.mean()})
    table = pd.DataFrame(rows)
    mean = table.groupby("threshold", as_index=False).agg(
        precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"), f05=("f05", "mean"),
        accepted_share=("accepted_share", "mean"), precision_min=("precision", "min"),
        precision_max=("precision", "max"))
    return pd.concat([table, mean.assign(fold="mean")], ignore_index=True)

VARIANTS = ["tfidf", "base", "finetuned", "finetuned+attrs", "finetuned+attrs+rerank", "finetuned+attrs+rerank_ft"]
_ATTR_FEATURES = ["brand_match", "size_ratio", "size_known", "variant_agree", "pack_match"]
_VARIANT_FEATURES = {
    "tfidf": ["cosine"], "base": ["cosine"], "finetuned": ["cosine"],
    "finetuned+attrs": ["cosine"] + _ATTR_FEATURES,
    "finetuned+attrs+rerank": ["cosine", "ce_score"] + _ATTR_FEATURES,
    "finetuned+attrs+rerank_ft": ["cosine", "ce_score"] + _ATTR_FEATURES,  # per-fold fine-tuned re-ranker
}
CE_SETTINGS = "e2-n7"  # tag of the `train --cv-reranker` run whose held-out scores are evaluated
FOLD_SETTINGS = (3, 3)  # (epochs, per_pair) of the `train --cv` run whose cached embeddings are evaluated


class _NoModel:
    def encode(self, texts, **kw):
        raise RuntimeError(f"{len(texts)} texts missing from the CV cache: run python -m src.similarity.train --cv")


def _oof_candidates(variant, products, pairs, folds, k):
    """Top-k Consum candidates for every BM query, each scored by a model that never trained on that query."""
    from src.similarity.encoder import Encoder
    from src.similarity.index import VectorIndex
    from src.similarity.train import CV_CACHE_PATH, fold_model_key

    text = products.set_index("product_key")["mtext"]
    consum = products[products["chain_id"] == "consum"].reset_index(drop=True)
    frames = []
    for i, (_, test) in enumerate(folds):
        q = pairs.loc[test, "key_bm"].to_numpy()
        if variant == "tfidf":
            tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(list(text[q]) + list(consum["mtext"]))
            sims = (tf.transform(text[q]) @ tf.transform(consum["mtext"]).T).toarray()
            order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
            keys, cos = consum["product_key"].to_numpy()[order], np.take_along_axis(sims, order, 1)
        else:
            if variant == "base":
                enc = Encoder(config.SIM["encoder_id"], config.SIM["encoder_revision"])
            else:
                model_id, revision = fold_model_key(i, *FOLD_SETTINGS)
                enc = Encoder(model_id, revision, CV_CACHE_PATH, model=_NoModel(), prefix_id=config.SIM["encoder_id"])
            index = VectorIndex(consum["product_key"], enc.encode(list(consum["mtext"]), "query"),
                                consum.set_index("product_key"))
            keys, cos = index.topk(enc.encode(list(text[q]), "query"), k)
        frames.append(pd.DataFrame({"key_a": np.repeat(q, keys.shape[1]), "key_b": keys.ravel(),
                                    "cosine": cos.ravel(), "fold": i}))
    cands = pd.concat(frames, ignore_index=True)
    cands["y"] = (cands["key_b"] == cands["key_a"].map(pairs.set_index("key_bm")["key_consum"])).astype(int)
    return cands


def _top5(scored, queries):
    """(n_queries, 5) array of each query's best keys by prob, padded with ''."""
    lists = (scored.sort_values(["key_a", "prob"], ascending=[True, False])
             .groupby("key_a")["key_b"].apply(lambda s: list(s)[:5]))
    return np.array([(lists.get(q, []) + [""] * 5)[:5] for q in queries], dtype=object)


def run_matching_cv(products, pairs, variant, k=config.SIM["top_k"], reranker=None, heldout=None, generic_cutoff=2):
    """Held-out metrics per outer fold: decider fit on the other folds, threshold chosen by an inner CV on them.

    `heldout`: optional list; gets one frame per fold (fold, key_a, prob, correct) of each query's best candidate.
    `generic_cutoff`: attributes.generic_brands' max_other_brands (sensitivity check only)."""
    from src.similarity import attributes
    from src.similarity.match import add_features, apply_veto, best_per_query, choose_threshold, fit_decider, predict

    folds = grouped_folds(pairs)
    cands = _oof_candidates("finetuned" if variant.startswith("finetuned") else variant, products, pairs, folds, k)
    by_key = products.set_index("product_key")
    keys = pd.unique(pd.concat([cands["key_a"], cands["key_b"]]))
    attrs = attributes.extract_all(products, keys, generic_cutoff)
    if "attrs" in variant:
        cands = apply_veto(cands, attrs, "same")
    cands = add_features(cands, by_key["mtext"].to_dict(), attrs, reranker if variant.endswith("rerank") else None)
    if variant.endswith("rerank_ft"):
        from src.similarity.train import CE_CV_SCORES_PATH

        ce = pd.read_parquet(CE_CV_SCORES_PATH)
        ce = ce[ce["tag"] == CE_SETTINGS].set_index(["key_a", "key_b"])["ce_score"]
        idx = pd.MultiIndex.from_arrays([cands["key_a"], cands["key_b"]])
        if not idx.isin(ce.index).all():
            raise RuntimeError("held-out re-ranker scores missing: run python -m src.similarity.train --cv-reranker")
        cands["ce_score"] = ce.reindex(idx).to_numpy(dtype=float)
    features = _VARIANT_FEATURES[variant]
    gold = pairs.set_index("key_bm")["key_consum"]
    rows = []
    for i, (_, test_idx) in enumerate(folds):
        train = cands[cands["fold"] != i]
        oof = np.zeros(len(train))
        for j in sorted(train["fold"].unique()):  # inner CV → out-of-fold probabilities for the threshold
            inner = (train["fold"] == j).to_numpy()
            oof[inner] = predict(fit_decider(train[~inner], train.loc[~inner, "y"], features), train[inner])
        best = train.assign(prob=oof).sort_values("prob", ascending=False, kind="stable").drop_duplicates("key_a")
        threshold = choose_threshold(best["prob"], best["y"], config.SIM["precision_target"])
        model = fit_decider(train, train["y"], features)
        queries = pairs.loc[test_idx, "key_bm"].to_numpy()
        test = cands[cands["fold"] == i].assign(prob=lambda d: predict(model, d))
        accepted = best_per_query(test, threshold).set_index("key_a")["key_b"].reindex(queries)
        if heldout is not None:
            top = best_per_query(test, -np.inf).set_index("key_a").reindex(queries)
            heldout.append(pd.DataFrame({"fold": i, "key_a": queries, "prob": top["prob"].fillna(0).to_numpy(),
                                         "correct": (top["key_b"] == gold[queries]).to_numpy()}))
        rows.append({"variant": variant, "fold": i, "threshold": round(threshold, 4),
                     **retrieval_metrics(_top5(test, queries), gold[queries].to_numpy(dtype=object)),
                     **prf(accepted, gold[queries])})
    table = pd.DataFrame(rows)
    mean = {"variant": variant, "fold": "mean", **table.drop(columns=["variant", "fold"]).mean().round(4).to_dict()}
    return pd.concat([table, pd.DataFrame([mean])], ignore_index=True)


# ---------------------------------------------------------------- report tables (spec §8, Task 13)


def md_table(df, digits=3):
    """A pandas frame as a GitHub markdown table (no tabulate dependency)."""
    fmt = lambda v: "" if pd.isna(v) else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def relative_to_cheapest(wide):
    """Per chain: median price relative to the cheapest chain on each row (1.00 = cheapest), rows, cheapest share."""
    wide = wide[wide.notna().sum(axis=1) >= 2]
    rel = wide.div(wide.min(axis=1), axis=0)
    cheapest = wide.idxmin(axis=1)
    out = pd.DataFrame({"median_vs_cheapest": rel.median(), "rows": wide.notna().sum(),
                        "cheapest_share": [(cheapest[wide[c].notna()] == c).mean() for c in wide.columns]})
    return out.sort_values("median_vs_cheapest").rename_axis("chain").reset_index()


def type_median_wide(df, exclude=()):
    """Type × chain table of median € per unit (the dashboard's comparison before matching)."""
    from src.analysis import prices

    usable = prices.usable(df)
    usable = usable[~usable["supermarket"].isin(exclude)]
    return usable.groupby(["product_type", "supermarket"])["price_per_unit"].median().unstack()


def pair_gap(wide, a, b):
    """Median % by which chain a is dearer than chain b on the rows both have."""
    both = wide.dropna(subset=[a, b]) if {a, b} <= set(wide.columns) else wide.iloc[0:0]
    gap = 100 * (both[a] / both[b] - 1)
    return {"n": int(len(both)), "median_gap_pct": float(gap.median()) if len(gap) else float("nan"),
            "a_cheaper_share": float((gap < -0.5).mean()) if len(gap) else float("nan")}


def _silver_tables(cands, labels):
    """Engine decision vs the Gemini judge on the other-chain candidates: by decision, and accepted pairs by chain."""
    s = cands.merge(labels.drop_duplicates(["key_a", "key_b"])[["key_a", "key_b", "label"]], on=["key_a", "key_b"])
    other = s[s["source"] == "other_chains"]
    count = lambda g: pd.Series({"n": len(g), "same": (g["label"] == "same").sum(),
                                 "equivalent": (g["label"] == "equivalent").sum(),
                                 "different": (g["label"] == "different").sum(),
                                 "match_share": g["label"].isin(["same", "equivalent"]).mean(),
                                 "same_share": (g["label"] == "same").mean()}, dtype=object)
    by_decision = other.groupby("accepted").apply(count).reindex(["high", "balanced", "no"]).dropna(how="all")
    by_chain = other[other["accepted"] != "no"].groupby("chain_a").apply(count)
    errs = s[s["source"] != "other_chains"].groupby("source").apply(count)
    return by_decision.reset_index(), by_chain.reset_index(), errs.reset_index()


def report(results_dir=None, sim_dir=None, df=None):
    """results/report_tables.md: every spec §8 question with its numbers, or why a number is missing."""
    results_dir = results_dir or config.SIM_RESULTS_DIR
    sim_dir = sim_dir or config.SIM_DIR
    out = ["# Similarity engine: report tables",
           "", "Generated by `python -m src.similarity.evaluate report` from the CSVs in data/similarity/results. "
           "Barcode metrics are 5-fold grouped CV on BM↔Consum barcode pairs (a category is never in train and test).", ""]

    def section(title, files, build):
        out.extend([f"## {title}", ""])
        missing = [f for f in files if not (f if hasattr(f, "exists") else results_dir / f).exists()]
        if missing:
            out.extend([f"_Not available: {', '.join(map(str, missing))} missing._", ""])
            return
        try:
            out.extend([build(), ""])
        except Exception as e:  # one broken table should not hide the others
            out.extend([f"_Not available: {type(e).__name__}: {e}_", ""])

    mean = lambda f: pd.read_csv(results_dir / f).query("fold == 'mean'")

    section("1. Right product retrieved? (top-1 / top-5 / MRR)", ["bakeoff.csv", "finetune_cv.csv"], lambda: "\n\n".join([
        "Off-the-shelf encoders vs TF-IDF (no fine-tuning, no attributes):",
        md_table(mean("bakeoff.csv")[["model", "top1", "top5", "mrr"]].assign(model=lambda d: d["model"].str.split("@").str[0])),
        "Fine-tuned bge-m3 (held-out folds): " + ", ".join(
            f"{c} {mean('finetune_cv.csv')[c].astype(float).iat[0]:.3f}" for c in ("top1", "top5", "mrr")) + "."]))

    def ablation():
        a = pd.read_csv(results_dir / "ablation.csv")
        folds = a[a["fold"] != "mean"].astype({"precision": float})
        rng = folds.groupby("variant")["precision"].agg(lambda p: f"{p.min():.2f}–{p.max():.2f}")
        m = a[a["fold"] == "mean"].set_index("variant")
        m["precision per fold"] = rng
        cols = ["top1", "top5", "mrr", "precision", "recall", "f1", "precision per fold"]
        return ("Each row adds one component. Decision rule: per fold, the lowest threshold with precision ≥ .90 "
                "on the training folds. The re-ranker was dropped (15 Nov); its rows are kept for the record.\n\n"
                + md_table(m[cols].astype({c: float for c in cols[:-1]}).reset_index()))

    section("2. Each component's value (ablation)", ["ablation.csv"], ablation)

    def operating():
        t = mean("tradeoff.csv").drop(columns="fold").astype(float)
        hi = t.loc[(t["threshold"] - config.SIM["high_confidence_t"]).abs().idxmin()]
        bal = pd.read_csv(results_dir / "ablation.csv").query("variant == 'finetuned+attrs' and fold == 'mean'").iloc[0]
        rows = pd.DataFrame([
            {"point": f"high (threshold {hi['threshold']:.2f})", "precision": hi["precision"], "recall": hi["recall"],
             "f1": hi["f1"], "precision per fold": f"{hi['precision_min']:.2f}–{hi['precision_max']:.2f}"},
            {"point": "balanced (P ≥ .90 rule)", "precision": float(bal["precision"]), "recall": float(bal["recall"]),
             "f1": float(bal["f1"]), "precision per fold": "see ablation"}])
        return ("Final decider (fine-tuned + attributes), held-out barcode pairs:\n\n" + md_table(rows)
                + "\n\nPrecision is a CV mean, not a guarantee: individual folds range as shown.")

    section("3. Accepted matches correct? (P / R / F1)", ["tradeoff.csv", "ablation.csv"], operating)

    def silver():
        dec, chain, errs = _silver_tables(pd.read_csv(sim_dir / "silver_candidates.csv"),
                                          pd.read_csv(sim_dir / "llm_pair_labels.csv"))
        return ("Gemini judge labels on candidates from the 5 chains without barcodes (one row per engine decision):\n\n"
                + md_table(dec) + "\n\nAccepted pairs (high + balanced) by chain:\n\n" + md_table(chain)
                + "\n\nBM↔Consum pairs the engine accepted whose barcodes disagree:\n\n" + md_table(errs)
                + "\n\nThe judge itself is validated by κ against humans (section 5): read these as silver, not gold.")

    section("4. Generalizes to other chains? (LLM silver labels)",
            [sim_dir / "silver_candidates.csv", sim_dir / "llm_pair_labels.csv"], silver)

    def kappa():
        k = pd.read_csv(results_dir / "kappa.csv")
        parts = ["Cohen's κ per rater pair ('unsure' rows left out):", md_table(k)]
        p = results_dir / "audit_consensus_summary.csv"
        if p.exists():
            parts += ["Consensus of the human raters (majority = more than half; Gemini is not a voter):",
                      md_table(pd.read_csv(p))]
        return "\n\n".join(parts)

    section("5. 'Fair equivalent' well defined? (κ, blind audit)", ["kappa.csv"], kappa)

    section("6. Audit: engine decisions vs Felipe's labels (100 pairs)", ["audit_matches.csv"], lambda: (
        "`same` / `match` = share of pairs Felipe labelled same / same-or-equivalent, per engine decision:\n\n"
        + md_table(pd.read_csv(results_dir / "audit_matches.csv").query("chain_a == 'all'"))))

    section("7. Equivalent-level decider", ["equivalent_decider.csv"],
            lambda: md_table(pd.read_csv(results_dir / "equivalent_decider.csv")))

    section("8. Classification better? (97 hand labels)", ["classification.csv"], lambda: (
        md_table(pd.read_csv(results_dir / "classification.csv").assign(model=lambda d: d["model"].str.split("@").str[0])[
            ["vectorizer", "model", "test_n", "test_accuracy", "test_accuracy_ci95", "test_macro_f1",
             "relevant_precision", "relevant_recall"]])
        + "\n\nProduction classifier: TF-IDF (confirmed 19 Nov); confusion matrix in results/confusion.png."))

    def clusters():
        parts = [md_table(pd.read_csv(results_dir / "cluster_agreement.csv")),
                 "'real types' leaves out no_relevante, one mixed bag that drags ARI to about .01."]
        review = results_dir / "cluster_review.xlsx"
        if review.exists():
            r = pd.read_excel(review)
            ok = r["name_ok"].astype(str).str.lower().value_counts()
            new = r[r["should_be_new_type"].astype(str).str.lower() == "yes"]
            parts += [f"Human review of the 30 largest clusters: name fits {ok.get('yes', 0)}, does not fit {ok.get('no', 0)} "
                      "(stray members at the edges). Clusters flagged as possible new types:",
                      md_table(new[["cluster_id", "gemini_name", "size", "share_of_that_type"]]
                               .rename(columns={"share_of_that_type": "share no_relevante"})),
                      "Follow-up (22 Nov): pasta, packaged rice and plant drinks turned out to be existing types the "
                      "TF-IDF classifier missed: pasta and rice descriptions were enriched (hand-label accuracy unchanged); "
                      "plant drinks found only by Consum's 'huevos' search stay out by design. The rest (biscuits, soft "
                      "drinks, spices, ...) are side catches of other searches from few chains, so they were not added "
                      "as types: they would need their own search term in the scrapers."]
        return "\n\n".join(parts)

    section("9. Emergent vs hand categories (ARI / NMI)", ["cluster_agreement.csv"], clusters)

    def app():
        from src.analysis.matched import EXCLUDED, same_product_table

        frame = pd.read_csv(config.CSV_PATH, encoding="utf-8") if df is None else df
        matches, canonical = pd.read_csv(sim_dir / config.SIM_MATCHES_PATH.name), pd.read_csv(sim_dir / config.SIM_CANONICAL_PATH.name)
        table = same_product_table(frame, matches, canonical, exclude=EXCLUDED)
        chains = [c for c in table.columns if c not in
                  ("canonical_id", "name", "unit", "chains", "cheapest", "gap_pct", "confidence")]
        before, after = type_median_wide(frame, exclude=EXCLUDED), table.set_index("canonical_id")[chains]
        gaps = pd.DataFrame([{"comparison": "before: type medians", **pair_gap(before, "Bm", "Consum")},
                             {"comparison": "after: same product", **pair_gap(after, "Bm", "Consum")}])
        rank = relative_to_cheapest(before).merge(relative_to_cheapest(after), on="chain", suffixes=(" (types)", " (same product)"))
        cover = matches.groupby(["level", "confidence"]).size().rename("pairs").reset_index()
        return ("BM vs Consum, % BM is dearer (median over shared rows):\n\n" + md_table(gaps)
                + "\n\nChain ranking: median € per unit relative to the cheapest chain on each row (1.00 = cheapest). "
                f"{', '.join(EXCLUDED)} left out (its scraper sometimes stores the €/kg price as the item price).\n\n"
                + md_table(rank) + f"\n\nMatches in matches.csv:\n\n{md_table(cover)}\n\n"
                f"Canonical groups: {canonical['canonical_id'].nunique():,}; comparable on € per unit (≥ 2 chains): {len(table):,}.")

    section("10. App improved? (price gap and ranking before vs after matching)",
            [sim_dir / config.SIM_MATCHES_PATH.name, sim_dir / config.SIM_CANONICAL_PATH.name], app)

    path = results_dir / "report_tables.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys

    from src.similarity.data import barcode_pairs, load_products

    if sys.argv[1:2] == ["report"]:  # Task 13: results/report_tables.md from the result CSVs (no GPU, no API)
        print(report())

    elif sys.argv[1:2] == ["tradeoff"]:  # precision–recall trade-off of the final decider (finetuned+attrs)
        products = load_products()
        heldout = []
        run_matching_cv(products, barcode_pairs(products), "finetuned+attrs", heldout=heldout)
        heldout = pd.concat(heldout, ignore_index=True)
        out = tradeoff(heldout)
        out.to_csv(config.SIM_RESULTS_DIR / "tradeoff.csv", index=False)
        mean = out[out["fold"] == "mean"]
        print(f"recall ceiling (best candidate correct): {heldout['correct'].mean():.3f}")
        print(mean.iloc[::5].round(3).to_string(index=False))
        for col in ("f1", "f05"):
            print(f"best {col}:", mean.drop(columns="fold").loc[mean[col].idxmax()].round(3).to_dict())

    elif sys.argv[1:2] == ["generic-sensitivity"]:  # does the generic-brand cutoff matter? (finetuned+attrs)
        from src.similarity import attributes

        products = load_products()
        pairs = barcode_pairs(products)
        by_key = products.drop_duplicates("product_key").set_index("product_key")
        names = [attributes.extract(r["name"], r["brand"], r["chain_id"]) for _, r in by_key.iterrows()]
        sets = {c: attributes.generic_brands([a.name for a in names], [a.brand for a in names], c)
                for c in (1, 2, 3, 4, 5, 7, 10, 15, 20)}
        rows = []
        for c, generic in sets.items():
            attrs = attributes.extract_all(products, pd.unique(pd.concat([pairs["key_bm"], pairs["key_consum"]])), c)
            vetoed = sum(attributes.conflict(attrs[a], attrs[b], "same") == "brand"
                         for a, b in zip(pairs["key_bm"], pairs["key_consum"]))
            heldout = []
            m = run_matching_cv(products, pairs, "finetuned+attrs", heldout=heldout, generic_cutoff=c)
            m = m[m["fold"] == "mean"].iloc[0]
            t50 = tradeoff(pd.concat(heldout, ignore_index=True), np.array([0.5]))
            t50 = t50[t50["fold"] == "mean"].iloc[0]
            rows.append({"cutoff": c, "n_generic": len(generic),
                         "jaccard_vs_2": round(len(generic & sets[2]) / max(len(generic | sets[2]), 1), 3),
                         "true_pairs_vetoed_brand": vetoed, "n_true_pairs": len(pairs),
                         "P_rule": m["precision"], "R_rule": m["recall"], "MRR": m["mrr"],
                         "P_t50": round(t50["precision"], 4), "R_t50": round(t50["recall"], 4)})
            print(rows[-1], flush=True)
        out = pd.DataFrame(rows)
        out.to_csv(config.SIM_RESULTS_DIR / "generic_sensitivity.csv", index=False)
        print(out.to_string(index=False))
        print("only at cutoff 2 vs 5:", sorted(sets[2] - sets[5])[:40])

    elif sys.argv[1:2] == ["matching"]:
        from src.similarity.rerank import load_reranker

        products = load_products()
        pairs = barcode_pairs(products)
        reranker = load_reranker()
        only = sys.argv[2:]  # optional variant names: run only these and replace their rows in ablation.csv
        out = pd.concat([run_matching_cv(products, pairs, v, reranker=reranker) for v in (only or VARIANTS)],
                        ignore_index=True)
        path = config.SIM_RESULTS_DIR / "ablation.csv"
        if only and path.exists():
            old = pd.read_csv(path)
            out = pd.concat([old[~old["variant"].isin(only)], out], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        print(out[out["fold"] == "mean"].to_string(index=False))
