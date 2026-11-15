"""Encoder bake-off on BM→Consum barcode pairs (no fine-tuning, no type filter).

Run: python -m src.similarity.bakeoff [model_id ...]   → data/similarity/results/bakeoff.csv
"""

import sys

import numpy as np
import pandas as pd

from src.pipeline import config
from src.similarity.data import barcode_pairs, load_products
from src.similarity.encoder import Encoder
from src.similarity.evaluate import grouped_folds, retrieval_metrics, tfidf_rank
from src.similarity.index import VectorIndex

CANDIDATES = ["intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large", "BAAI/bge-m3"]
K = 50


def hub_revision(model_id):
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id).sha


def _fold_rows(name, ranked, gold, folds):
    rows = []
    for i, (_, test) in enumerate(folds):
        rows.append({"model": name, "fold": i, **retrieval_metrics(ranked[test], gold[test])})
    mean = pd.DataFrame(rows)[["top1", "top5", "mrr"]].mean().round(4).to_dict()
    rows.append({"model": name, "fold": "mean", **mean})
    return rows


def run(models=CANDIDATES):
    products = load_products()
    pairs = barcode_pairs(products)
    folds = grouped_folds(pairs)
    by_key = products.set_index("product_key")
    consum = products[products["chain_id"] == "consum"].reset_index(drop=True)
    gold = pairs["key_consum"].to_numpy(dtype=object)
    rows = []

    for label, col in [("tfidf (name_clean)", "name_clean"), ("tfidf (brand + name)", "mtext")]:
        ranked = tfidf_rank(by_key.loc[pairs["key_bm"], col], consum[col], consum["product_key"], K)
        rows += _fold_rows(label, ranked, gold, folds)

    for model_id in models:
        revision = hub_revision(model_id)
        enc = Encoder(model_id, revision)
        index = VectorIndex(consum["product_key"], enc.encode(list(consum["mtext"]), "query"),
                            consum.set_index("product_key"))
        ranked, _ = index.topk(enc.encode(list(by_key.loc[pairs["key_bm"], "mtext"]), "query"), K)
        enc.save_cache()
        rows += _fold_rows(f"{model_id}@{revision}", ranked, gold, folds)
        print(rows[-1], flush=True)
        del enc

    table = pd.DataFrame(rows)
    path = config.SIM_RESULTS_DIR / "bakeoff.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


if __name__ == "__main__":
    print(run(sys.argv[1:] or CANDIDATES).query("fold == 'mean'").to_string(index=False))
