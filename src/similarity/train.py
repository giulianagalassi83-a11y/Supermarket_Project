"""Contrastive fine-tuning of the bi-encoder on BM–Consum barcode pairs (laptop / GPU only).

  python -m src.similarity.train --cv      one model per grouped fold; held-out retrieval → results/finetune_cv.csv
                                           (fold models are deleted after their embeddings are cached in
                                           data/similarity/cv_embeddings.parquet, used by evaluate.py)
  python -m src.similarity.train --final   all pairs → config.SIM["finetuned_path"]
"""

import argparse
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from src.pipeline import config
from src.similarity import attributes
from src.similarity.data import barcode_pairs, load_products
from src.similarity.encoder import Encoder, _prefix
from src.similarity.evaluate import grouped_folds, retrieval_metrics
from src.similarity.index import VectorIndex

CV_CACHE_PATH = config.SIM_DIR / "cv_embeddings.parquet"  # gitignored
MAX_SEQ_LENGTH = 64  # product titles are short; keeps bge-m3 inside 8 GB of VRAM
CV_PARTIAL_PATH = config.SIM_DIR / "finetune_cv_partial.csv"  # finished folds, so a crash resumes mid-CV

# Laptop guard (14 Nov: a sustained full-load run restarted the laptop). Sleep after every step to cap the
# duty cycle, and pause whenever the GPU gets hot.
THROTTLE_S = 0.25        # sleep per optimizer step (~0.42 s of compute) -> roughly 60 % duty cycle
HOT_C, COOL_C = 78, 68   # pause at HOT_C until the GPU is back to COOL_C
FOLD_COOLDOWN_S = 90


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.split()[0])
    except Exception:
        return None


def cool_down(target=COOL_C, max_wait=600):
    start = time.time()
    while (t := gpu_temp()) is not None and t > target and time.time() - start < max_wait:
        time.sleep(5)


def _throttle_callback(throttle_s):
    from transformers import TrainerCallback

    class Throttle(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            time.sleep(throttle_s)
            if state.global_step % 10 == 0 and (t := gpu_temp()) is not None and t >= HOT_C:
                print(f"[throttle] GPU {t} C at step {state.global_step}; pausing until {COOL_C} C", flush=True)
                cool_down()

    return Throttle()


def mine_hard_negatives(pairs, products, index, encoder, per_pair=3, exclude=frozenset(), depth=15):
    """Top-ranked Consum products other than the gold one, for each pair.

    Skips candidates with no attribute conflict against the gold at the 'same' level (likely the same product
    under another EAN); multipacks, other sizes and other variants are kept on purpose (spec §9b).
    `exclude` holds keys that must never be used (e.g. the held-out fold's products).
    """
    by_key = products.set_index("product_key")
    attrs = {k: attributes.extract(r["name"], r["brand"], r["chain_id"]) for k, r in by_key.iterrows()}
    ranked, _ = index.topk(encoder.encode(list(by_key.loc[pairs["key_bm"], "mtext"]), "query"), depth)
    rows = []
    for (key_bm, gold), cands in zip(pairs[["key_bm", "key_consum"]].itertuples(index=False), ranked):
        taken = 0
        for cand in cands:
            if taken == per_pair:
                break
            if not cand or cand == gold or cand in exclude:
                continue
            if attributes.conflict(attrs[gold], attrs[cand], "same") is None:
                continue
            rows.append((key_bm, gold, cand))
            taken += 1
    return pd.DataFrame(rows, columns=["key_bm", "key_consum", "key_neg"])


def fine_tune(train_triplets, products, base_id, base_revision, out_dir, epochs=3, batch_size=32,
              throttle_s=THROTTLE_S):
    """MultipleNegativesRankingLoss over (anchor, positive, hard negative); returns out_dir."""
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    text = products.set_index("product_key")["mtext"]
    p = _prefix(base_id, "query")
    data = Dataset.from_dict({
        "anchor": [p + t for t in text[train_triplets["key_bm"]]],
        "positive": [p + t for t in text[train_triplets["key_consum"]]],
        "negative": [p + t for t in text[train_triplets["key_neg"]]],
    })
    model = SentenceTransformer(base_id, revision=base_revision)
    model.max_seq_length = MAX_SEQ_LENGTH
    # Freeze the token-embedding matrix (≈ 256M of bge-m3's 568M params): fits 8 GB of VRAM with AdamW
    # and avoids overfitting the vocabulary to ~600 pairs.
    model[0].auto_model.get_input_embeddings().weight.requires_grad_(False)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "_checkpoints"), num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=2e-5, warmup_ratio=0.1, fp16=True, seed=config.SEED, save_strategy="no",
        optim="adafactor",  # AdamW's two fp32 moments overflow 8 GB of VRAM with bge-m3
        logging_steps=50, report_to="none", batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=data,
                                         loss=losses.MultipleNegativesRankingLoss(model),
                                         callbacks=[_throttle_callback(throttle_s)])
    trainer.train()
    model.save_pretrained(str(out_dir))
    shutil.rmtree(out_dir / "_checkpoints", ignore_errors=True)
    return out_dir


def _consum_index(encoder, consum):
    vectors = encoder.encode(list(consum["mtext"]), "query")
    return VectorIndex(consum["product_key"], vectors, consum.set_index("product_key"))


def fold_model_key(i, epochs, per_pair):
    """Cache key of a fold model's embeddings (the weights themselves are not kept)."""
    return f"{config.SIM['encoder_id']}-ft-fold{i}", f"{config.SIM['encoder_revision'][:8]}-e{epochs}-n{per_pair}"


def run_cv(epochs=3, per_pair=3, throttle_s=THROTTLE_S):
    from sentence_transformers import SentenceTransformer

    products = load_products()
    pairs = barcode_pairs(products)
    bm_consum = products[products["chain_id"].isin(["bm", "consum"])]
    consum = products[products["chain_id"] == "consum"].reset_index(drop=True)
    base = Encoder(config.SIM["encoder_id"], config.SIM["encoder_revision"])
    base_index = _consum_index(base, consum)
    done = pd.read_csv(CV_PARTIAL_PATH) if CV_PARTIAL_PATH.exists() else pd.DataFrame()
    if len(done):
        done = done[(done["epochs"] == epochs) & (done["per_pair"] == per_pair)]
    rows = done.to_dict("records")
    for i, (train, test) in enumerate(grouped_folds(pairs)):
        if any(r["fold"] == i for r in rows):
            print(f"fold {i}: already done, skipping", flush=True)
            continue
        cool_down()
        held_out = set(pairs.loc[test, "key_bm"]) | set(pairs.loc[test, "key_consum"])
        triplets = mine_hard_negatives(pairs.loc[train], products, base_index, base, per_pair, exclude=held_out)
        out = config.MODELS_DIR / f"fold_{i}"
        fine_tune(triplets, products, config.SIM["encoder_id"], config.SIM["encoder_revision"], out, epochs,
                  throttle_s=throttle_s)
        model_id, revision = fold_model_key(i, epochs, per_pair)
        enc = Encoder(model_id, revision, CV_CACHE_PATH, model=SentenceTransformer(str(out)),
                      prefix_id=config.SIM["encoder_id"])
        enc.encode(list(bm_consum["mtext"]), "query")  # every BM + Consum text, for evaluate.py
        enc.save_cache()
        index = _consum_index(enc, consum)
        ranked, _ = index.topk(enc.encode(list(products.set_index("product_key").loc[pairs.loc[test, "key_bm"], "mtext"]),
                                          "query"), config.SIM["top_k"])
        row = {"fold": i, "epochs": epochs, "per_pair": per_pair, "triplets": len(triplets),
               **retrieval_metrics(ranked, pairs.loc[test, "key_consum"].to_numpy(dtype=object))}
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame([row]).to_csv(CV_PARTIAL_PATH, mode="a", header=not CV_PARTIAL_PATH.exists(), index=False)
        del enc
        shutil.rmtree(out, ignore_errors=True)
        import torch
        torch.cuda.empty_cache()
        time.sleep(FOLD_COOLDOWN_S)
    table = pd.DataFrame(rows)
    mean = {"fold": "mean", "epochs": epochs, "per_pair": per_pair, "triplets": int(table["triplets"].mean()),
            **table[["top1", "top5", "mrr"]].mean().round(4).to_dict()}
    table = pd.concat([table, pd.DataFrame([mean])], ignore_index=True)
    path = config.SIM_RESULTS_DIR / "finetune_cv.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, mode="a", header=not path.exists(), index=False)
    CV_PARTIAL_PATH.unlink(missing_ok=True)
    return table


# ---------------------------------------------------------------- cross-encoder fine-tuning (Felipe, 15 Nov)

CE_CV_SCORES_PATH = config.SIM_DIR / "cv_ce_scores.parquet"  # gitignored: held-out re-ranker scores per fold
CE_MAX_LENGTH = 96  # two short titles


def ce_training_pairs(cands, attrs, texts, gold, neg_per_query=7):
    """(text_a, text_b, label) rows: the gold pair plus the top hard negatives per query.

    Negatives with no 'same'-level attribute conflict against the gold are skipped (likely the same product
    under another EAN), as in mine_hard_negatives.
    """
    rows = []
    for key_a, group in cands.sort_values("cosine", ascending=False).groupby("key_a", sort=False):
        g = gold[key_a]
        rows.append((texts[key_a], texts[g], 1.0))
        taken = 0
        for key_b in group["key_b"]:
            if taken == neg_per_query:
                break
            if not key_b or key_b == g or attributes.conflict(attrs[g], attrs[key_b], "same") is None:
                continue
            rows.append((texts[key_a], texts[key_b], 0.0))
            taken += 1
    return pd.DataFrame(rows, columns=["text_a", "text_b", "label"])


def fine_tune_reranker(train_rows, out_dir, epochs=2, batch_size=16, throttle_s=THROTTLE_S):
    """BCE fine-tune of the configured cross-encoder on (text_a, text_b, label) rows; returns out_dir."""
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                     CrossEncoderTrainingArguments)
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    model = CrossEncoder(config.SIM["reranker_id"], revision=config.SIM["reranker_revision"],
                         max_length=CE_MAX_LENGTH, num_labels=1)
    model.model.get_input_embeddings().weight.requires_grad_(False)  # 8 GB VRAM, as in fine_tune
    data = Dataset.from_dict({c: list(train_rows[c]) for c in ["text_a", "text_b", "label"]})
    pos_weight = float((train_rows["label"] == 0).sum() / max(1, (train_rows["label"] == 1).sum()))
    args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / "_checkpoints"), num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=2e-5, warmup_ratio=0.1, fp16=True, seed=config.SEED, save_strategy="no",
        optim="adafactor", logging_steps=50, report_to="none",
    )
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=data,
                                  loss=BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(pos_weight)),
                                  callbacks=[_throttle_callback(throttle_s)])
    trainer.train()
    model.save_pretrained(str(out_dir))
    shutil.rmtree(out_dir / "_checkpoints", ignore_errors=True)
    return out_dir


def run_cv_reranker(epochs=2, neg_per_query=7, throttle_s=THROTTLE_S):
    """Per outer fold: fine-tune the re-ranker on the other folds' queries, score the held-out fold's candidates
    (fine-tuned bi-encoder top-k) and save them to CE_CV_SCORES_PATH. Finished folds are skipped."""
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder

    from src.similarity.evaluate import _oof_candidates
    from src.similarity.rerank import Reranker

    products = load_products()
    pairs = barcode_pairs(products)
    folds = grouped_folds(pairs)
    cands = _oof_candidates("finetuned", products, pairs, folds, config.SIM["top_k"])
    cands = cands[cands["key_b"] != ""]
    by_key = products.set_index("product_key")
    texts = by_key["mtext"].to_dict()
    keys = pd.unique(pd.concat([cands["key_a"], cands["key_b"], pairs["key_consum"]]))
    attrs = {k: attributes.extract(by_key.at[k, "name"], by_key.at[k, "brand"], by_key.at[k, "chain_id"])
             for k in keys}
    gold = pairs.set_index("key_bm")["key_consum"]
    tag = f"e{epochs}-n{neg_per_query}"
    done = (pd.read_parquet(CE_CV_SCORES_PATH) if CE_CV_SCORES_PATH.exists()
            else pd.DataFrame({"fold": pd.Series(dtype=int), "tag": pd.Series(dtype=str)}))
    for i in range(len(folds)):
        if ((done["fold"] == i) & (done["tag"] == tag)).any():
            print(f"re-ranker fold {i}: already done, skipping", flush=True)
            continue
        cool_down()
        rows = ce_training_pairs(cands[cands["fold"] != i], attrs, texts, gold, neg_per_query)
        print(f"re-ranker fold {i}: {len(rows)} training pairs ({int(rows['label'].sum())} positive)", flush=True)
        out = config.MODELS_DIR / f"ce_fold_{i}"
        fine_tune_reranker(rows, out, epochs, throttle_s=throttle_s)
        test = cands[cands["fold"] == i]
        ce = Reranker(model=CrossEncoder(str(out), max_length=CE_MAX_LENGTH, activation_fn=torch.nn.Identity(),
                                         model_kwargs={"torch_dtype": torch.float16}))
        scored = test[["key_a", "key_b"]].assign(
            ce_score=ce.score(list(zip(test["key_a"].map(texts), test["key_b"].map(texts)))), fold=i, tag=tag)
        done = pd.concat([done, scored], ignore_index=True)
        done.to_parquet(CE_CV_SCORES_PATH, index=False)
        print(f"re-ranker fold {i}: scored {len(scored)} held-out pairs", flush=True)
        del ce
        shutil.rmtree(out, ignore_errors=True)
        torch.cuda.empty_cache()
        time.sleep(FOLD_COOLDOWN_S)
    return done


def run_final(epochs=3, per_pair=3, throttle_s=THROTTLE_S):
    products = load_products()
    pairs = barcode_pairs(products)
    consum = products[products["chain_id"] == "consum"].reset_index(drop=True)
    base = Encoder(config.SIM["encoder_id"], config.SIM["encoder_revision"])
    triplets = mine_hard_negatives(pairs, products, _consum_index(base, consum), base, per_pair)
    return fine_tune(triplets, products, config.SIM["encoder_id"], config.SIM["encoder_revision"],
                     config.SIM["finetuned_path"], epochs, throttle_s=throttle_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--cv-reranker", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-pair", type=int, default=3)
    parser.add_argument("--throttle", type=float, default=THROTTLE_S, help="sleep per step (s); 0 = full speed")
    a = parser.parse_args()
    if a.cv:
        print(run_cv(a.epochs, a.per_pair, a.throttle).to_string(index=False))
    if a.final:
        print(run_final(a.epochs, a.per_pair, a.throttle))
    if a.cv_reranker:
        run_cv_reranker(throttle_s=a.throttle)
