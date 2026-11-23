"""LLM silver labels (offline only), cached to CSV so nothing is sent twice; plus the blind audit sample and κ.

Judge: Gemini (config.SIM["judge_model"]). The key is read from GEMINI_API_KEY (environment or the gitignored
.env) and never stored.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.pipeline import config

PAIR_LABELS = ["same", "equivalent", "different"]

# Level definitions copied from the spec (§6.2); the "up to 2×" follows config.SIM["equiv_size_tol"] = 0.50.
# Strict vs loose equivalence is derived from sizes afterwards (attributes.equivalence_tier), not judged.
PAIR_PROMPT = """You compare two supermarket products from different Spanish chains.

Levels:
- same: same brand (or both own-brand of the same chain), same variant, size within ±2 %, same pack count.
- equivalent: a fair equivalent: same type of product, no variant conflict (e.g. whole vs skimmed milk, \
extra virgin vs refined oil), per-item size at most 2× apart (e.g. 1 L vs 1,5 L or 500 g vs 1 kg is fine; prices \
are compared per unit, so a multipack of the same item counts), brand may differ (national vs own brand allowed).
- different: anything else.

Product A: {a}
Product B: {b}

Answer with the level and a one-sentence rationale."""

TYPE_PROMPT = """Classify this supermarket product into exactly one of the allowed types.

Product: {product}

Allowed types (id: description):
{types}

Use "no_relevante" when the product is not any of the listed types."""


def _schema(labels):
    return {"type": "object",
            "properties": {"label": {"type": "string", "enum": labels}, "rationale": {"type": "string"}},
            "required": ["label", "rationale"]}


def _client(client):
    if client is not None:
        return client
    import os

    from dotenv import dotenv_values
    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or dotenv_values(config.ROOT / ".env").get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No Gemini credentials: set GEMINI_API_KEY in the environment or .env (kept out of the repo).")
    return genai.Client(api_key=key)


def _ask(client, prompt, labels):
    """(label, rationale); (None, reason) when the reply is not valid JSON or the label is unknown."""
    from google.genai import types

    try:
        response = client.models.generate_content(
            model=config.SIM["judge_model"], contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_json_schema=_schema(labels),
                thinking_config=types.ThinkingConfig(thinking_level="low"), temperature=0),
        )
    except Exception as exc:  # network / rate limit: leave the row unlabelled so the next run retries it
        return None, f"error: {type(exc).__name__}: {str(exc)[:120]}"
    try:
        reply = json.loads(response.text or "")
    except (json.JSONDecodeError, ValueError):
        return None, "invalid json"
    label = reply.get("label")
    return (label, reply.get("rationale", "")) if label in labels else (None, f"unknown label {label!r}")


def _cached_run(todo, cache_path, key_cols, make_prompt, labels_for, client, workers=8):
    cache = pd.read_csv(cache_path, dtype=str) if cache_path.exists() else pd.DataFrame(columns=key_cols + ["label", "rationale", "model"])
    done = cache[cache["label"].notna()]
    seen = set(map(tuple, done[key_cols].to_numpy()))
    pending = [row for row in todo[key_cols].itertuples(index=False, name=None) if row not in seen]
    if pending:
        client = _client(client)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers = list(pool.map(lambda r: _ask(client, make_prompt(r), labels_for(r)), pending))
        new = pd.DataFrame(pending, columns=key_cols)
        new["label"] = [a[0] for a in answers]
        new["rationale"] = [a[1] for a in answers]
        new["model"] = config.SIM["judge_model"]
        cache = pd.concat([done, new], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(cache_path, index=False)
    wanted = todo[key_cols].astype(str)
    return wanted.merge(cache.astype({c: str for c in key_cols}), on=key_cols, how="left")


def judge_pairs(pairs, texts, client=None, cache_path=config.SIM_DIR / "llm_pair_labels.csv"):
    """Label each (key_a, key_b) as same / equivalent / different. `texts`: key → 'name | brand | size' text."""
    return _cached_run(pairs, cache_path, ["key_a", "key_b"],
                       lambda r: PAIR_PROMPT.format(a=texts[r[0]], b=texts[r[1]]), lambda r: PAIR_LABELS, client)


def judge_types(products, hits, taxonomy, client=None, cache_path=config.SIM_DIR / "llm_type_labels.csv"):
    """Type label per product, constrained to its search terms' real types + no_relevante."""
    real = taxonomy[taxonomy["is_negative"] == 0]
    options = hits.merge(real[["type_id", "search_term", "description"]], on="search_term")
    by_product = options.groupby("product_key").apply(lambda d: list(zip(d["type_id"], d["description"])))
    name = products.set_index("product_key")["name"]

    def prompt(r):
        types = "\n".join(f"{t}: {d}" for t, d in by_product.get(r[0], [])) + f"\n{config.NO_RELEVANTE}: none of these"
        return TYPE_PROMPT.format(product=name[r[0]], types=types)

    def labels(r):
        return [t for t, _ in by_product.get(r[0], [])] + [config.NO_RELEVANTE]

    out = _cached_run(products[["product_key"]], cache_path, ["product_key"], prompt, labels, client)
    return out.rename(columns={"label": "type_id"})


AUDIT_QUOTAS = {"high": 0.25, "balanced": 0.30, "rejected": 0.25, "barcode_error": 0.20}


def sample_audit(silver, n=100, seed=config.SEED, quotas=AUDIT_QUOTAS):
    """Blind audit sample from silver_candidates + LLM labels: `quotas` shares of engine-high, engine-balanced,
    engine-rejected (other chains) and barcode_error pairs; inside each group round-robin over (chain_a, tier) so
    every chain and both size tiers show up. Returns the sample with its `group`; the caller hides it from raters."""
    d = silver.copy()
    d["group"] = np.where(d["source"] == "barcode_error", "barcode_error",
                          d["accepted"].map({"high": "high", "balanced": "balanced", "no": "rejected"}))
    d = d.sample(frac=1, random_state=seed)
    d["turn"] = d.groupby(["group", "chain_a", "tier"]).cumcount()
    d = d.sort_values("turn", kind="stable")
    sizes = {g: int(round(q * n)) for g, q in quotas.items()}
    sizes[max(sizes, key=sizes.get)] += n - sum(sizes.values())
    picked = pd.concat([d[d["group"] == g].head(k) for g, k in sizes.items()])
    return picked.drop(columns="turn").reset_index(drop=True)


def kappa(a, b):
    """Cohen's κ between two label series aligned by index."""
    both = pd.concat([a, b], axis=1, join="inner").dropna()
    return float(cohen_kappa_score(both.iloc[:, 0], both.iloc[:, 1]))
