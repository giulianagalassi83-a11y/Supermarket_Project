"""Settings for the normalization pipeline: paths, search terms, chains, model, thresholds."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "supermarkets.db"
CSV_PATH = DATA_DIR / "products_normalized.csv"
TAXONOMY_PATH = DATA_DIR / "taxonomy" / "product_types.csv"
LABELS_PATH = DATA_DIR / "labels" / "hand_labels.csv"
CACHE_PATH = DATA_DIR / "cache" / "embeddings.parquet"

# Same order as in the collectors (mercadona_api.ipynb keeps the first match in this order)
SEARCH_TERMS = [
    "leche", "huevos", "manzana", "plátano", "tomate", "arroz",
    "pasta", "pan", "pollo", "carne picada", "salmón", "yogur",
    "queso", "agua", "zumo", "atún", "garbanzos", "aceite",
    "detergente", "papel higiénico",
]

# Display names must match the store names used by dashboard/app.py
CHAINS = {
    "dia":       {"name": "Dia",       "collection_method": "selenium"},
    "mercadona": {"name": "Mercadona", "collection_method": "api"},
    "aldi":      {"name": "Aldi",      "collection_method": "selenium"},
    "alcampo":   {"name": "Alcampo",   "collection_method": "selenium"},
    "ahorramas": {"name": "Ahorramás", "collection_method": "selenium"},
    "bm":        {"name": "Bm",        "collection_method": "api"},
    "consum":    {"name": "Consum",    "collection_method": "api"},
}

RAW_FILES = {
    "dia": "dia_products_raw.csv",
    "mercadona_api": "mercadona_api_products_raw.csv",
    "mercadona_selenium": "mercadona_products_raw.csv",
    "aldi": "aldi_products_raw.csv",
    "alcampo": "alcampo_products_raw.csv",
    "ahorramas": "ahorramas_products_raw.csv",
    "bm": "bm_products_raw.csv",
    "consum": "consum_products_raw.csv",
}

# Historical RAW snapshots left out of the price history (chain_id, date) -> reason.
# The files stay on disk for the record; the pipeline never loads them.
EXCLUDED_SNAPSHOTS = {
    ("ahorramas", "2026-11-17"): "scraper read only the cents of each price (site markup change)",
    ("ahorramas", "2026-11-20"): "scraper read only the cents of each price (site markup change)",
}

# Which Mercadona collector to use: "api" or "selenium"
MERCADONA_SOURCE = "api"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TFIDF_NAME = "tfidf-char-3-5"

THRESHOLDS = {
    "embedding": {"tau": 0.35, "delta": 0.03},  # PLACEHOLDER — tune with evaluate.py
    "tfidf": {"tau": 0.15, "delta": 0.02},      # PLACEHOLDER — tune with evaluate.py
}

# A chain category containing any of these (lowercase) means the product is not food/cleaning we compare
EXCLUDED_CATEGORIES = [
    "mascota", "perro", "gato", "cosmética", "cuidado facial", "cuidado del cabello",
    "maquillaje", "bebé", "parafarmacia",
]

NO_RELEVANTE = "no_relevante"
SEED = 42

# Keyword rules applied after the classifier, for sibling types that TF-IDF character n-grams cannot tell
# apart (24 Nov: 207 of 315 "agua_con_gas" products were still water). Within `types`, a product whose
# accent-free name matches `pattern` gets `if_match`, any other gets `otherwise`.
KEYWORD_RULES = [
    {
        "types": ["agua_sin_gas", "agua_con_gas"],
        "pattern": r"con gas|gasificad|carbonatad|\bsoda\b|vichy|perrier|pellegrino|sifon|burbuja|sparkling",
        "if_match": "agua_con_gas",
        "otherwise": "agua_sin_gas",
    },
]

# ---------------------------------------------------------------- similarity engine (src/similarity)
SIM_DIR = DATA_DIR / "similarity"
SIM_RESULTS_DIR = SIM_DIR / "results"
SIM_CACHE_PATH = DATA_DIR / "cache" / "embeddings_f16.parquet"
SIM_ASSIGNMENTS_PATH = SIM_DIR / "assignments.csv"  # written by src.similarity.build, read by src.pipeline.build
SIM_MATCHES_PATH = SIM_DIR / "matches.csv"
SIM_CANONICAL_PATH = SIM_DIR / "canonical_products.csv"
MODELS_DIR = ROOT / "models"  # gitignored: fine-tuned models live on Felipe's laptop only

SIM = {
    "encoder_id": "BAAI/bge-m3",                     # bake-off winner, 14 Nov (MRR 0.806; results/bakeoff.csv)
    "encoder_revision": "5617a9f61b028005a4858fdac845db406aefb181",
    "reranker_id": "BAAI/bge-reranker-v2-m3",
    "reranker_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "finetuned_path": MODELS_DIR / "encoder_final",
    "top_k": 50,
    "precision_target": 0.90,   # same-product matches, barcode pairs (spec §9b)
    "equiv_precision_target": 0.85,  # equivalent level (Gemini silver, OOF): recall .79 vs .45 for the same-product decider at P .85 (18 Nov)
    "same_size_tol": 0.02,      # same product: size within ±2 %
    "strict_equiv_size_tol": 0.25,  # strict equivalent: smaller item ≥ 75 % of the bigger (up to 1.33×) — basket swaps
    "equiv_size_tol": 0.50,         # equivalent: smaller item ≥ 50 % of the bigger (up to 2×, 1 L ~ 1,5 L) — €/unit
    "db_pack_veto": False,  # DB pack counts never veto 'same': A/B 16 Nov, off = +1–2 pt recall at equal precision
    "high_confidence_t": 0.90,  # operating point 1: held-out P .95 / R .53; point 2 ("balanced") = precision_target rule
    "n_folds": 5,
    # Held-out fold of each search term in the `train --cv` run (15 Nov) whose fold models made the cached CV vectors.
    # GroupKFold reshuffles terms when the catalogue changes, so the held-out decider uses this fixed map instead
    # (recovered 22 Nov: each fold model scores its held-out terms' barcode pairs lowest; small terms checked
    # against the run's fold sizes 137/134/138/137/133).
    "cv_term_folds": {"agua": 0, "leche": 0, "salmón": 0, "aceite": 1, "tomate": 1, "zumo": 1, "plátano": 1,
                      "pasta": 2, "detergente": 2, "atún": 2, "huevos": 2,
                      "manzana": 3, "papel higiénico": 3, "pollo": 3, "arroz": 3, "queso": 4, "pan": 4, "yogur": 4},
    "judge_model": "gemini-3.8-flash",
    # Type classifier for src.similarity.build. Engine τ/δ tuned on 300 Gemini labels (python -m src.similarity.classify →
    # data/similarity/results/classification.csv). PROVISIONAL tfidf (18 Nov): the engine does not beat TF-IDF on the
    # 97 hand labels, and the candidate type filter used for silver/audit came from TF-IDF types — awaiting Felipe.
    # tfidf keeps the previous production τ .15 / δ .02: the Gemini-tuned τ .12 moved 524 products and Gemini judged
    # the old type right in 118 / new in 46 of 200 of them (ledger, 18 Nov).
    "classifier": "tfidf",
    "classifier_thresholds": {
        "tfidf": {"tau": 0.15, "delta": 0.02},
        "base": {"tau": 0.56, "delta": 0.005},
        "finetuned": {"tau": 0.19, "delta": 0.01},
    },
}
