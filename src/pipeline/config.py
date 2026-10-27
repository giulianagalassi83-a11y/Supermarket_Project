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
    "alcampo": "alcampo_multiple_searches.csv",
    "ahorramas": "ahorramas_products_raw.csv",
    "bm": "bm_products_raw.csv",
    "consum": "consum_products_raw.csv",
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
