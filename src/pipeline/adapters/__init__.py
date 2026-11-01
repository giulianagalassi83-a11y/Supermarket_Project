"""One adapter per raw source. Each turns a raw CSV into the COMMON_COLUMNS layout."""

from src.pipeline import config
from src.pipeline.adapters import apis, scraped

LOADERS = {
    "dia": scraped.load_dia,
    "aldi": scraped.load_aldi,
    "alcampo": scraped.load_alcampo,
    "ahorramas": scraped.load_ahorramas,
    "mercadona_selenium": scraped.load_mercadona_selenium,
    "mercadona_api": apis.load_mercadona_api,
    "bm": apis.load_bm,
    "consum": apis.load_consum,
}


def load_source(source, raw_dir):
    """Load one raw source ('dia', 'mercadona_api', ...) from `raw_dir`."""
    return LOADERS[source](raw_dir / config.RAW_FILES[source])


def source_for(chain_id):
    """Mercadona has two collectors; config.MERCADONA_SOURCE picks one."""
    return f"mercadona_{config.MERCADONA_SOURCE}" if chain_id == "mercadona" else chain_id


def load_chain(chain_id, raw_dir):
    return load_source(source_for(chain_id), raw_dir)
