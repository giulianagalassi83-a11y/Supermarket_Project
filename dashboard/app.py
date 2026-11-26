# Main project dashboard (the old, deprecated version is dashboard/app_legacy.py).
# TO RUN IT:  streamlit run dashboard/app.py --theme.primaryColor "#17242D" --theme.base light
#   (it also works without the flags; the code applies the theme from the first load)
#
# How it is organized:
#   - Top navigation (Basket, Compare, Map, Trends, Analysis); no sidebar.
#   - The basket is the centre: you search the 70 product types, not the 20 search terms.
#   - Compare joins price tags per chain, the product x chain matrix, the route and two charts.
#   - Trends and Analysis read src/analysis/history.py and src/analysis/insights.py;
#     their charts are in dashboard/charts.py.
#   - The optimizer is src/analysis/optimizer.py.
#   - If Overpass or Nominatim fail, the app uses data/stores_madrid.csv and a saved address.
#   - One color per chain everywhere. Product names stay in Spanish, as the chains sell them.

import base64
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import re
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

ROOT_DIR = Path(__file__).resolve().parents[1]
LOGO_BASKIT = ROOT_DIR / "assets" / "baskit_2.png"
FAVICON_BASKIT = ROOT_DIR / "assets" / "b.png"
sys.path.insert(0, str(ROOT_DIR))
from dashboard import charts  # noqa: E402
from src.analysis import history, insights  # noqa: E402
from src.analysis import optimizer as opt  # noqa: E402
from src.analysis import prices as pricing  # noqa: E402
from src.analysis import matched  # noqa: E402

# Theme for this app only (a .streamlit/config.toml would also change app_legacy.py)
from streamlit import config as st_config  # noqa: E402

for option, value in {
    "theme.base": "light",
    "theme.primaryColor": "#13301F",
    "theme.backgroundColor": "#F4EFE4",
    "theme.secondaryBackgroundColor": "#FFFFFF",
    "theme.textColor": "#1A211E",
}.items():
    st_config.set_option(option, value)

st.set_page_config(page_title="Baskit", page_icon=str(FAVICON_BASKIT) if FAVICON_BASKIT.exists() else "🛒", layout="wide")


# ============================================================
# STYLE: one place for colors, type and components
# ============================================================

CHAIN_COLORS = {
    "Mercadona": "#0F8B4C",
    "Dia": "#D7192D",
    "Aldi": "#1B3A8C",
    "Alcampo": "#E6730F",
    "Ahorramás": "#7A3E9D",
    "Bm": "#0E9AA7",
    "Consum": "#B8930B",
}
CHAIN_NAMES = {"Bm": "BM"}  # how each chain is written on screen
INK, MUTED, BEST, TAG = "#17242D", "#5F6E78", "#12805C", "#FFD60A"

st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,300..900&display=swap');
:root{
    --shelf:#F4EFE4;
    --surface:#FFFFFF;
    --ink:#26332D;
    --muted:#64736B;
    --line:#E2DACC;
    --tag:#F2A65A;
    --best:#13301F;
    --best-bg:#E8E3D8;
    --accent:#F2A65A;
}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:var(--shelf);color:var(--ink);font-family:'Archivo',sans-serif;}
.stApp *:not([data-testid="stIconMaterial"]),[role="dialog"] *:not([data-testid="stIconMaterial"]){font-family:'Archivo',sans-serif;}
span[data-baseweb="tag"]{
    background:#2E7D5B!important;
    color:#fff!important;
}

.block-container{padding-top:4.2rem;max-width:1240px;}
h1,h2,h3{font-family:'Archivo',sans-serif;font-stretch:112%;font-weight:750;color:var(--ink);letter-spacing:-0.01em;}
.hero{font-stretch:108%;font-weight:760;font-size:clamp(1.7rem,3.4vw,2.7rem);line-height:1.08;margin:0.2rem 0 0.4rem;max-width:26ch;}
.lede{color:var(--muted);font-size:1.05rem;max-width:62ch;margin-bottom:1.2rem;}
.price{font-stretch:70%;font-weight:850;font-variant-numeric:tabular-nums;line-height:0.9;white-space:nowrap;}
.price .eu{font-size:3.3rem;} .price .ct{font-size:1.45rem;vertical-align:1.55rem;margin-left:1px;} .price .cur{font-size:1.3rem;margin-right:3px;}
.price.sm .eu{font-size:2.2rem;} .price.sm .ct{font-size:1rem;vertical-align:1rem;} .price.sm .cur{font-size:0.95rem;}
.tags{display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:14px;margin:0.6rem 0 1.6rem;}
.tag{background:var(--surface);border:1px solid var(--line);border-left:7px solid var(--chain);border-radius:4px 10px 10px 4px;padding:14px 16px 12px;position:relative;}
.tag.win{background:#FFE599;border-color:#E9BE00;border-left-color:var(--chain);box-shadow:0 6px 18px -10px rgba(120,90,0,.55);}
.tag .chain{font-stretch:115%;font-weight:760;font-size:1.05rem;}
.tag .badge{position:absolute;top:12px;right:12px;font-size:.78rem;font-weight:700; background:#2E7D5B; color:white; padding:2px 8px; border-radius:20px;}
.tag .diff{font-weight:650;font-size:.95rem;color:#A23B2A;margin:2px 0 8px;}
.tag.win .diff{color:var(--ink);}
.tag .meta{display:flex;justify-content:space-between;font-size:.88rem;color:var(--muted);border-top:1px dashed rgba(23,36,45,.25);padding-top:7px;margin-top:8px;}
.tag.win .meta{color:var(--ink);}
.tag.partial{opacity:.72;} .tag.partial .diff{color:var(--muted);}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--chain);margin-right:6px;vertical-align:1px;}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 20px;}
.matrix-wrap{overflow-x:auto;background:#ffe6cc;border:1px solid #D8D0C2;border-radius:12px;}
.offer-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:12px 0 8px 0;}
.offer-card{background:#ffe6cc;border:1px solid #D8D0C2;border-radius:15px;padding:19px 20px;min-width:0;box-shadow:0 2px 8px rgba(23,36,45,0.035);}
.offer-chain{display:inline-block;color:#FFFFFF;font-size:0.82rem;font-weight:750;padding:6px 11px;border-radius:7px;margin-bottom:13px;background:var(--chain-color);}
.offer-number{color:#1E5736;font-size:2.15rem;line-height:1.1;font-weight:800;letter-spacing:-0.06em;margin-bottom:5px;}
.offer-label{color:#1E5736;font-size:0.88rem;font-weight:700;}
.offer-detail{color:#1E5736;opacity:0.75;font-size:0.76rem;margin-top:5px;}
.offer-divider{height:1px;background:#D8D0C2;margin:15px 0 11px 0;}
.offer-discount{color:#1E5736;font-size:0.78rem;font-weight:650;}
@media (max-width:700px){.offer-grid{grid-template-columns:1fr;}}
table.matrix{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.95rem;background:transparent;}
table.matrix th,table.matrix td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap;}
table.matrix th{font-stretch:110%;font-weight:700;font-size:.9rem;background:transparent;}
table.matrix th:first-child,
table.matrix td:first-child{text-align:left;position:sticky;left:0;background:#ffe6cc;z-index:1;}
table.matrix th:first-child{background:#ffe6cc;}
table.matrix td .u{display:block;color:var(--muted);font-size:.78rem;}
table.matrix td.best span.v{background:#1e5736;color:#FFFFFF;font-weight:750;padding:3px 8px;border-radius:5px;}
table.matrix td.none{color:#A8B3B9;}
table.matrix tr.total td{font-weight:800;font-stretch:90%;font-size:1.05rem;border-bottom:none;}
.route{list-style:none;padding:0;margin:.4rem 0 0;}
.route li{position:relative;padding:0 0 14px 26px;}
.route li:before{content:"";position:absolute;left:6px;top:8px;bottom:-6px;border-left:2px dashed #B9C4C9;}
.route li:last-child:before{display:none;}
.route li .stop{position:absolute;left:0;top:3px;width:14px;height:14px;border-radius:50%;background:var(--chain);border:2px solid #fff;box-shadow:0 0 0 1px var(--chain);}
.route li .km{color:var(--muted);font-size:.85rem;}
.verdict{font-size:1.15rem;font-weight:700;font-stretch:105%;margin-bottom:.3rem;}
.small{color:var(--muted);font-size:.88rem;}
.item-row{display:flex;align-items:baseline;gap:10px;}
.item-row .name{font-weight:700;font-size:1.05rem;}
.from{color:var(--muted);font-size:.9rem;}
.from b{color:var(--best);}
.store-row b{white-space:nowrap;}
.store-row{display:flex;gap:12px;justify-content:space-between;align-items:center;padding:10px 12px;border-bottom:1px solid var(--line);}
.store-row.sel{background:#FFF6C2;}
.store-row.in-route {
    background: #FFE6CC;
    border-left: 4px solid #F0883E;
}
.store-row .addr{color:var(--muted);font-size:.82rem;}
.bar{height:8px;border-radius:4px;background:var(--chain);}
.pl{display:inline-block;font-size:.72rem;font-weight:700;padding:1px 6px;border-radius:4px;background:#E7ECEF;color:var(--ink);margin-left:6px;}
/* Main buttons */
.stButton>button{
    background:#13301F !important;
    border:1px solid #13301F !important;
    color:white !important;
}

.stButton>button:hover{
    background:#24523A !important;
    border-color:#24523A !important;
    color:white !important;
}

.stButton>button[kind="primary"]{
    background:#13301F !important;
    border-color:#13301F !important;
    color:white !important;
    font-weight:700;
    font-stretch:110%;
    padding:.7rem 1.6rem;
    font-size:1.05rem;
}

/* ============================================================
   TOP NAVIGATION
   ============================================================ */

/* Dark green navigation bar */
[data-testid="stTopNav"],
[data-testid="stTopNav"] > div,
[data-testid="stTopNav"] > div > div {
    background: #13301F !important;
    background-color: #13301F !important;
}

/* Navigation links */
a[data-testid="stTopNavLink"] {
    background: transparent !important;
    color: #F0883E !important;
    border-radius: 6px !important;
}

/* Navigation labels */
a[data-testid="stTopNavLink"] p,
a[data-testid="stTopNavLink"] span {
    color: #F0883E !important;
}

/* Navigation icons */
a[data-testid="stTopNavLink"] [data-testid="stIconMaterial"] {
    color: #F0883E !important;
    -webkit-text-fill-color: #F0883E !important;
}

/* Hover effect */
a[data-testid="stTopNavLink"]:hover {
    background: #24523A !important;
    background-color: #24523A !important;
}

/* Selected page */
a[data-testid="stTopNavLink"][aria-current="page"] {
    background: #24523A !important;
    background-color: #24523A !important;
    border-radius: 6px !important;
    box-shadow: inset 0 -3px 0 #F0883E !important;
}

/* Selected page: text and icon */
a[data-testid="stTopNavLink"][aria-current="page"] p,
a[data-testid="stTopNavLink"][aria-current="page"] span,
a[data-testid="stTopNavLink"][aria-current="page"] [data-testid="stIconMaterial"] {
    color: #F0883E !important;
    -webkit-text-fill-color: #F0883E !important;
}

/* Streamlit header */
header[data-testid="stHeader"],
[data-testid="stAppHeader"] {
    background: #13301F !important;
    background-color: #13301F !important;
}

/* Streamlit top-right controls: Stop, Deploy and main menu */
[data-testid="stAppDeployButton"],
[data-testid="stToolbar"],
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] span,
[data-testid="stToolbar"] p,
[data-testid="stToolbar"] [data-testid="stIconMaterial"] {
    color: #F0883E !important;
    -webkit-text-fill-color: #F0883E !important;
}

/* Color only visible SVG strokes/shapes without filling transparent bounding boxes */
[data-testid="stToolbar"] svg path:not([fill="none"]),
[data-testid="stToolbar"] svg circle:not([fill="none"]),
[data-testid="stToolbar"] svg rect:not([fill="none"]) {
    fill: #F0883E !important;
}
[data-testid="stToolbar"] svg [fill="none"] {
    fill: none !important;
}

/* Calculation settings background */
div[data-testid="stExpander"] {
    background-color: #F4EFE4 !important;
    border: 1px solid #E2DACC !important;
    border-radius: 10px !important;
}

div[data-testid="stExpander"] details,
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background-color: #F4EFE4 !important;
}


/* Location popover */
[data-testid="stPopover"] button {
    background-color: #FFFFFF !important;
    color: #13301F !important;
    border: 1px solid #E2DACC !important;
}

[data-testid="stPopover"] button:hover {
    background-color: #E8E3D8 !important;
    color: #13301F !important;
}

[data-testid="stPopover"] [data-testid="stPopoverBody"] {
    background-color: #FFFFFF !important;
}

/* White input fields and visible numbers */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
}

/* Quantity numbers */
.stNumberInput input {
    color: #26332D !important;
    -webkit-text-fill-color: #26332D !important;
    background-color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Quantity buttons */
.stNumberInput button {
    background-color: #FFFFFF !important;
    color: #13301F !important;
    border-color: #E2DACC !important;
}

/* Basket product search */
.stMultiSelect div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #E2DACC !important;
}

/* Search text */
.stMultiSelect input {
    background-color: #FFFFFF !important;
    color: #26332D !important;
    -webkit-text-fill-color: #26332D !important;
}

/* Selected basket products: orange */
div[data-testid="stMultiSelect"] span[data-baseweb="tag"],
div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: #F0883E !important;
    background-color: #F0883E !important;
    border: 1px solid #F0883E !important;
    color: #FFFFFF !important;
}

/* Selected product text and remove icon */
div[data-testid="stMultiSelect"] span[data-baseweb="tag"] *,
div[data-testid="stMultiSelect"] [data-baseweb="tag"] * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* Search dropdown */
[data-baseweb="popover"] {
    background-color: #FFFFFF !important;
}

[data-baseweb="popover"] li {
    background-color: #FFFFFF !important;
    color: #26332D !important;
}

[data-baseweb="popover"] li:hover {
    background-color: #F4EFE4 !important;
}

/* Cream application background */
html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #F4EFE4 !important;
}

@media (max-width:640px){.price .eu{font-size:2.6rem;}.block-container{padding-left:16px;padding-right:16px;}}
</style>""",
    unsafe_allow_html=True,
)


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def chain_name(chain):
    return CHAIN_NAMES.get(chain, chain)


def eur(x, decimals=2):
    """6.68 -> '€6.68'."""
    return f"€{x:,.{decimals}f}"


def km_text(x):
    return f"{x:,.1f} km" if x >= 1 else f"{x * 1000:.0f} m"


def price_tag(x, small=False):
    """A price like a shelf label: big euros, small cents above."""
    euros, cents = f"{x:.2f}".split(".")
    return (
        f'<div class="price{" sm" if small else ""}"><span class="cur">€</span><span class="eu">{euros}</span>'
        f'<span class="ct">.{cents}</span></div>'
    )


# ============================================================
# PRODUCT DATA
# ============================================================

UNIT_TEXT = {"LITRO": "l", "KILO": "kg", "UNIDAD": "unit", "LAVADO": "wash"}
DEFAULT_BASKET = {
    "leche_semidesnatada": 2.0,
    "aceite_oliva_virgen_extra": 1.0,
    "huevos_gallina": 12.0,
    "pasta_larga": 1.0,
    "yogur_natural": 0.5,
}


@st.cache_data
def load_data():
    return pricing.load_normalized()


try:
    df = load_data()
except FileNotFoundError:
    st.error("The normalized data is missing. From the project folder, run: python -m src.pipeline.build")
    st.stop()

usable_df = pricing.usable(df)
TYPES = (
    usable_df.groupby("product_type")
    .agg(label=("product_type_label", "first"), unit=("unit", lambda s: s.mode().iat[0]))
    .sort_values("label")
)
CHAINS = sorted(usable_df["supermarket"].unique())


def label(type_id):
    return TYPES.at[type_id, "label"]


def unit(type_id):
    return UNIT_TEXT.get(TYPES.at[type_id, "unit"], TYPES.at[type_id, "unit"].lower())


@st.cache_data
def unit_prices(type_ids, stat):
    """Types × chains, € per unit (median or cheapest)."""
    return pricing.price_table(df, list(type_ids), stat).reindex(index=list(type_ids), columns=CHAINS)

@st.cache_data
def unit_counts(type_ids):
    """Types × chains table of comparable product counts."""
    return (
        pricing.price_count_table(df, list(type_ids))
        .reindex(index=list(type_ids), columns=CHAINS)
        .fillna(0)
        .astype(int)
    )

# ============================================================
# STATE SHARED BETWEEN PAGES
# ============================================================
# Streamlit clears the widget state of a page that isn't showing; reassigning it at the start of
# each run keeps it (the basket, quantities and settings travel between pages).

DEFAULTS = {
    "basket": list(DEFAULT_BASKET),
    "address": "Avenida de la Universidad, 30, Leganés",
    "radius": 1.5,
    "cost_per_km": 0.25,
    "cost_per_stop": 0.0,
    "value_of_time": 10.0,
    "speed_kmh": 25.0,
    "min_per_stop": 5.0,
    "max_stores": 3,
    "stat": "Typical price",
    "sort": "Price",
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
for type_id, qty in DEFAULT_BASKET.items():
    st.session_state.setdefault(f"qty_{type_id}", qty)
for key in list(st.session_state):
    if key in DEFAULTS or key.startswith("qty_"):
        st.session_state[key] = st.session_state[key]

basket = [t for t in st.session_state.basket if t in TYPES.index]
for type_id in basket:
    st.session_state.setdefault(f"qty_{type_id}", 1.0 if TYPES.at[type_id, "unit"] != "UNIDAD" else 6.0)
quantities = pd.Series({t: st.session_state[f"qty_{t}"] for t in basket}, dtype=float)
STAT = "min" if st.session_state.stat == "Cheapest" else "median"

def time_params():
    """Time settings, passed to the optimizer in every call to score / greedy_plan / sensitivity."""
    return dict(
        value_of_time=st.session_state.value_of_time,
        speed_kmh=st.session_state.speed_kmh,
        min_per_stop=st.session_state.min_per_stop,
    )


# ============================================================
# LOCATION AND STORES (with a fallback if the APIs fail)
# ============================================================

SAVED_ADDRESSES = {"avenida de la universidad, 30, leganés": (40.3326, -3.7651, "Av. de la Universidad 30, Leganés")}
HEADERS = {"User-Agent": "UC3M-Supermarket-Project/1.0"}


@st.cache_data(ttl=3600, show_spinner=False)
def geocode(address):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "jsonv2", "limit": 1, "countrycodes": "es"},
            headers=HEADERS, timeout=10,
        )
        r.raise_for_status()
        hits = r.json()
        if hits:
            short = ", ".join(hits[0]["display_name"].split(", ")[:3])
            return float(hits[0]["lat"]), float(hits[0]["lon"]), short, False
        return None
    except Exception:
        saved = SAVED_ADDRESSES.get(address.strip().lower())
        return (*saved, True) if saved else "error"


def chain_of(name):
    name = str(name).lower().replace("á", "a")
    for pattern, chain in [
        ("mercadona", "Mercadona"), ("ahorramas", "Ahorramás"), ("alcampo", "Alcampo"),
        (r"\baldi\b", "Aldi"), (r"\bdia\b", "Dia"), (r"\bbm\b", "Bm"), ("consum", "Consum"),
    ]:
        if re.search(pattern, name):
            return chain
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_stores(lat, lon, box=0.15):
    """OpenStreetMap supermarkets around the point; if Overpass fails, the saved list."""
    query = f"""[out:json][timeout:20];
    (nwr["shop"="supermarket"]["brand"~"Mercadona|DIA|ALDI|Alcampo|Ahorram|BM|Consum",i]({lat-box},{lon-box},{lat+box},{lon+box});
     nwr["shop"="supermarket"]["name"~"Mercadona|DIA|ALDI|Alcampo|Ahorram|BM|Consum",i]({lat-box},{lon-box},{lat+box},{lon+box}););
    out center tags;"""
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data=query.encode("utf-8"), headers=HEADERS, timeout=20)
        r.raise_for_status()
        rows = []
        for e in r.json().get("elements", []):
            tags = e.get("tags", {})
            name = tags.get("name") or tags.get("brand")
            chain = chain_of(name)
            point = e if "lat" in e else e.get("center")
            if chain is None or point is None:
                continue
            address = ", ".join(tags[k] for k in ["addr:street", "addr:housenumber", "addr:city"] if tags.get(k))
            rows.append({"supermarket": chain, "name": name, "address": address, "lat": point["lat"], "lon": point["lon"]})
        stores = pd.DataFrame(rows).drop_duplicates(["supermarket", "lat", "lon"])
        if not stores.empty:
            return stores, False
    except Exception:
        pass
    saved = pd.read_csv(ROOT_DIR / "data" / "stores_madrid.csv", encoding="utf-8-sig")
    saved["address"] = saved["address"].fillna("")
    near = saved[saved["lat"].between(lat - box, lat + box) & saved["lon"].between(lon - box, lon + box)]
    return near.reset_index(drop=True), True

st.session_state.setdefault(
    "last_valid_location",
    (40.3326, -3.7651, "Av. de la Universidad 30, Leganés", True, "Avenida de la Universidad, 30, Leganés"),
)
st.session_state.setdefault("address_warning", None)

if not st.session_state.get("_app_booted", False):
    loader = st.empty()
    if LOGO_BASKIT.exists():
        b64 = base64.b64encode(LOGO_BASKIT.read_bytes()).decode()
        logo_tag = f'<img src="data:image/jpeg;base64,{b64}" style="width: 540px; max-width: 75vw; height: auto;" />'
    else:
        logo_tag = '<div style="font-size: 4rem; font-weight: 800; color: #13301F;">baskit<span style="color: #F0883E;">.</span></div>'

    loader.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;
                    min-height: 62vh; text-align: center; gap: 16px;">
            {logo_tag}
            <div style="font-size: 1rem; color: #64736B; font-weight: 500;">
                Finding supermarkets near you…
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    location = geocode(st.session_state.address)
    loader.empty()
    st.session_state["_app_booted"] = True
else:
    location = geocode(st.session_state.address)

def _in_madrid(lat, lon):
    return 39.85 <= float(lat) <= 41.20 and -4.60 <= float(lon) <= -3.05

prev_lat, prev_lon, prev_short, prev_offline, prev_addr = st.session_state["last_valid_location"]

if location == "error":
    st.session_state["address_warning"] = (
        f"The address service is not responding. Keeping your previous address ({prev_addr})."
    )
    st.session_state["address"] = prev_addr
    user_lat, user_lon, _, offline_geo = prev_lat, prev_lon, prev_short, prev_offline
elif location is None:
    st.session_state["address_warning"] = (
        f'We could not find "{st.session_state.address}". Enter street, number and town in Madrid '
        f'(for example "Calle Mayor 1, Madrid"). Keeping your previous address ({prev_addr}).'
    )
    st.session_state["address"] = prev_addr
    user_lat, user_lon, _, offline_geo = prev_lat, prev_lon, prev_short, prev_offline
elif not _in_madrid(location[0], location[1]):
    st.session_state["address_warning"] = (
        f'"{st.session_state.address}" is outside Madrid (only Madrid stores are supported). '
        f"Keeping your previous address ({prev_addr})."
    )
    st.session_state["address"] = prev_addr
    user_lat, user_lon, _, offline_geo = prev_lat, prev_lon, prev_short, prev_offline
else:
    user_lat, user_lon, short_name, offline_geo = location
    st.session_state["last_valid_location"] = (
        user_lat, user_lon, short_name, offline_geo, st.session_state.address
    )
    st.session_state["address_warning"] = None

HOME = (user_lat, user_lon)
all_stores, offline_stores = load_stores(user_lat, user_lon)

candidates = opt.candidate_stores(all_stores, HOME, per_chain=3, max_km=st.session_state.radius)
nearest = candidates.drop_duplicates("supermarket")  # the nearest store of each chain


# ============================================================
# BASKET CALCULATIONS (shared by every page)
# ============================================================

ppu = unit_prices(tuple(basket), STAT) if basket else pd.DataFrame(columns=CHAINS)
cost = ppu.mul(quantities, axis=0)  # € of each product in the quantity asked for


def chain_cards():
    """One row per chain with a store in range: basket cost, coverage and cost including the trip."""
    rows = []
    for _, store in nearest.iterrows():
        chain = store["supermarket"]
        prices = cost[chain].dropna() if chain in cost else pd.Series(dtype=float)
        trip_km = 2 * store["distance_km"] * opt.DETOUR
        complete = len(prices) == len(basket)
        rows.append({
            "chain": chain, "store": store["name"], "address": store["address"],
            "distance_km": store["distance_km"], "basket_eur": prices.sum() if len(prices) else np.nan,
            "have": len(prices), "complete": complete,
            "balanced_eur": prices.sum() + trip_km * st.session_state.cost_per_km if complete else np.nan,
        })
    cards = pd.DataFrame(rows)
    if cards.empty:
        return cards
    key = {"Price": "basket_eur", "Distance": "distance_km", "Balance": "balanced_eur"}[st.session_state.sort or "Price"]
    return cards.sort_values(["complete", key], ascending=[False, True]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def plans_for(cost, stores, home, max_stores):
    return opt.enumerate_plans(cost, stores, home, max_stores)


# SHARED HEADER: address and basket always at hand

def top_bar_logo():
    if not FAVICON_BASKIT.exists():
        return

    b64 = base64.b64encode(FAVICON_BASKIT.read_bytes()).decode()

    components.html(
        f"""
        <script>
        (function() {{
            const doc = window.parent.document;

            let icon = doc.getElementById('baskit-navbar-icon');

            if (!icon) {{
                icon = doc.createElement('img');
                icon.id = 'baskit-navbar-icon';
                icon.src = 'data:image/png;base64,{b64}';
                icon.alt = 'baskit';
                icon.style.height = '22px';
                icon.style.width = '22px';
                icon.style.objectFit = 'contain';
                icon.style.borderRadius = '4px';
                icon.style.margin = '0 10px';
                icon.style.flexShrink = '0';
            }}

            // Locate the Deploy button.
            const deploy = doc.querySelector('[data-testid="stAppDeployButton"]')
                || Array.from(doc.querySelectorAll('button'))
                    .find(el => el.innerText.trim() === 'Deploy');

            if (!deploy || !deploy.parentElement) return;

            // Move the existing logo immediately before Deploy.
            deploy.parentElement.insertBefore(icon, deploy);

        }})();
        </script>
        """,
        height=0,
    )


def header():
    left, mid, right, logo = st.columns([3.3, 4.7, 2, 1.8], vertical_alignment="center")
    left.markdown(
        '<div style="font-stretch:118%;font-weight:800;font-size:1.12rem;white-space:nowrap;">Supermarket comparison</div>',
        unsafe_allow_html=True,
    )
    top_bar_logo()
    with mid.popover(f"Shopping near {st.session_state.address}, within {km_text(st.session_state.radius)}", width="stretch"):
        st.text_input("Address", key="address")
        st.slider("Maximum distance to the store (km)", 0.3, 5.0, step=0.1, key="radius")
    with right.popover(f"Basket ({len(basket)})", icon=":material/shopping_basket:", width="stretch"):
        if not basket:
            st.write("Your basket is empty.")
        for t in basket:
            st.markdown(f"**{label(t)}** · {st.session_state[f'qty_{t}']:g} {unit(t)}")
    with logo:
        if LOGO_BASKIT.exists():
            mime = "image/png" if LOGO_BASKIT.suffix.lower() == ".png" else "image/jpeg"
            logo_b64 = base64.b64encode(LOGO_BASKIT.read_bytes()).decode()
            st.markdown(
                f"""
                <div style="
                    position: fixed;
                    top: 70px;
                    right: 20px;
                    z-index: 90;
                ">
                    <img src="data:{mime};base64,{logo_b64}" style="width:88px;height:auto;display:block;" />
                </div>
                """,
                unsafe_allow_html=True,
            )
    if st.session_state.get("address_warning"):
        st.warning(st.session_state.pop("address_warning"))
    if offline_geo:
        st.caption("The address service is not responding: using the saved location for this address.")
    if offline_stores:
        st.caption("OpenStreetMap is not responding: using the saved Madrid store list in data/stores_madrid.csv.")

# ============================================================
# PRODUCT SHEET (pop-up window)
# ============================================================

@st.dialog("Product sheet", width="large")
def product_sheet(type_id):
    rows = pricing.type_rows(df, type_id)
    st.markdown(f"## {label(type_id)}")
    st.caption(f"{len(rows)} products in {rows['supermarket'].nunique()} chains · prices per {unit(type_id)}")

    by_chain = (
        rows.groupby("supermarket")["price_per_unit"]
        .agg(["median", "min", "size"]).sort_values("median")
    )
    top = by_chain["median"].max()
    lines = []
    for chain, r in by_chain.iterrows():
        width = 100 * r["median"] / top
        lines.append(
            f'<div style="display:grid;grid-template-columns:120px 1fr 210px;gap:12px;align-items:center;margin:7px 0;--chain:{CHAIN_COLORS.get(chain, INK)}">'
            f'<div><span class="dot"></span><b>{chain_name(chain)}</b></div>'
            f'<div><div class="bar" style="width:{width:.0f}%"></div></div>'
            f'<div style="text-align:right"><b>{eur(r["median"])}/{unit(type_id)}</b>'
            f'<div class="small">from {eur(r["min"])}, {int(r["size"])} products</div></div></div>'
        )
    html("".join(lines))

    own = rows[rows["is_private_label"] == 1]["price_per_unit"].median()
    brand = rows[rows["is_private_label"] == 0]["price_per_unit"].median()
    if pd.notna(own) and pd.notna(brand) and brand > 0:
        gap = 100 * (brand - own) / brand
        st.info(
            f"Own brand has a median of {eur(own)}/{unit(type_id)} against {eur(brand)} for national brands: "
            f"{abs(gap):.0f}% {'less' if gap >= 0 else 'more'}."
        )

    st.markdown("#### The five cheapest")
    cheapest = rows.nsmallest(5, "price_per_unit")
    items = []
    for _, p in cheapest.iterrows():
        tag = '<span class="pl">Own brand</span>' if p["is_private_label"] == 1 else ""
        items.append(
            f'<div class="store-row" style="--chain:{CHAIN_COLORS.get(p["supermarket"], INK)}">'
            f'<div><span class="dot"></span>{p["name"]}{tag}<div class="addr">{chain_name(p["supermarket"])}</div></div>'
            f'<div style="text-align:right"><b>{eur(p["price_per_unit"])}/{unit(type_id)}</b>'
            f'<div class="addr">{eur(p["price"]) if pd.notna(p["price"]) else ""} per pack</div></div></div>'
        )
    html('<div class="panel" style="padding:4px 0">' + "".join(items) + "</div>")

    same = matched_for_type(type_id)
    if not same.empty:
        st.markdown("#### Same product at other chains")
        st.caption("Branded products our similarity engine found at two or more chains (same brand, variant and "
                   "size), compared per unit. Most sold first.")
        matched_view(same.sort_values(["chains", "gap_pct"], ascending=False).head(8),
                     [c for c in same.columns if c in CHAINS])


def add_to_basket(type_id):
    if type_id not in st.session_state.basket:
        st.session_state.basket = [*st.session_state.basket, type_id]


def basket_finder():
    """Free-text search: the engine maps a description to product types and to exact products."""
    query = st.text_input("Not sure of the name? Describe it", key="finder",
                          placeholder="e.g. huevos camperos, leche de avena, atún claro en lata")
    if not query.strip():
        return
    found, hits = search_products(query)
    top = hits.head(30)
    types = [t for t in dict.fromkeys(top["type_id"].dropna()) if t in TYPES.index][:3]
    if types:
        cols = st.columns(len(types))
        for col, type_id in zip(cols, types):
            col.button(f"Add {label(type_id)}", key=f"add_{type_id}", on_click=add_to_basket, args=(type_id,),
                       disabled=type_id in basket, width="stretch")
    else:
        st.caption("No product type fits this search; these are the closest products.")
    if not found.empty:
        st.caption("The same product at each chain, as matched by our similarity engine:")
        matched_view(found.head(5), [c for c in found.columns if c in CHAINS])
    else:
        near = top.head(5).merge(df[["product_key", "price"]], on="product_key", how="left")
        near["chain_id"] = near["chain_id"].replace(CHAIN_NAMES)
        st.dataframe(near.rename(columns={"name": "Product", "chain_id": "Chain", "price": "Price"})
                     [["Product", "Chain", "Price"]], hide_index=True, width="stretch",
                     column_config={"Price": st.column_config.NumberColumn(format="%.2f €")})


def open_sheet(type_id):
    st.session_state.sheet = type_id


def pick_sheet():
    open_sheet(st.session_state.sheet_pick)
    st.session_state.sheet_pick = None


def show_pending_sheet():
    if st.session_state.get("sheet"):
        product_sheet(st.session_state.pop("sheet"))


# ============================================================
# PAGE: BASKET
# ============================================================

def page_basket():
    header()
    html('<div class="hero">What are you buying?</div>')
    html(f'<div class="lede">Choose from {len(TYPES)} product types across {len(CHAINS)} chains. '
         "We compare each type with its equivalent elsewhere, not the brand. Product names stay in Spanish, as the chains sell them.</div>")

    st.multiselect(
        "Search products", TYPES.index, key="basket", format_func=label,
        placeholder="Search: leche, atún en aceite, pan de molde…", label_visibility="collapsed",
    )
    basket_finder()
    if not basket:
        html('<div class="panel">Your basket is empty. Search for a product above to start.</div>')
        return

    cheapest_chain = ppu.idxmin(axis=1)
    for type_id in basket:
        c1, c2, c3 = st.columns(
            [6, 2, 1.3],
            vertical_alignment="center",
            gap="small",
        )
        best = ppu.loc[type_id].min()
        hint = (
            f'from <b>{eur(best)}/{unit(type_id)}</b> at {chain_name(cheapest_chain[type_id])}'
            if pd.notna(best) else "no price at any chain"
        )
        c1.markdown(
            f'<div class="item-row"><span class="name">{label(type_id)}</span></div><div class="from">{hint}</div>',
            unsafe_allow_html=True,
        )
        is_count = TYPES.at[type_id, "unit"] in ("UNIDAD", "LAVADO")

        with c2:
            st.caption(f"Quantity ({unit(type_id)})")
            st.number_input(
                f"Quantity ({unit(type_id)})",
                min_value=0.25 if not is_count else 1.0,
                step=0.5 if not is_count else 1.0,
                key=f"qty_{type_id}",
                label_visibility="collapsed",
            )

        with c3:
            st.button(
                "Details",
                key=f"sheet_{type_id}",
                on_click=open_sheet,
                args=(type_id,),
                width="stretch",
            )

        st.divider()

    if st.button("Compare my basket", type="primary"):
        st.switch_page(PAGES["compare"])
    show_pending_sheet()


# ============================================================
# PAGE: COMPARE
# ============================================================

def matrix_html(cards):
    counts = unit_counts(tuple(basket))
    chains = list(cards["chain"])
    head = "".join(
        f'<th style="--chain:{CHAIN_COLORS.get(c, INK)}"><span class="dot"></span>{chain_name(c)}</th>' for c in chains
    )
    body = []
    for type_id in basket:
        row = cost.loc[type_id, chains]
        best = row.min()
        qty = st.session_state[f"qty_{type_id}"]
        cells = []
        for c in chains:
            v = row[c]
            if pd.isna(v):
                cells.append('<td class="none">—</td>')
            else:
                cls = ' class="best"' if v == best else ""
                n = counts.at[type_id, c]
                cells.append(
                    f'<td{cls}><span class="v">{eur(v)}</span>'
                    f'<span class="u">{eur(ppu.at[type_id, c])}/{unit(type_id)}'
                    f' · {n} products compared</span></td>'
            )        
        body.append(f'<tr><td><b>{label(type_id)}</b><span class="u">{qty:g} {unit(type_id)}</span></td>{"".join(cells)}</tr>')
    totals = "".join(
        f'<td>{eur(r.basket_eur) if r.complete else f"{r.have}/{len(basket)}"}</td>' for r in cards.itertuples()
    )
    body.append(f'<tr class="total"><td>Basket total</td>{totals}</tr>')
    return f'<div class="matrix-wrap"><table class="matrix"><tr><th>Product</th>{head}</tr>{"".join(body)}</table></div>'


def route_html(plan, stores):
    steps = opt.describe_route(plan, stores)
    items = [f'<li style="--chain:{INK}"><span class="stop"></span><b>Leave home</b></li>']
    for _, to, km in steps:
        what = [label(t) for t, c in plan["buy_at"].items() if c == to]
        color = CHAIN_COLORS.get(to, INK)
        title = "Back home" if to == opt.HOME else f"{chain_name(to)}"
        detail = f'<div class="small">{", ".join(what)}</div>' if what else ""
        items.append(
            f'<li style="--chain:{color if to != opt.HOME else INK}"><span class="stop"></span>'
            f'<b>{title}</b> <span class="km">{km_text(km)}</span>{detail}</li>'
        )
    return f'<ul class="route">{"".join(items)}</ul>'


def plan_summary(ranked):
    """The recommendation in words: best plan, headline, one-line reason and saving vs the best single store."""
    best = ranked.iloc[0]
    single = ranked[ranked["n_stores"] == 1]
    multi = ranked[ranked["n_stores"] > 1]
    if best["n_stores"] == 1:
        headline = f"Go to {chain_name(best['chains'][0])} only"
        extra = best["products_eur"] - multi["products_eur"].min() if not multi.empty else 0
        note = (
            f"Splitting the shop would save {eur(extra)} on products, "
            "but the extra trip and stop cost more than that."
            if extra > 0.005 else "No other store beats the price of your products."
        )
        saving = 0.0
    else:
        headline = "Split the shop: " + " and ".join(chain_name(c) for c in best["chains"])
        saving = single.iloc[0]["total_eur"] - best["total_eur"] if not single.empty else None
        note = (
            f"You save {eur(saving)} over the best single store, counting the trip and the extra stop."
            if saving is not None else "No single store has your whole basket."
        )
    return best, headline, note, saving


def alternatives_table(ranked):
    """Best plan for 1, 2 and 3 stores, plus the 'cheapest chain per item' baseline."""
    best = ranked.groupby("n_stores").head(1).sort_values("n_stores").reset_index(drop=True)    
    top = ranked.iloc[0]["total_eur"]
    table = pd.DataFrame({
        "Option": best["n_stores"].map(lambda n: f"Best with {n} store{'s' if n > 1 else ''}"),
        "Where": best["chains"].map(lambda cs: " + ".join(chain_name(c) for c in cs)),
        "Products": best["products_eur"].map(eur),
        "Travel + stops": (best["travel_eur"] + best["stops_eur"]).map(eur),
        "Time": best.apply(lambda r: f"{r['time_min']:.0f} min ({eur(r['time_eur'])})", axis=1),
        "Total": best["total_eur"].map(eur),
        "vs. recommended": (best["total_eur"] - top).map(
            lambda x: "Recommended" if x < 0.005 else f"+{eur(x)}"),
    })
    greedy = opt.greedy_plan(cost, candidates, HOME, st.session_state.cost_per_km,
                             st.session_state.cost_per_stop, st.session_state.max_stores,
                              **time_params())
    if greedy is not None:
        table.loc[len(table)] = [
            "Cheapest chain per item", " + ".join(chain_name(c) for c in greedy["chains"]),
            eur(greedy["products_eur"]), eur(greedy["travel_eur"] + greedy["stops_eur"]),
            f"{greedy['time_min']:.0f} min ({eur(greedy['time_eur'])})",
            eur(greedy["total_eur"]), f"+{eur(greedy['total_eur'] - top)}",
        ]
    return table


def recommendation_card(ranked, plan_stores):
    """Compact recommendation card with highlighted plan and total cost."""
    best, headline, note, saving = plan_summary(ranked)

    st.markdown(
        """
        <style>
        .st-key-recommended-plan {
            background-color: #ffe6cc;
            border: 1px solid #D8D0C2;
            border-radius: 12px;
            padding: 1.2rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="recommended-plan"):
        # Highlighted recommendation label
        html("""
        <div style="
            display:inline-block;
            background:#E3F1E8;
            border:1px solid #B9D8C5;
            color:#2E7D5B;
            padding:5px 11px;
            border-radius:7px;
            font-size:0.72rem;
            font-weight:750;
            letter-spacing:0.09em;
            margin-bottom:0.8rem;
        ">
            ✦ RECOMMENDED PLAN
        </div>
        """)

        left, right = st.columns([3, 1.5], vertical_alignment="center")

        with left:
            html(
                f'<div class="hero" style="max-width:none; margin:0; '
                f'font-size:1.4rem; line-height:1.15">{headline}</div>'
            )
            html(
                f'<div class="lede" style="margin:0.3rem 0 0; '
                f'font-size:0.86rem; opacity:0.8">{note}</div>'
            )

        with right:
            html(f"""
            <div style="
                background:#E3F1E8;
                border:1px solid #B9D8C5;
                border-radius:10px;
                padding:10px 13px;
                text-align:center;
            ">
                <div style="
                    font-size:0.68rem;
                    font-weight:750;
                    letter-spacing:0.09em;
                    color:#2E7D5B;
                    margin-bottom:3px;
                ">TOTAL TRIP COST</div>
                <div style="
                    font-size:1.65rem;
                    font-weight:800;
                    letter-spacing:-0.04em;
                    line-height:1.2;
                ">{eur(best["total_eur"])}</div>
            </div>
            """)
                
        st.caption(
            f"**Goal:** minimise total cost = products + travel ({eur(st.session_state.cost_per_km)}/km) "
            f"+ your time ({eur(st.session_state.value_of_time)}/hour) "
            f"+ {eur(st.session_state.cost_per_stop)} per extra stop, "
            f"using at most {st.session_state.max_stores} supermarkets."
        )

        st.markdown(
            '<div style="height:1px; background:rgba(128,128,128,0.2); '
            'margin:0.75rem 0 0.55rem"></div>',
            unsafe_allow_html=True
        )

        # Smaller cost breakdown
        a, b, c, t, d = st.columns(5)

        small_metric = """
        <style>
        div[data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.05rem !important;
        }
        </style>
        """
        st.markdown(small_metric, unsafe_allow_html=True)

        a.metric("Products", eur(best["products_eur"]))
        b.metric(
            "Travel",
            eur(best["travel_eur"]),
            help=f"{km_text(best['route_km'])} at "
                 f"{eur(st.session_state.cost_per_km)}/km"
        )
        c.metric("Extra stops", eur(best["stops_eur"]))
        t.metric(
            f"Time ({best['time_min']:.0f} min)",
            eur(best["time_eur"]),
            help=f"Route + {st.session_state.min_per_stop:.0f} min per store, "
                 f"valued at {eur(st.session_state.value_of_time)}/hour. Included in the total."
        )
        d.metric(
            "Saved vs. best single store",
            eur(saving) if saving else "—"
        )
        st.markdown("#### Your best route")
        html(route_html(best, plan_stores))
        with st.expander("Calculation settings"):
                    st.slider(
                        "Travel cost (€/km)", 0.0, 1.0,
                        step=0.05, key="cost_per_km",
                        help="0 if you walk. About 0.25 €/km by car."
                    )
                    st.slider(
                        "Value of your time (€/hour)", 0.0, 30.0,
                        step=1.0, key="value_of_time",
                        help="How much an hour of your time is worth. 0 ignores time."
                    )
                    st.slider(
                        "Travel speed (km/h)", 3.0, 40.0,
                        step=1.0, key="speed_kmh",
                        help="About 5 km/h walking, 25 km/h by car in the city."
                    )
                    st.slider(
                        "Minutes spent in each store", 0.0, 20.0,
                        step=1.0, key="min_per_stop",
                        help="Entering, finding the products, queueing and paying."
                    )
                    st.slider(
                        "Other cost of each extra stop (€)", 0.0, 5.0,
                        step=0.25, key="cost_per_stop",
                        help="Fixed costs besides time: parking, bags…"
                    )
                    st.radio(
                        "Maximum stores", [1, 2, 3],
                        key="max_stores", horizontal=True
                    )
                    st.radio(
                        "Price of each product", ["Typical price", "Cheapest"],
                        key="stat", horizontal=True,
                        help="Typical price = median of its type at the chain. Cheapest = the cheapest product of that type."
                    )
                    st.caption(
                        f"Straight-line distances × {opt.DETOUR:.1f}, because streets are not straight. "
                        "It tries up to 3 stores per chain and every visiting order."
                    )
        
        with st.expander("Compare with other options"):
            st.dataframe(alternatives_table(ranked), hide_index=True, width="stretch")



def page_compare():
    header()
    if not basket:
        html('<div class="hero">Your basket is empty</div>')
        if st.button("Add products", type="primary"):
            st.switch_page(PAGES["basket"])
        return
    if nearest.empty:
        html('<div class="hero">No supermarkets within your range</div>')
        st.write("Increase the maximum distance in the bar above.")
        return

    cards = chain_cards()
    plans, plan_stores = plans_for(cost, candidates, HOME, st.session_state.max_stores)
    ranked = opt.score(plans, st.session_state.cost_per_km, st.session_state.cost_per_stop, **time_params())
    if ranked.empty:
        html('<div class="hero">No combination of nearby stores has your whole basket</div>')
    else:
        recommendation_card(ranked, plan_stores)
    n_products = len(basket)
    html(f'<div class="lede">{n_products} products, {len(cards)} chains with a store within '
         f'{km_text(st.session_state.radius)}. Price of each product: '
         f'{"the cheapest of its type" if STAT == "min" else "the median of its type"} at each chain.</div>')

    st.segmented_control("Sort by", ["Price", "Distance", "Balance"], key="sort",
                         help="Balance adds the basket and the cost of the trip to the store and back.")
    best_price = cards.loc[cards["complete"], "basket_eur"].min()
    tags = []
    for _, r in cards.iterrows():
        win = r.complete and r.basket_eur == best_price
        if not r.complete:
            diff, cls = f"Missing {len(basket) - r.have} of your {len(basket)} products: not comparable", "partial"
        elif win:
            diff, cls = "Cheapest", "win"
        else:
            diff, cls = f"+{eur(r.basket_eur - best_price)} ({100 * (r.basket_eur / best_price - 1):.0f}%)", ""
        badge = '<span class="badge">Best price</span>' if win else ""
        total = price_tag(r.basket_eur, small=not r.complete) if pd.notna(r.basket_eur) else price_tag(0, small=True)
        tags.append(
            f'<div class="tag {cls}" style="--chain:{CHAIN_COLORS.get(r.chain, INK)}">{badge}'
            f'<div class="chain">{chain_name(r.chain)}</div>{total}<div class="diff">{diff}</div>'
            f'<div class="meta"><span>{km_text(r.distance_km)}</span><span>{r.have}/{len(basket)} products</span></div></div>'
        )
    html(f'<div class="tags">{"".join(tags)}</div>')

    st.markdown("### Where each product is cheapest")
    html(matrix_html(cards))
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #888; margin: 6px 0 12px 0;">
            <span style="
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 14px;
                height: 14px;
                border: 1px solid #999;
                border-radius: 50%;
                font-size: 10px;
                font-style: normal;
                margin-right: 5px;
            ">?</span>
            <i>Products compared: number of product records used to calculate
            the price for each supermarket.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.pills("See details for", basket, format_func=label, key="sheet_pick", selection_mode="single",
             on_change=pick_sheet)

        # Map each product type to its original category (searched_item).
    basket_cat_map = (
        df.dropna(subset=["product_type", "searched_item"])
        .assign(
            product_type=lambda x: x["product_type"].astype(str).str.strip()
        )
        .drop_duplicates("product_type")
        .set_index("product_type")["searched_item"]
        .to_dict()
        if {"product_type", "searched_item"}.issubset(df.columns)
        else {}
    )

    available_chains = (
        [c for c in cards["chain"] if c in cost.columns]
        if not cards.empty
        else list(cost.columns)
    )

    best_plan = ranked.iloc[0] if not ranked.empty else None
    buy_at = (
        best_plan["buy_at"]
        if best_plan is not None and isinstance(best_plan.get("buy_at"), dict)
        else {}
    )

    basket_rows = []

    for type_id in basket:
        type_key = str(type_id).strip()
        qty = float(quantities.get(type_id, 0.0))
        chosen_chain = buy_at.get(type_id)

        if (
            chosen_chain
            and type_id in cost.index
            and chosen_chain in cost.columns
            and pd.notna(cost.at[type_id, chosen_chain])
        ):
            line_eur = float(cost.at[type_id, chosen_chain])
        elif type_id in cost.index and available_chains:
            row_prices = pd.to_numeric(
                cost.loc[type_id, available_chains],
                errors="coerce"
            ).dropna()
            line_eur = float(row_prices.min()) if not row_prices.empty else 0.0
        else:
            line_eur = 0.0

        raw_cat = str(basket_cat_map.get(type_key, "")).strip()
        pretty_cat = insights.category_name(raw_cat) if raw_cat else ""

        # Replace generic or missing categories with the product name.
        if pretty_cat.casefold() in {"", "other", "otros", "uncategorized"}:
            pretty_cat = label(type_id)

        basket_rows.append({
            "product_type": type_key,
            "category": pretty_cat,
            "qty": qty,
            "amount_eur": line_eur,
        })

    basket_spend_df = pd.DataFrame(basket_rows)

    st.markdown("#### Spending by category in the recommended plan")
    _, donut_col, _ = st.columns([0.8, 2.4, 0.8], vertical_alignment="center")
    with donut_col:
        st.altair_chart(
            charts.basket_category_donut(
                basket_spend_df,
                bg_color="#F4EFE4",
            ),
            width="stretch",
            theme=None,
        )
    html(
            '<div style="font-size:0.875rem; color:var(--muted); margin-top:2px; margin-left:18%;">'
            'Share of product spending across categories in your recommended plan.'
            '</div>'
        )
    st.write("")
    left_chart, right_chart = st.columns(2, gap="large")

    with left_chart:
        st.markdown("#### Is it worth going further?")
        st.altair_chart(
            charts.worth_going_further(
                cards,
                CHAIN_COLORS,
                CHAIN_NAMES,
                st.session_state.cost_per_km,
                cost,
            ),
            width="stretch",
            theme=None,
        )
        st.caption(
            "Right of +0.00 means the extra trip is worth it per comparable item;"
            "left means travel cost exceeds saving. Y-axis shows basket coverage"
        )

    with right_chart:
        st.markdown("#### When do two stores pay off?")
        if ranked.empty:
            st.write("This needs at least one combination of stores with your whole basket.")
        else:
            chart, threshold = sensitivity_view(plans)
            st.altair_chart(chart, width="stretch", theme=None)
            st.caption(threshold)
    show_pending_sheet()


def sensitivity_view(plans):
    km_costs = np.round(np.arange(0, 1.01, 0.05), 2)
    sens = opt.sensitivity(plans, km_costs, st.session_state.cost_per_stop, **time_params())
    split = sens[sens["best_n_stores"] > 1]
    if split.empty:
        text = "With your cost per stop, one store is always best."
    elif len(split) == len(sens):
        text = "Splitting the shop pays off even at 1 €/km."
    else:
        text = f"Splitting the shop pays off up to about {split['cost_per_km'].max():.2f} €/km; above that, one store is better."
    return charts.sensitivity(sens, st.session_state.cost_per_km), text + " The dotted vertical line is your current setting."


# ============================================================
# PAGE: MAP
# ============================================================

def page_map():
    header()
    html('<div class="hero">Your basket, store by store</div>')
    html(f'<div class="lede">Each pin shows what your basket would cost at that store. '
         f'The circle marks your maximum distance ({km_text(st.session_state.radius)}).</div>')
    if nearest.empty:
        st.write("No supermarkets within your range. Increase the maximum distance in the bar above.")
        return

    cards = chain_cards().set_index("chain") if basket else pd.DataFrame()
    shown = opt.candidate_stores(all_stores, HOME, per_chain=6, max_km=st.session_state.radius).head(30)
    plans, plan_stores = plans_for(cost, candidates, HOME, st.session_state.max_stores) if basket else (pd.DataFrame(), None)
    ranked = opt.score(plans, st.session_state.cost_per_km, st.session_state.cost_per_stop, **time_params())

    m = folium.Map(location=HOME, zoom_start=14, tiles=None, control_scale=True)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri, © OpenStreetMap contributors", max_zoom=16,
    ).add_to(m)
    folium.Circle(HOME, radius=st.session_state.radius * 1000, color=INK, weight=1.5, dash_array="6 6", fill=True,
                  fill_opacity=0.04).add_to(m)
    folium.Marker(HOME, tooltip="Your address", icon=folium.DivIcon(
        html=f'<div style="background:{INK};color:#fff;font:700 12px Archivo,sans-serif;padding:5px 9px;border-radius:14px;'
             'white-space:nowrap;transform:translate(-50%,-50%);display:inline-block">Home</div>'), z_index_offset=1000).add_to(m)

    def pin_text(chain):
        if chain not in cards.index:
            return "?"
        r = cards.loc[chain]
        return eur(r.basket_eur) if r.complete else f"{r.have}/{len(basket)}"

    full = cards[cards["complete"]] if len(cards) else cards
    cheapest_chain = full["basket_eur"].idxmin() if len(full) else None
    for _, s in shown.iterrows():
        color = CHAIN_COLORS.get(s["supermarket"], INK)
        ring = f"outline:3px solid {TAG};" if s["supermarket"] == cheapest_chain else ""
        folium.Marker(
            (s["lat"], s["lon"]), tooltip=f'{chain_name(s["supermarket"])}, {km_text(s["distance_km"])}',
            icon=folium.DivIcon(html=(
                f'<div style="background:{color};color:#fff;font:800 13px Archivo,sans-serif;padding:4px 8px;'
                f'border-radius:6px;white-space:nowrap;display:inline-block;transform:translate(-50%,-100%);'
                f'box-shadow:0 2px 6px rgba(0,0,0,.25);{ring}">{pin_text(s["supermarket"])}</div>')),
        ).add_to(m)

    if not ranked.empty:
        best = ranked.iloc[0]
        path = [HOME, *((plan_stores.at[i, "lat"], plan_stores.at[i, "lon"]) for i in best["store_ids"]), HOME]
        folium.PolyLine(path, color=BEST, weight=4, opacity=0.85, dash_array="2 8").add_to(m)

    left, right = st.columns([2, 3], gap="medium")
    with right:
        clicked = st_folium(m, height=560, use_container_width=True, returned_objects=["last_object_clicked"])
    picked = None
    if clicked and clicked.get("last_object_clicked"):
        p = clicked["last_object_clicked"]
        d = opt.haversine(p["lat"], p["lng"], shown["lat"], shown["lon"])
        picked = int(np.argmin(d)) if d.min() < 0.01 else None
    route_coords = set()
    if not ranked.empty:
        best = ranked.iloc[0]
        route_coords = {
            (round(float(plan_stores.at[i, "lat"]), 6), round(float(plan_stores.at[i, "lon"]), 6))
            for i in best["store_ids"]
        }
    with left:
        if not ranked.empty:
            html(f'<div class="small" style="margin-bottom:8px">The green line is your best route: '
                 f'{" and ".join(chain_name(c) for c in ranked.iloc[0]["chains"])}.</div>')
        rows = []
        for i, s in enumerate(shown.itertuples()):
            is_route = (round(float(s.lat), 6), round(float(s.lon), 6)) in route_coords
            sel = " sel" if i == picked else ""
            route_cls = " in-route" if is_route else ""
            route_badge = (
                ' <span style="font-size:0.72rem;font-weight:700;color:#C85A17;'
                'background:#FFD2A6;padding:1px 6px;border-radius:4px;margin-left:6px;">In route</span>'
                if is_route else ""
            )
            rows.append(
                f'<div class="store-row{sel}{route_cls}" style="--chain:{CHAIN_COLORS.get(s.supermarket, INK)}">'
                f'<div><span class="dot"></span><b>{chain_name(s.supermarket)}</b> '
                f'<span class="small">{km_text(s.distance_km)}</span>{route_badge}'
                f'<div class="addr">{s.address or "See on the map"}</div></div>'
                f'<div><b>{pin_text(s.supermarket)}</b></div></div>'
            )
        html('<div class="panel" style="padding:0;max-height:560px;overflow-y:auto">' + "".join(rows) + "</div>")
        st.caption("Click a pin to highlight it in the list.")


# ============================================================
# PAGE: TRENDS
# ============================================================

@st.cache_data
def load_changes():
    return history.load_changes()


def show(chart):
    st.altair_chart(chart, width="stretch", theme=None)


def day(date):
    """Format a collection date, for example '2026-11-10' as '10 Nov'."""
    return pd.Timestamp(date).strftime("%d %b").lstrip("0")


def _trend_flag(values):
    """Interpret boolean flags read from the historical comparison CSV."""
    return values.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


@st.cache_data(show_spinner=False)
def load_series():
    """(exact-product series, product options, product-type series) for the price-over-time chart."""
    from src.pipeline import config

    hist = history.load_history()
    types = history.type_series(hist)
    if not config.SIM_CANONICAL_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), types
    products = history.product_series(hist, pd.read_csv(config.SIM_CANONICAL_PATH))
    # Most chains first, then most dates, then the products whose prices moved most
    options = products.groupby("canonical_id").agg(
        name=("name", "first"), chains=("supermarket", "nunique"), dates=("collection_date", "nunique"),
        moved=("price", lambda p: p.max() / p.min() - 1),
    ).sort_values(["chains", "dates", "moved"], ascending=False)
    return products, options, types


def section_price_over_time():
    st.markdown("### Price over time")
    products, options, types = load_series()
    modes = ["Exact product", "Product type"] if not options.empty else ["Product type"]
    mode = st.segmented_control("Compare", modes, default=modes[0], key="trend_mode", label_visibility="collapsed") or modes[0]
    if mode == "Exact product":
        choice = st.selectbox(
            "Product (same brand and size at every chain, matched by our similarity engine)", options.index,
            format_func=lambda c: f"{options.at[c, 'name']}  ·  {options.at[c, 'chains']} chains", key="trend_product")
        series = products[products["canonical_id"] == choice]
        note = "Engine-matched product: same brand, variant and size at each chain."
    else:
        in_basket = [t for t in st.session_state.get("basket", {}) if t in TYPES.index]
        type_ids = TYPES.index.tolist()
        choice = st.selectbox("Product type", type_ids, index=type_ids.index(in_basket[0]) if in_basket else 0,
                              format_func=label, key="trend_type")
        series = types[types["product_type"] == choice]
        note = ("Median of the products each chain had on each date. It also moves when a chain's range changes, "
                "not only when prices change.")
    if series.empty:
        st.caption("No price history for this choice yet.")
        return

    unit_text = UNIT_TEXT.get(series["unit"].iat[0], str(series["unit"].iat[0]).lower())
    dates = sorted(series["collection_date"].unique())
    if len(dates) > 2:
        start, end = st.select_slider("Window", options=dates, value=(dates[0], dates[-1]), format_func=day,
                                      key=f"trend_window_{mode}_{choice}")
        series = series[series["collection_date"].between(start, end)]
    show(charts.price_over_time(series, CHAIN_COLORS, CHAIN_NAMES, unit_text))

    pills = []
    for chain in [c for c in CHAIN_COLORS if c in set(series["supermarket"])]:
        rows = series[series["supermarket"] == chain].sort_values("collection_date")
        first, last = rows.iloc[0], rows.iloc[-1]
        change = 100 * (last.price / first.price - 1)
        change_text = "0.0%" if abs(change) < 0.05 else f"{change:+.1f}%"

        if abs(change) < 0.05:
            badge_bg, badge_fg = "#EFECE4", "#5F6E78"
        elif change > 0:
            badge_bg, badge_fg = "#E3F1E8", "#1E5736"
        else:
            badge_bg, badge_fg = "#F9E4C9", "#6E5844"

        span = (
            f"{day(first.collection_date)} to {day(last.collection_date)}"
            if len(rows) > 1
            else f"only {day(first.collection_date)}"
        )
        pills.append(
            f'<div class="trend-pill" title="{len(rows)} dates">'
            f'<div class="trend-pill-main">'
            f'<span class="trend-dot" style="background:{CHAIN_COLORS[chain]};"></span>'
            f'<span class="trend-chain">{chain_name(chain)}</span>'
            f'<span class="trend-sep">·</span>'
            f'<span class="trend-price">{last.price:.2f} €/{unit_text}</span>'
            f'<span class="trend-badge" style="background:{badge_bg};color:{badge_fg};">({change_text})</span>'
            f'</div>'
            f'<div class="trend-date">{span}</div>'
            f'</div>'
        )

    html(
        """
        <style>
        .trend-pills {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 12px;
            margin: 0.85rem 0 0.75rem;
        }
        .trend-pill {
            display: inline-flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 3px;
            background: #FFFFFF;
            border: 1px solid #E2DACC;
            border-radius: 16px;
            padding: 9px 16px;
            font-size: 0.98rem;
            color: #26332D;
            white-space: nowrap;
        }
        .trend-pill-main {
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .trend-dot {
            width: 11px;
            height: 11px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .trend-chain {
            font-weight: 800;
        }
        .trend-sep {
            color: #64736B;
            font-weight: 700;
        }
        .trend-price {
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }
        .trend-badge {
            font-weight: 700;
            font-size: 0.86rem;
            padding: 2px 8px;
            border-radius: 999px;
            font-variant-numeric: tabular-nums;
        }
        .trend-date {
            padding-left: 19px;
            font-size: 0.78rem;
            color: #64736B;
        }
        </style>
        """
        + f'<div class="trend-pills">{"".join(pills)}</div>'
    )
    st.caption(f"{note} Prices per {unit_text}. Chains are only drawn on the dates we collected them.")

def page_trends():
    header()
    try:
        changes = load_changes()
    except FileNotFoundError:
        st.error("Price changes are missing. From the project folder, run: python -m src.analysis.history")
        return

    if changes.empty:
        st.info("There are no price comparisons yet. Each chain needs at least two collection dates.")
        return

    # Only include products that pass the optimizer's comparability checks.
    comparable = changes[_trend_flag(changes["comparable"])].copy()
    held_back = _trend_flag(comparable["held_back"])
    valid_changes = comparable[~held_back].copy()

    if valid_changes.empty:
        st.info("There are historical matches, but no price changes pass the comparability and review filters yet.")
        st.caption(f"{len(changes):,} matched products; {len(comparable):,} comparable products.")
        return

    summary = history.chain_summary(valid_changes)
    top = summary.iloc[0]
    html(f'<div class="hero">{top.changed_pct:.0f}% of comparable {chain_name(top.chain)} prices changed between collections</div>')
    n_chains = len(summary)
    html(f'<div class="lede">Price comparisons are available for {n_chains} supermarket'
         f'{"s" if n_chains != 1 else ""}. Each chain uses its two latest collection dates. '
         'Further collections will allow longer-term price trends.</div>')
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(
            f"""
            <span style="font-size:0.8rem; color:var(--muted); white-space:nowrap;">
                {len(changes):,} matched products · {len(comparable):,} comparable ·
                {len(valid_changes):,} included after review filters.
            </span>
            """,
            unsafe_allow_html=True,
        )

    with st.popover("ⓘ"):
        st.markdown("**Matched products**")
        st.caption("Products identified and matched across the two collection dates.")

        st.markdown("**Comparable products**")
        st.caption("Matched products that meet the criteria for a reliable price comparison.")

        st.markdown("**Included after review filters**")
        st.caption("Comparable products retained after excluding unusually large price changes flagged for review.")
    tags = []
    for r in summary.itertuples():
        tags.append(
            f'<div class="tag" style="--chain:{CHAIN_COLORS.get(r.chain, INK)}"><div class="chain">{chain_name(r.chain)}</div>'
            f'<div class="price sm"><span class="eu">{r.average_pct:+.1f}</span><span class="cur">%</span></div>'
            f'<div class="diff" style="color:var(--muted)">average change across comparable products</div>'
            f'<div class="meta"><span>{day(r.old_date)} to {day(r.new_date)}</span><span>{r.products} comparable products</span></div></div>'
        )
    html(f'<div class="tags">{"".join(tags)}</div>')

    section_price_over_time()

    st.markdown("### How many prices moved")
    show(charts.moved_share(summary, CHAIN_NAMES, fall_color="#1E5736", rise_color="#F0883E", same_color="#ABB4AD"))
    
    st.markdown("### Biggest changes")

    rises, falls = history.biggest_moves(valid_changes)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Falls**")
        st.caption("Largest decreases in comparable products")

        if not falls.empty:
            show(charts.moves(falls, CHAIN_NAMES, "#1E5736", rounded_left=True))
        else:
            st.caption("No price decreases available.")

    with right:
        st.markdown("**Rises**")
        st.caption("Largest increases in comparable products")

        if not rises.empty:
            show(charts.moves(rises, CHAIN_NAMES, "#F0883E", rounded_left=False))
        else:
            st.caption("No price increases available.")

# ============================================================
# PAGE: ANALYSIS
# ============================================================

@st.cache_data(show_spinner=False)
def route_comparison(stores, home, cost_per_km, cost_per_stop, **time):
    return insights.greedy_vs_exact(df, stores, home, cost_per_km, cost_per_stop, **time)


@st.cache_resource(show_spinner="Loading product search…")
def load_searcher():
    from src.pipeline import config
    from src.pipeline.classify import load_taxonomy, types_for_db
    from src.similarity.search import Searcher

    try:
        from src.similarity.encoder import load_encoder
        encoder = load_encoder("finetuned")
    except Exception:  # no torch / model on this computer: text search
        encoder = None
    products = df.rename(columns={"supermarket": "chain_id", "product_type": "type_id"})
    return Searcher(products, encoder, types_for_db(load_taxonomy(config.TAXONOMY_PATH)))


@st.cache_data
def load_matched():
    from src.pipeline import config

    if not (config.SIM_MATCHES_PATH.exists() and config.SIM_CANONICAL_PATH.exists()):
        return None
    canonical = pd.read_csv(config.SIM_CANONICAL_PATH)
    table = matched.same_product_table(df, pd.read_csv(config.SIM_MATCHES_PATH), canonical)
    return table, canonical.set_index("product_key")["canonical_id"]


RELEVANT_SHARE = 0.85  # matched products scoring below 85 % of the best matched one are not shown
MATCHED_FLOOR = 0.7  # ... nor below 70 % of the best product overall (19 Nov, 10 queries: relevant ≥ .82, unrelated ≤ .53)


def matched_for_type(type_id):
    """Engine-matched products (sold at 2+ chains) whose members belong to this product type."""
    loaded = load_matched()
    if loaded is None:
        return pd.DataFrame()
    table, canonical_of = loaded
    keys = df.loc[df["product_type"] == type_id, "product_key"]
    ids = set(canonical_of.reindex(keys).dropna())
    return table[table["canonical_id"].isin(ids)]


def search_products(query):
    """Engine search → (matched rows for the query, raw hits). Weak tails are cut like in the Analysis tab."""
    loaded, searcher = load_matched(), load_searcher()
    hits = searcher.search(query, k=400)
    if loaded is None or hits.empty:
        return pd.DataFrame(), hits
    table, canonical_of = loaded
    found = hits[hits["product_key"].isin(canonical_of.index)]  # own brands are never matched
    if found.empty:
        return pd.DataFrame(), hits
    floor = max(RELEVANT_SHARE * found["score"].max(), MATCHED_FLOOR * hits["score"].max())
    found = found[found["score"] >= floor]
    ids = list(dict.fromkeys(canonical_of.reindex(found["product_key"]).dropna()))
    return table.set_index("canonical_id").reindex(ids).dropna(how="all").reset_index(), hits


def matched_view(shown, chains):
    """Product × chain price table for matched products, in the dashboard's format."""
    shown = shown.assign(cheapest=shown["cheapest"].replace(CHAIN_NAMES),
                         unit=shown["unit"].map(lambda u: "€/" + UNIT_TEXT.get(u, str(u).lower())))
    chains = [c for c in chains if shown[c].notna().any()]
    view = shown[["name", "unit", *chains, "cheapest", "gap_pct"]].rename(
        columns={"name": "Product", "unit": "Price per", "cheapest": "Cheapest", "gap_pct": "Dearest vs cheapest",
                 **CHAIN_NAMES})
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        **{CHAIN_NAMES.get(c, c): st.column_config.NumberColumn(format="%.2f €") for c in chains},
        "Dearest vs cheapest": st.column_config.NumberColumn(format="+%.0f%%")})


def matched_search(chains, key="matched_query"):
    """Search box over the engine-matched products; shows the product × chain table for a query."""
    query = st.text_input("Find a product", placeholder="e.g. leche sin lactosa, atún claro, yogur griego", key=key)
    if load_searcher().mode == "tfidf":
        st.caption("Search runs in text mode: the embedding model isn't installed on this computer.")
    if query.strip():
        found, hits = search_products(query)
        if found.empty:
            st.write("No matched product for this search. These are the closest products we found:")
            near = hits.head(10).merge(df[["product_key", "price"]], on="product_key", how="left")
            near["chain_id"] = near["chain_id"].replace(CHAIN_NAMES)
            st.dataframe(near.rename(columns={"name": "Product", "chain_id": "Chain", "price": "Price"})
                         [["Product", "Chain", "Price"]], hide_index=True, width="stretch")
        else:
            matched_view(found, chains)
        return True
    return False


def section_matched():
    loaded = load_matched()
    if loaded is None:
        return
    table, _ = loaded
    chains = [c for c in table.columns if c in CHAINS]
    st.markdown(f"### {len(table):,} products matched across chains by our similarity engine")
    if matched_search(chains):
        st.divider()

    shown =table.sort_values(["chains", "gap_pct"], ascending=False).head(50).assign(
        cheapest=lambda d: d["cheapest"].replace(CHAIN_NAMES),
        unit=lambda d: d["unit"].map(lambda u: "€/" + UNIT_TEXT.get(u, str(u).lower())),
    )
    view = shown[["name", "unit", *chains, "cheapest", "gap_pct"]].rename(
        columns={
            "name": "Product",
            "unit": "Price per",
            "cheapest": "Cheapest",
            "gap_pct": "Dearest vs cheapest",
            **CHAIN_NAMES,
        }
    )
    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        column_config={
            **{CHAIN_NAMES.get(c, c): st.column_config.NumberColumn(format="%.2f €") for c in chains},
            "Dearest vs cheapest": st.column_config.NumberColumn(format="+%.0f%%"),
        },
    )
    st.caption(
        "Same brand, variant and size at each chain, as matched by the engine, compared per kilo, litre or "
        "unit. Without a search, the 50 matched products sold at the most chains."
    )


def section_route():
    st.markdown("### What the exact optimizer adds")

    if candidates["supermarket"].nunique() < 2:
        st.write(
            "This needs stores from two chains within your range. "
            "Increase the maximum distance in the bar above."
        )
        return

    with st.spinner("Comparing random baskets…"):
        table = route_comparison(
            candidates,
            HOME,
            st.session_state.cost_per_km,
            st.session_state.cost_per_stop,
            **time_params(),
        )

    if table.empty:
        st.write(
            "None of the random baskets could be bought within your range. "
            "Increase the maximum distance."
        )
        return

    # Calculate the three summary indicators.
    avg_saving = table["saving_eur"].mean()
    avg_stops_saved = (table["greedy_stops"] - table["exact_stops"]).mean()
    improved = int((table["saving_eur"] > 0).sum())
    total_baskets = len(table)

    st.caption(
        "Comparing the exact optimizer against the Price list approach "
        f"across {total_baskets} baskets."
    )

    # KPI cards.
    kpis_html = (
        '<div class="offer-grid">'
        f'<div class="offer-card"><div class="offer-label">Average saving per basket</div>'
        f'<div class="offer-number" style="margin-top:8px">{eur(avg_saving)}</div>'
        f'<div class="offer-detail">Total cost reduction vs. Price list</div></div>'
        f'<div class="offer-card"><div class="offer-label">Average stops saved</div>'
        f'<div class="offer-number" style="margin-top:8px">{avg_stops_saved:+.1f}</div>'
        f'<div class="offer-detail">{table["exact_stops"].mean():.1f} stops vs. {table["greedy_stops"].mean():.1f}</div></div>'
        f'<div class="offer-card"><div class="offer-label">Baskets improved</div>'
        f'<div class="offer-number" style="margin-top:8px">{improved} / {total_baskets}</div>'
        f'<div class="offer-detail">{100 * improved / total_baskets:.0f}% of simulated baskets</div></div>'
        '</div>'
    )
    html(kpis_html)

    st.markdown("#### Saving per basket")
    show(charts.optimizer_savings(table))

    st.caption(
        f"{total_baskets} random baskets of 5 product types. "
        "Price list means buying each item at its cheapest chain, "
        "with a maximum of 3 stores. Both methods use the same "
        "travel, stop and time-cost assumptions. Totals include "
        "product costs, travel and extra stops."
    )


# ---------- Shopper view: plain questions, answers in euros ----------

def shopper_list():
    table, n_types = insights.list_cost(df)
    best, worst = table.iloc[0], table.iloc[-1]
    st.markdown(f"### The same shopping list is cheapest at {chain_name(best.chain)}: "
                f"{eur(worst.extra)} less than at {chain_name(worst.chain)}")
    st.caption(f"One kg, litre or unit of each of the {n_types} product types all 7 chains sell, "
               "buying the cheapest product of each type at that chain.")
    show(charts.list_cost(table, CHAIN_COLORS, CHAIN_NAMES))
    typical = table.sort_values("typical").iloc[0]
    st.caption(f"If you pick typical (median) products instead of the cheapest, {chain_name(typical.chain)} "
               f"comes out cheapest ({eur(typical.typical)}). A chain with a large range can be cheap at the bottom "
               "and dear in the middle.")


def shopper_aisles():
    table = insights.category_prices(df, "min")
    table = table[~table["low_data"]]
    st.markdown("### Where to buy each aisle")
    st.caption("Cheapest chain per category, buying the cheapest product of each type. "
               "The runner-up shows how much more the next best chain costs.")
    chart_rows = []
    for category, group in sorted(table.groupby("category"), key=lambda g: insights.category_name(g[0])):
        group = group.sort_values("relative")
        win = group.iloc[0]
        aisle_label = insights.category_name(category)
        if len(group) > 1:
            second = group.iloc[1]
            extra = 100 * (second.relative / win.relative - 1)
            second_chain = second.chain
            extra_val = float(extra)
        else:
            second_chain = None
            extra_val = np.nan

        chart_rows.append({
            "aisle": aisle_label,
            "winner": win.chain,
            "runner": second_chain,
            "zero": 0.0,
            "extra_pct": extra_val,
            "label_x": extra_val if pd.notna(extra_val) else 0.0,
            "n_chains": len(group),
        })

    if chart_rows:
        show(charts.aisle_runnerup_dumbbell(pd.DataFrame(chart_rows), CHAIN_COLORS, CHAIN_NAMES))
        st.caption("Left dot (0%): cheapest chain for that aisle. Right dot: second-cheapest chain and how much more it costs.")


def shopper_same_product():
    loaded = load_matched()
    if loaded is None:
        return
    table, _ = loaded
    chains = [c for c in table.columns if c in CHAINS]
    st.markdown("### Same product, different price: check before you buy")
    st.caption("Identical products (same brand, variant and size) found at several chains by our similarity engine.")
    if not matched_search(chains, key="shopper_query"):
        sure = table[(table["confidence"] == "high") & (table["chains"] >= 3)]
        st.markdown("**Biggest gaps on identical products**")
        matched_view(sure.sort_values("gap_pct", ascending=False).head(10), chains)
        st.caption("Only products found at 3 or more chains with a high-confidence match. Prices per kg, litre or unit.")


def page_analysis():
    header()
    html('<div class="hero">What the prices tell us</div>')
    html('<div class="lede">Where to shop, in euros. Across all products we collected, not just your basket.</div>')
    sections = (shopper_list, shopper_aisles, shopper_same_product, section_route)
    for i, section in enumerate(sections):
        section()
        if i < len(sections) - 1:
            st.divider()


# ============================================================
# NAVIGATION
# ============================================================

PAGES = {
    "basket": st.Page(page_basket, title="Basket", icon=":material/shopping_basket:", url_path="basket", default=True),
    "compare": st.Page(page_compare, title="Compare", icon=":material/compare_arrows:", url_path="compare"),
    "map": st.Page(page_map, title="Map", icon=":material/map:", url_path="map"),
    "trends": st.Page(page_trends, title="Trends", icon=":material/trending_up:", url_path="trends"),
    "analysis": st.Page(page_analysis, title="Analysis", icon=":material/insights:", url_path="analysis"),
}
st.navigation(list(PAGES.values()), position="top").run()

st.markdown("---")
st.caption("Stores: © OpenStreetMap contributors. Prices: collected by web scraping and each chain's APIs.")
