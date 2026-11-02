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

import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import re
import requests
import streamlit as st
from streamlit_folium import st_folium

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from dashboard import charts  # noqa: E402
from src.analysis import history, insights  # noqa: E402
from src.analysis import optimizer as opt  # noqa: E402
from src.analysis import prices as pricing  # noqa: E402

# Theme for this app only (a .streamlit/config.toml would also change app_legacy.py)
from streamlit import config as st_config  # noqa: E402

for option, value in {"theme.base": "light", "theme.primaryColor": "#17242D", "theme.backgroundColor": "#F1F4F1",
                      "theme.secondaryBackgroundColor": "#E6EBE7", "theme.textColor": "#17242D"}.items():
    st_config.set_option(option, value)

st.set_page_config(page_title="Supermarket comparison", page_icon="🛒", layout="wide")


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
:root{--shelf:#F1F4F1;--surface:#FFFFFF;--ink:#17242D;--muted:#5F6E78;--line:#DCE3DE;--tag:#FFD60A;--best:#12805C;--best-bg:#E3F3EA;}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:var(--shelf);color:var(--ink);font-family:'Archivo',sans-serif;}
.stApp *:not([data-testid="stIconMaterial"]),[role="dialog"] *:not([data-testid="stIconMaterial"]){font-family:'Archivo',sans-serif;}
span[data-baseweb="tag"]{background:var(--ink)!important;color:#fff!important;}
button[data-testid="stBaseButton-segmented_controlActive"],button[data-testid="stBaseButton-pillsActive"]{background:var(--ink)!important;color:#fff!important;border-color:var(--ink)!important;}
button[data-testid="stBaseButton-segmented_controlActive"] *,button[data-testid="stBaseButton-pillsActive"] *{color:#fff!important;}
.block-container{padding-top:4.2rem;max-width:1240px;}
h1,h2,h3{font-family:'Archivo',sans-serif;font-stretch:112%;font-weight:750;color:var(--ink);letter-spacing:-0.01em;}
.hero{font-stretch:108%;font-weight:760;font-size:clamp(1.7rem,3.4vw,2.7rem);line-height:1.08;margin:0.2rem 0 0.4rem;max-width:26ch;}
.lede{color:var(--muted);font-size:1.05rem;max-width:62ch;margin-bottom:1.2rem;}
.price{font-stretch:70%;font-weight:850;font-variant-numeric:tabular-nums;line-height:0.9;white-space:nowrap;}
.price .eu{font-size:3.3rem;} .price .ct{font-size:1.45rem;vertical-align:1.55rem;margin-left:1px;} .price .cur{font-size:1.3rem;margin-right:3px;}
.price.sm .eu{font-size:2.2rem;} .price.sm .ct{font-size:1rem;vertical-align:1rem;} .price.sm .cur{font-size:0.95rem;}
.tags{display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:14px;margin:0.6rem 0 1.6rem;}
.tag{background:var(--surface);border:1px solid var(--line);border-left:7px solid var(--chain);border-radius:4px 10px 10px 4px;padding:14px 16px 12px;position:relative;}
.tag.win{background:var(--tag);border-color:#E9BE00;border-left-color:var(--chain);box-shadow:0 6px 18px -10px rgba(120,90,0,.55);}
.tag .chain{font-stretch:115%;font-weight:760;font-size:1.05rem;}
.tag .badge{position:absolute;top:12px;right:12px;font-size:.78rem;font-weight:700;background:var(--ink);color:var(--tag);padding:2px 8px;border-radius:20px;}
.tag .diff{font-weight:650;font-size:.95rem;color:#A23B2A;margin:2px 0 8px;}
.tag.win .diff{color:var(--ink);}
.tag .meta{display:flex;justify-content:space-between;font-size:.88rem;color:var(--muted);border-top:1px dashed rgba(23,36,45,.25);padding-top:7px;margin-top:8px;}
.tag.win .meta{color:var(--ink);}
.tag.partial{opacity:.72;} .tag.partial .diff{color:var(--muted);}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--chain);margin-right:6px;vertical-align:1px;}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 20px;}
.matrix-wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:12px;}
table.matrix{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.95rem;}
table.matrix th,table.matrix td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap;}
table.matrix th{font-stretch:110%;font-weight:700;font-size:.9rem;background:#FAFBFA;}
table.matrix th:first-child,table.matrix td:first-child{text-align:left;position:sticky;left:0;background:var(--surface);z-index:1;}
table.matrix th:first-child{background:#FAFBFA;}
table.matrix td .u{display:block;color:var(--muted);font-size:.78rem;}
table.matrix td.best span.v{background:var(--best-bg);color:var(--best);font-weight:750;padding:2px 7px;border-radius:5px;}
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
.store-row .addr{color:var(--muted);font-size:.82rem;}
.bar{height:8px;border-radius:4px;background:var(--chain);}
.pl{display:inline-block;font-size:.72rem;font-weight:700;padding:1px 6px;border-radius:4px;background:#E7ECEF;color:var(--ink);margin-left:6px;}
.stButton>button[kind="primary"]{background:var(--ink);border-color:var(--ink);font-weight:700;font-stretch:110%;padding:.7rem 1.6rem;font-size:1.05rem;}
.stButton>button[kind="primary"]:hover{background:#0E7650;border-color:#0E7650;}
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
    "cost_per_stop": 1.0,
    "max_stores": 2,
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


location = geocode(st.session_state.address)
if location == "error":
    st.error("The address service is not responding. Try again in a few seconds, or use the example address.")
    st.stop()
if location is None:
    st.error("We could not find that address. Enter street, number and town, for example \"Calle Mayor 1, Madrid\".")
    st.stop()
user_lat, user_lon, _, offline_geo = location
HOME = (user_lat, user_lon)

with st.spinner("Finding supermarkets near you…"):
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


# ============================================================
# SHARED HEADER: address and basket always at hand
# ============================================================

def header():
    left, mid, right = st.columns([3.3, 4.7, 2], vertical_alignment="center")
    left.markdown(
        '<div style="font-stretch:118%;font-weight:800;font-size:1.12rem;white-space:nowrap;">🛒 Supermarket comparison</div>',
        unsafe_allow_html=True,
    )
    with mid.popover(f"📍 Shopping near {st.session_state.address}, within {km_text(st.session_state.radius)}", width="stretch"):
        st.text_input("Address", key="address")
        st.slider("Maximum distance to the store (km)", 0.3, 5.0, step=0.1, key="radius")
    with right.popover(f"Basket ({len(basket)})", icon=":material/shopping_basket:", width="stretch"):
        if not basket:
            st.write("Your basket is empty.")
        for t in basket:
            st.markdown(f"**{label(t)}** · {st.session_state[f'qty_{t}']:g} {unit(t)}")
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
    if not basket:
        html('<div class="panel">Your basket is empty. Search for a product above to start.</div>')
        return

    st.write("")
    cheapest_chain = ppu.idxmin(axis=1)
    for type_id in basket:
        c1, c2, c3 = st.columns([6, 2, 1.3], vertical_alignment="center")
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
        c2.number_input(
            f"Quantity ({unit(type_id)})", min_value=0.25 if not is_count else 1.0,
            step=0.5 if not is_count else 1.0, key=f"qty_{type_id}",
        )
        c3.button("Details", key=f"sheet_{type_id}", on_click=open_sheet, args=(type_id,), width="stretch")
        st.divider()

    if st.button("Compare my basket", type="primary"):
        st.switch_page(PAGES["compare"])
    show_pending_sheet()


# ============================================================
# PAGE: COMPARE
# ============================================================

def matrix_html(cards):
    chains = list(cards["chain"])
    head = "".join(
        f'<th style="--chain:{CHAIN_COLORS.get(c, INK)}"><span class="dot"></span>{chain_name(c)}</th>' for c in chains
    )
    body = []
    for type_id in basket:
        row = cost.loc[type_id, chains]
        best = row.min()
        cells = []
        for c in chains:
            v = row[c]
            if pd.isna(v):
                cells.append('<td class="none">—</td>')
            else:
                cls = ' class="best"' if v == best else ""
                cells.append(f'<td{cls}><span class="v">{eur(v)}</span><span class="u">{eur(ppu.at[type_id, c])}/{unit(type_id)}</span></td>')
        qty = st.session_state[f"qty_{type_id}"]
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
    complete = cards[cards["complete"]]
    if complete.empty:
        html('<div class="hero">No nearby store has your whole basket</div>')
    else:
        cheapest = complete.sort_values("basket_eur").iloc[0]
        html(f'<div class="hero">Your basket costs {eur(cheapest.basket_eur)} at {chain_name(cheapest.chain)}</div>')
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
    st.pills("See details for", basket, format_func=label, key="sheet_pick", selection_mode="single",
             on_change=pick_sheet)

    st.write("")
    right, chart_area = st.columns([2, 3], gap="large")
    with right:
        st.markdown("### Your best route")
        plans, plan_stores = plans_for(cost, candidates, HOME, st.session_state.max_stores)
        ranked = opt.score(plans, st.session_state.cost_per_km, st.session_state.cost_per_stop)
        if ranked.empty:
            st.warning("No combination of nearby stores has your whole basket. Increase the distance or remove products.")
        else:
            best = ranked.iloc[0]
            single = ranked[ranked["n_stores"] == 1]
            if best["n_stores"] == 1:
                verdict = f"Go to {chain_name(best['chains'][0])} only"
                multi = ranked[ranked["n_stores"] > 1]
                saving = best["products_eur"] - multi["products_eur"].min() if not multi.empty else 0
                note = (
                    f"Splitting the shop would save {eur(saving)} on products, "
                    "but the extra trip and stop cost more than that."
                    if saving > 0.005 else "No other store beats the price of your products."
                )
            else:
                verdict = "Split the shop: " + " and ".join(chain_name(c) for c in best["chains"])
                note = (
                    f"You save {eur(single.iloc[0]['total_eur'] - best['total_eur'])} over the best single store, "
                    "counting the trip and the extra stop."
                    if not single.empty else "No single store has your whole basket."
                )
            html(
                f'<div class="panel"><div class="verdict">{verdict}</div>'
                f'{price_tag(best["total_eur"], small=True)}'
                f'<div class="small" style="margin:6px 0 10px">{eur(best["products_eur"])} products + '
                f'{eur(best["travel_eur"])} travel ({km_text(best["route_km"])}) + {eur(best["stops_eur"])} extra stops</div>'
                f'{route_html(best, plan_stores)}<div class="small">{note}</div></div>'
            )
        with st.expander("Calculation settings"):
            st.slider("Travel cost (€/km)", 0.0, 1.0, step=0.05, key="cost_per_km",
                      help="0 if you walk. About 0.25 €/km by car.")
            st.slider("Cost of each extra stop (€)", 0.0, 5.0, step=0.25, key="cost_per_stop",
                      help="Time, queues, parking… Splitting the shop only pays if it saves more than this.")
            st.radio("Maximum stores", [1, 2, 3], key="max_stores", horizontal=True)
            st.radio("Price of each product", ["Typical price", "Cheapest"], key="stat", horizontal=True,
                     help="Typical price = median of its type at the chain. Cheapest = the cheapest product of that type.")
            st.caption(f"Straight-line distances × {opt.DETOUR:.1f}, because streets are not straight. "
                       "It tries up to 3 stores per chain and every visiting order.")

    with chart_area:
        st.markdown("### Two questions about your basket")
        a, b = st.tabs(["Is it worth going further?", "When do two stores pay off?"])
        with a:
            st.altair_chart(charts.scatter(cards, CHAIN_COLORS, CHAIN_NAMES, eur, km_text), width="stretch", theme=None)
            st.caption("Each point is the nearest store of a chain. Faded points have another store that is both cheaper and closer.")
        with b:
            if ranked.empty:
                st.write("This needs at least one combination of stores with your whole basket.")
            else:
                chart, threshold = sensitivity_view(plans)
                st.altair_chart(chart, width="stretch", theme=None)
                st.caption(threshold)
    show_pending_sheet()


def sensitivity_view(plans):
    km_costs = np.round(np.arange(0, 1.01, 0.05), 2)
    sens = opt.sensitivity(plans, km_costs, st.session_state.cost_per_stop)
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
    ranked = opt.score(plans, st.session_state.cost_per_km, st.session_state.cost_per_stop)

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
    with left:
        if not ranked.empty:
            html(f'<div class="small" style="margin-bottom:8px">The green line is your best route: '
                 f'{" and ".join(chain_name(c) for c in ranked.iloc[0]["chains"])}.</div>')
        rows = []
        for i, s in enumerate(shown.itertuples()):
            sel = " sel" if i == picked else ""
            rows.append(
                f'<div class="store-row{sel}" style="--chain:{CHAIN_COLORS.get(s.supermarket, INK)}">'
                f'<div><span class="dot"></span><b>{chain_name(s.supermarket)}</b> '
                f'<span class="small">{km_text(s.distance_km)}</span>'
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
    """'2026-09-23' -> '23 Sep'."""
    return pd.Timestamp(date).strftime("%d %b").lstrip("0")


def page_trends():
    header()
    try:
        changes = load_changes()
    except FileNotFoundError:
        st.error("The price changes are missing. From the project folder, run: python -m src.analysis.history")
        return
    summary = history.chain_summary(changes)
    top = summary.iloc[0]
    html(f'<div class="hero">{top.changed_pct:.0f}% of {chain_name(top.chain)} prices changed between collections</div>')
    html('<div class="lede">Three chains were collected on two dates, so this page shows price changes, '
         "not a trend line yet. The line chart comes once weekly collections add more dates.</div>")

    tags = []
    for r in summary.itertuples():
        tags.append(
            f'<div class="tag" style="--chain:{CHAIN_COLORS.get(r.chain, INK)}"><div class="chain">{chain_name(r.chain)}</div>'
            f'<div class="price sm"><span class="eu">{r.average_pct:+.1f}</span><span class="cur">%</span></div>'
            f'<div class="diff" style="color:var(--muted)">average price change</div>'
            f'<div class="meta"><span>{day(r.old_date)} to {day(r.new_date)}</span><span>{r.products} products</span></div></div>'
        )
    html(f'<div class="tags">{"".join(tags)}</div>')

    st.markdown("### How many prices moved")
    show(charts.moved_share(summary, CHAIN_NAMES))

    st.markdown("### Biggest changes")
    rises, falls = history.biggest_moves(changes)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Rises**")
        show(charts.moves(rises, CHAIN_NAMES, charts.RISE))
    with right:
        st.markdown("**Falls**")
        show(charts.moves(falls, CHAIN_NAMES, charts.FALL))
    st.caption(f"{int(changes['held_back'].sum())} larger changes are held back for review, in case they are data errors.")

    st.markdown("### Change by category")
    show(charts.category_change(history.category_changes(changes), CHAIN_NAMES, insights.category_name))
    st.caption(f"Only cells with at least {history.MIN_CATEGORY_PRODUCTS} comparable products. Empty cells have too few.")

    st.markdown("### Offers today")
    table, former = history.offers()
    columns = st.columns(len(table) + 1)
    for column, r in zip(columns, table.itertuples()):
        column.metric(f"{chain_name(r.chain)} products on offer", f"{r.on_offer_pct:.1f}%", help=f"{r.on_offer} of {r.products}")
        column.caption(f"Median discount {r.median_discount_pct:.0f}%")
    columns[-1].metric("Mercadona with a higher former price", f"{former['products']}")
    columns[-1].caption(f"Median {former['median_drop_pct']:.0f}% lower now")
    st.caption("Average change is a geometric mean. Mercadona compares two collectors joined on product name, "
               "so it covers fewer products; treat it as a rough measure.")


# ============================================================
# PAGE: ANALYSIS
# ============================================================

@st.cache_data(show_spinner=False)
def route_comparison(stores, home, cost_per_km, cost_per_stop):
    return insights.greedy_vs_exact(df, stores, home, cost_per_km, cost_per_stop)


def section_chains():
    table, n_types = insights.basket_index(df)
    bm = table.set_index("chain").loc["Bm"]
    st.markdown(f"### The method moves BM from {bm['median']:.0f} to {bm['cheapest']:.0f}")
    show(charts.index_dumbbell(table, CHAIN_COLORS, CHAIN_NAMES))
    st.caption(f"Sum of {n_types} product types that all 7 chains sell. A wide gap means a chain's median is pulled up "
               "by the products it stocks, not only by its prices.")


def section_categories():
    winners = insights.category_winners(df)
    disagree = int((~winners["agree"]).sum())
    st.markdown(f"### The cheapest chain depends on the category, and on the method in {disagree} of {len(winners)} categories")
    choice = st.segmented_control("Price of each type", ["Median", "Cheapest product"], default="Median", key="category_stat")
    stat = "min" if choice == "Cheapest product" else "median"
    show(charts.category_heatmap(insights.category_prices(df, stat), CHAIN_NAMES, insights.category_name))
    st.caption(f'Each type is compared on its own, then a median is taken. "few" means fewer than {insights.MIN_PRODUCTS} '
               "products. Colors stop at 2.5× the cheapest; very large gaps can be data problems.")
    table = winners.assign(
        Category=winners["category"].map(insights.category_name),
        Median=winners["median"].map(chain_name), Cheapest=winners["min"].map(chain_name),
    ).sort_values("Category")
    with st.expander("Winner of each category"):
        st.dataframe(table[["Category", "Median", "Cheapest"]], hide_index=True, width="stretch")


def section_own_brand():
    table = insights.own_brand_gap(df)
    best = table.iloc[0]
    st.markdown(f"### Own brand saves most at {chain_name(best.chain)}: {best.gap_pct:.0f}% on a typical type")
    left, right = st.columns(2, gap="large")
    with left:
        show(charts.own_brand(table, CHAIN_COLORS, CHAIN_NAMES))
    with right:
        show(charts.own_share(table, CHAIN_COLORS, CHAIN_NAMES))
    st.caption("Saving: median across the types where the chain sells both own and national brands.")


def section_spread():
    table = insights.price_spread(df)
    more = int((table["within"] > table["between"]).sum())
    st.markdown(f"### In {more} of {len(table)} categories, choosing the product matters more than choosing the store")
    show(charts.spread(table, insights.category_name))
    st.caption("Between products: p90 against p10 of a type within one chain. Between chains: dearest against "
               "cheapest chain median of a type. Both are medians across the category.")


def section_barcode():
    pairs = insights.same_barcode()
    gap, by_type = pairs["gap_pct"].median(), insights.type_gap(df)
    st.markdown(f"### On the same barcode, BM is {gap:.0f}% dearer than Consum; our type medians say {by_type:.0f}%")
    show(charts.barcode_gap(pairs))
    same = (pairs["gap_pct"].abs() <= 0.5).mean()
    st.caption(f"{len(pairs)} products with the same barcode at both chains; {100 * same:.0f}% have the same price. "
               "This is the only exact comparison possible today, and it is why product matching comes next.")


def section_route():
    st.markdown("### What the exact optimizer adds")
    if candidates["supermarket"].nunique() < 2:
        st.write("This needs stores from two chains within your range. Increase the maximum distance in the bar above.")
        return
    with st.spinner("Comparing random baskets…"):
        table = route_comparison(candidates, HOME, st.session_state.cost_per_km, st.session_state.cost_per_stop)
    if table.empty:
        st.write("None of the random baskets could be bought within your range. Increase the maximum distance.")
        return
    st.markdown(f"It saves a mean **{eur(table['saving_eur'].mean())}** per basket over shopping from a price list, "
                f"with {table['exact_stops'].mean():.1f} stops instead of {table['greedy_stops'].mean():.1f}.")
    show(charts.route_savings(table))
    st.caption(f"{len(table)} random baskets of 5 product types near your address. Price list means buying each item "
               "at its cheapest chain, with the same limit of 3 stores. Totals include travel and extra stops.")


def section_coverage():
    table = insights.coverage(df)
    st.markdown(f"### {int(table['products'].sum()):,} comparable products behind these charts")
    show(charts.coverage(table, CHAIN_NAMES, insights.category_name))
    st.caption("Number of products per category and chain. Small numbers mean a price here rests on few products.")


def page_analysis():
    header()
    html('<div class="hero">What the prices tell us</div>')
    html('<div class="lede">Across all products, not just your basket. These results are provisional until '
         "the product classifier is fully evaluated.</div>")
    tabs = st.tabs(["Chains", "Categories", "Own brand", "Product or store", "Same product", "Route savings", "Data coverage"])
    sections = [section_chains, section_categories, section_own_brand, section_spread,
                section_barcode, section_route, section_coverage]
    for tab, section in zip(tabs, sections):
        with tab:
            section()


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
