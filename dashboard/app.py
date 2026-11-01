# Dashboard principal del proyecto (la versión anterior, obsoleta, está en dashboard/app_legacy.py).
# PARA LANZARLO:  streamlit run dashboard/app.py --theme.primaryColor "#17242D" --theme.base light
#   (sin los flags también funciona; el tema se aplica desde el código a partir de la primera carga)
#
# Qué cambia respecto a app_legacy.py:
#   - Navegación arriba (Cesta, Comparar, Mapa, Tendencias); sin barra lateral.
#   - La cesta es el centro: se busca entre los 70 tipos de producto, no entre los 20 términos.
#   - Comparar junta Overview + Visualizations + Optimizer: etiquetas de precio por cadena,
#     matriz producto × cadena, recomendación de ruta y dos gráficos de análisis.
#   - El optimizador vive en src/analysis/optimizer.py (varias tiendas por cadena, coste por parada,
#     factor de rodeo, ruta mostrada tramo a tramo).
#   - Si Overpass o Nominatim fallan, se usan data/stores_madrid.csv y una dirección guardada.
#   - Un solo color por cadena en todas partes, precios en formato español.

import sys
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import re
import requests
import streamlit as st
from streamlit_folium import st_folium

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from src.analysis import optimizer as opt  # noqa: E402
from src.analysis import prices as pricing  # noqa: E402

# Tema solo para esta app (un .streamlit/config.toml cambiaría también app_legacy.py)
from streamlit import config as st_config  # noqa: E402

for option, value in {"theme.base": "light", "theme.primaryColor": "#17242D", "theme.backgroundColor": "#F1F4F1",
                      "theme.secondaryBackgroundColor": "#E6EBE7", "theme.textColor": "#17242D"}.items():
    st_config.set_option(option, value)

st.set_page_config(page_title="Comparador de supermercados", page_icon="🛒", layout="wide")


# ============================================================
# ESTILO: un solo sitio para colores, tipografía y componentes
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
CHAIN_NAMES = {"Bm": "BM"}  # cómo se escribe cada cadena en pantalla
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
.price .eu{font-size:3.3rem;} .price .ct{font-size:1.45rem;vertical-align:1.55rem;margin-left:1px;} .price .cur{font-size:1.3rem;margin-left:4px;}
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
.placeholder{border:2px dashed #B9C4C9;border-radius:14px;padding:26px 28px;background:repeating-linear-gradient(-45deg,#F7F9F7,#F7F9F7 12px,#F1F4F1 12px,#F1F4F1 24px);}
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
    """6.68 -> '6,68 €' (formato español)."""
    s = f"{x:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} €"


def km_text(x):
    return f"{x:,.1f} km".replace(".", ",") if x >= 1 else f"{x * 1000:.0f} m"


def price_tag(x, small=False):
    """Precio como en la etiqueta de la estantería: euros grandes, céntimos pequeños arriba."""
    euros, cents = f"{x:.2f}".split(".")
    return (
        f'<div class="price{" sm" if small else ""}"><span class="eu">{euros}</span>'
        f'<span class="ct">,{cents}</span><span class="cur">€</span></div>'
    )


# ============================================================
# DATOS DE PRODUCTOS
# ============================================================

UNIT_TEXT = {"LITRO": "l", "KILO": "kg", "UNIDAD": "ud.", "LAVADO": "lavados"}
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
    st.error("Faltan los datos normalizados. Ejecuta desde la carpeta del proyecto: python -m src.pipeline.build")
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
    """Tipos × cadenas, € por unidad (mediana o mínimo)."""
    return pricing.price_table(df, list(type_ids), stat).reindex(index=list(type_ids), columns=CHAINS)


# ============================================================
# ESTADO COMPARTIDO ENTRE PÁGINAS
# ============================================================
# Streamlit borra el estado de los widgets de una página que no se está mostrando; reasignarlo
# al principio de cada ejecución lo conserva (la cesta, cantidades y ajustes viajan entre páginas).

DEFAULTS = {
    "basket": list(DEFAULT_BASKET),
    "address": "Avenida de la Universidad, 30, Leganés",
    "radius": 1.5,
    "cost_per_km": 0.25,
    "cost_per_stop": 1.0,
    "max_stores": 2,
    "stat": "Precio típico",
    "sort": "Precio",
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
STAT = "min" if st.session_state.stat == "Lo más barato" else "median"


# ============================================================
# UBICACIÓN Y TIENDAS (con plan B si las APIs fallan)
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
    """Supermercados de OpenStreetMap alrededor del punto; si Overpass falla, la lista guardada."""
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
    st.error("El servicio de direcciones no responde. Prueba otra vez en unos segundos o usa la dirección de ejemplo.")
    st.stop()
if location is None:
    st.error("No encontramos esa dirección. Escribe calle, número y localidad, por ejemplo «Calle Mayor 1, Madrid».")
    st.stop()
user_lat, user_lon, _, offline_geo = location
HOME = (user_lat, user_lon)

with st.spinner("Buscando supermercados cerca de ti…"):
    all_stores, offline_stores = load_stores(user_lat, user_lon)

candidates = opt.candidate_stores(all_stores, HOME, per_chain=3, max_km=st.session_state.radius)
nearest = candidates.drop_duplicates("supermarket")  # la tienda más cercana de cada cadena


# ============================================================
# CÁLCULOS DE LA CESTA (compartidos por todas las páginas)
# ============================================================

ppu = unit_prices(tuple(basket), STAT) if basket else pd.DataFrame(columns=CHAINS)
cost = ppu.mul(quantities, axis=0)  # € de cada producto en la cantidad pedida


def chain_cards():
    """Una fila por cadena con tienda en el radio: coste de la cesta, cobertura y coste con el viaje."""
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
    key = {"Precio": "basket_eur", "Distancia": "distance_km", "Equilibrio": "balanced_eur"}[st.session_state.sort or "Precio"]
    return cards.sort_values(["complete", key], ascending=[False, True]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def plans_for(cost, stores, home, max_stores):
    return opt.enumerate_plans(cost, stores, home, max_stores)


# ============================================================
# CABECERA COMÚN: dirección y cesta siempre a mano
# ============================================================

def header():
    left, mid, right = st.columns([3.3, 4.7, 2], vertical_alignment="center")
    left.markdown(
        '<div style="font-stretch:118%;font-weight:800;font-size:1.12rem;white-space:nowrap;">🛒 Comparador de supermercados</div>',
        unsafe_allow_html=True,
    )
    with mid.popover(f"📍 Comprando cerca de {st.session_state.address}, hasta {km_text(st.session_state.radius)}", width="stretch"):
        st.text_input("Dirección", key="address")
        st.slider("Distancia máxima a la tienda (km)", 0.3, 5.0, step=0.1, key="radius")
    with right.popover(f"Cesta ({len(basket)})", icon=":material/shopping_basket:", width="stretch"):
        if not basket:
            st.write("Tu cesta está vacía.")
        for t in basket:
            st.markdown(f"**{label(t)}** · {st.session_state[f'qty_{t}']:g} {unit(t)}")
    if offline_geo:
        st.caption("El servicio de direcciones no responde: usamos la ubicación guardada de esta dirección.")
    if offline_stores:
        st.caption("OpenStreetMap no responde: usamos la lista de tiendas de Madrid guardada en data/stores_madrid.csv.")


# ============================================================
# FICHA DE PRODUCTO (ventana emergente)
# ============================================================

@st.dialog("Ficha de producto", width="large")
def product_sheet(type_id):
    rows = pricing.type_rows(df, type_id)
    st.markdown(f"## {label(type_id)}")
    st.caption(f"{len(rows)} productos en {rows['supermarket'].nunique()} cadenas · precios por {unit(type_id)}")

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
            f'<div class="small">desde {eur(r["min"])}, {int(r["size"])} productos</div></div></div>'
        )
    html("".join(lines))

    own = rows[rows["is_private_label"] == 1]["price_per_unit"].median()
    brand = rows[rows["is_private_label"] == 0]["price_per_unit"].median()
    if pd.notna(own) and pd.notna(brand) and brand > 0:
        gap = 100 * (brand - own) / brand
        st.info(
            f"La marca blanca cuesta de mediana {eur(own)}/{unit(type_id)} frente a {eur(brand)} de las marcas: "
            f"un {gap:.0f} % {'menos' if gap >= 0 else 'más'}."
        )

    st.markdown("#### Los cinco más baratos")
    cheapest = rows.nsmallest(5, "price_per_unit")
    items = []
    for _, p in cheapest.iterrows():
        tag = '<span class="pl">Marca blanca</span>' if p["is_private_label"] == 1 else ""
        items.append(
            f'<div class="store-row" style="--chain:{CHAIN_COLORS.get(p["supermarket"], INK)}">'
            f'<div><span class="dot"></span>{p["name"]}{tag}<div class="addr">{chain_name(p["supermarket"])}</div></div>'
            f'<div style="text-align:right"><b>{eur(p["price_per_unit"])}/{unit(type_id)}</b>'
            f'<div class="addr">{eur(p["price"]) if pd.notna(p["price"]) else ""} el envase</div></div></div>'
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
# PÁGINA: CESTA
# ============================================================

def page_basket():
    header()
    html('<div class="hero">¿Qué vas a comprar?</div>')
    html(f'<div class="lede">Elige entre {len(TYPES)} tipos de producto de {len(CHAINS)} cadenas. '
         "Comparamos cada tipo con su equivalente en las demás, no la marca.</div>")

    st.multiselect(
        "Buscar productos", TYPES.index, key="basket", format_func=label,
        placeholder="Busca: leche, atún en aceite, pan de molde…", label_visibility="collapsed",
    )
    if not basket:
        html('<div class="panel">Tu cesta está vacía. Busca arriba un producto para empezar.</div>')
        return

    st.write("")
    cheapest_chain = ppu.idxmin(axis=1)
    for type_id in basket:
        c1, c2, c3 = st.columns([6, 2, 1.3], vertical_alignment="center")
        best = ppu.loc[type_id].min()
        hint = (
            f'desde <b>{eur(best)}/{unit(type_id)}</b> en {chain_name(cheapest_chain[type_id])}'
            if pd.notna(best) else "sin precio en ninguna cadena"
        )
        c1.markdown(
            f'<div class="item-row"><span class="name">{label(type_id)}</span></div><div class="from">{hint}</div>',
            unsafe_allow_html=True,
        )
        is_count = TYPES.at[type_id, "unit"] in ("UNIDAD", "LAVADO")
        c2.number_input(
            f"Cantidad ({unit(type_id)})", min_value=0.25 if not is_count else 1.0,
            step=0.5 if not is_count else 1.0, key=f"qty_{type_id}",
        )
        c3.button("Ficha", key=f"sheet_{type_id}", on_click=open_sheet, args=(type_id,), width="stretch")
        st.divider()

    if st.button("Comparar mi cesta", type="primary"):
        st.switch_page(PAGES["compare"])
    show_pending_sheet()


# ============================================================
# PÁGINA: COMPARAR
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
    body.append(f'<tr class="total"><td>Total cesta</td>{totals}</tr>')
    return f'<div class="matrix-wrap"><table class="matrix"><tr><th>Producto</th>{head}</tr>{"".join(body)}</table></div>'


def route_html(plan, stores):
    steps = opt.describe_route(plan, stores)
    items = [f'<li style="--chain:{INK}"><span class="stop"></span><b>Sales de casa</b></li>']
    for _, to, km in steps:
        what = [label(t) for t, c in plan["buy_at"].items() if c == to]
        color = CHAIN_COLORS.get(to, INK)
        title = "Vuelves a casa" if to == opt.HOME else f"{chain_name(to)}"
        detail = f'<div class="small">{", ".join(what)}</div>' if what else ""
        items.append(
            f'<li style="--chain:{color if to != opt.HOME else INK}"><span class="stop"></span>'
            f'<b>{title}</b> <span class="km">a {km_text(km)}</span>{detail}</li>'
        )
    return f'<ul class="route">{"".join(items)}</ul>'


def page_compare():
    header()
    if not basket:
        html('<div class="hero">Tu cesta está vacía</div>')
        if st.button("Añadir productos", type="primary"):
            st.switch_page(PAGES["basket"])
        return
    if nearest.empty:
        html('<div class="hero">No hay supermercados en tu radio</div>')
        st.write("Amplía la distancia máxima en la barra de arriba.")
        return

    cards = chain_cards()
    complete = cards[cards["complete"]]
    if complete.empty:
        html('<div class="hero">Ninguna tienda cercana tiene toda tu cesta</div>')
    else:
        cheapest = complete.sort_values("basket_eur").iloc[0]
        html(f'<div class="hero">Tu cesta cuesta {eur(cheapest.basket_eur)} en {chain_name(cheapest.chain)}</div>')
    n_products = len(basket)
    html(f'<div class="lede">{n_products} productos, {len(cards)} cadenas con tienda a menos de '
         f'{km_text(st.session_state.radius)}. Precio de cada producto: '
         f'{"el más barato de su tipo" if STAT == "min" else "la mediana de su tipo"} en cada cadena.</div>')

    st.segmented_control("Ordenar por", ["Precio", "Distancia", "Equilibrio"], key="sort",
                         help="Equilibrio suma la cesta y el coste de ir y volver a la tienda.")
    best_price = cards.loc[cards["complete"], "basket_eur"].min()
    tags = []
    for i, r in cards.iterrows():
        win = r.complete and r.basket_eur == best_price
        if not r.complete:
            diff, cls = f"Sin {len(basket) - r.have} de tus {len(basket)} productos: no comparable", "partial"
        elif win:
            diff, cls = "La más barata", "win"
        else:
            diff, cls = f"+{eur(r.basket_eur - best_price)} ({100 * (r.basket_eur / best_price - 1):.0f} %)", ""
        badge = '<span class="badge">Mejor precio</span>' if win else ""
        total = price_tag(r.basket_eur, small=not r.complete) if pd.notna(r.basket_eur) else price_tag(0, small=True)
        tags.append(
            f'<div class="tag {cls}" style="--chain:{CHAIN_COLORS.get(r.chain, INK)}">{badge}'
            f'<div class="chain">{chain_name(r.chain)}</div>{total}<div class="diff">{diff}</div>'
            f'<div class="meta"><span>{km_text(r.distance_km)}</span><span>{r.have}/{len(basket)} productos</span></div></div>'
        )
    html(f'<div class="tags">{"".join(tags)}</div>')

    st.markdown("### Dónde es más barato cada producto")
    html(matrix_html(cards))
    st.pills("Ver ficha de", basket, format_func=label, key="sheet_pick", selection_mode="single",
             on_change=pick_sheet)

    st.write("")
    right, charts = st.columns([2, 3], gap="large")
    with right:
        st.markdown("### Tu mejor ruta")
        plans, plan_stores = plans_for(cost, candidates, HOME, st.session_state.max_stores)
        ranked = opt.score(plans, st.session_state.cost_per_km, st.session_state.cost_per_stop)
        if ranked.empty:
            st.warning("Ninguna combinación de tiendas cercanas tiene toda tu cesta. Amplía la distancia o quita productos.")
        else:
            best = ranked.iloc[0]
            single = ranked[ranked["n_stores"] == 1]
            if best["n_stores"] == 1:
                verdict = f"Ve solo a {chain_name(best['chains'][0])}"
                multi = ranked[ranked["n_stores"] > 1]
                saving = best["products_eur"] - multi["products_eur"].min() if not multi.empty else 0
                note = (
                    f"Repartir la compra ahorraría {eur(saving)} en productos, "
                    "pero el viaje extra y la parada no lo compensan."
                    if saving > 0.005 else "Ninguna otra tienda mejora el precio de tus productos."
                )
            else:
                verdict = "Reparte la compra: " + " y ".join(chain_name(c) for c in best["chains"])
                note = (
                    f"Ahorras {eur(single.iloc[0]['total_eur'] - best['total_eur'])} frente a la mejor tienda única, "
                    "contando el viaje y la parada extra."
                    if not single.empty else "Ninguna tienda sola tiene toda tu cesta."
                )
            html(
                f'<div class="panel"><div class="verdict">{verdict}</div>'
                f'{price_tag(best["total_eur"], small=True)}'
                f'<div class="small" style="margin:6px 0 10px">{eur(best["products_eur"])} de productos + '
                f'{eur(best["travel_eur"])} de viaje ({km_text(best["route_km"])}) + {eur(best["stops_eur"])} por paradas extra</div>'
                f'{route_html(best, plan_stores)}<div class="small">{note}</div></div>'
            )
        with st.expander("Ajustes del cálculo"):
            st.slider("Coste de desplazarte (€/km)", 0.0, 1.0, step=0.05, key="cost_per_km",
                      help="0 si vas andando. Unos 0,25 €/km en coche.")
            st.slider("Coste de cada parada extra (€)", 0.0, 5.0, step=0.25, key="cost_per_stop",
                      help="Tiempo, cola, aparcar… Repartir la compra solo compensa si ahorra más que esto.")
            st.radio("Tiendas como máximo", [1, 2, 3], key="max_stores", horizontal=True)
            st.radio("Precio de cada producto", ["Precio típico", "Lo más barato"], key="stat", horizontal=True,
                     help="Precio típico = mediana de su tipo en la cadena. Lo más barato = el producto más barato de ese tipo.")
            st.caption(f"Distancias en línea recta × {opt.DETOUR:.1f} (las calles no van en línea recta). "
                       "Se prueban hasta 3 tiendas por cadena y todos los órdenes de visita.")

    with charts:
        st.markdown("### Dos preguntas sobre tu cesta")
        a, b = st.tabs(["¿Merece la pena ir más lejos?", "¿Cuándo compensa ir a dos tiendas?"])
        with a:
            st.altair_chart(scatter_chart(cards), width="stretch", theme=None)
            st.caption("Cada punto es la tienda más cercana de una cadena. Las difuminadas tienen otra tienda más barata y más cerca.")
        with b:
            if ranked.empty:
                st.write("Hace falta al menos una combinación de tiendas con toda tu cesta.")
            else:
                chart, threshold = sensitivity_chart(plans)
                st.altair_chart(chart, width="stretch", theme=None)
                st.caption(threshold)
    show_pending_sheet()


def styled(chart):
    return chart.configure(font="Archivo", background="#FFFFFF").configure_view(strokeWidth=0).configure_axis(
        labelColor=MUTED, titleColor=MUTED, gridColor="#E8EDEA", domainColor="#C9D2CD", labelFontSize=12, titleFontSize=12,
    )


def scatter_chart(cards):
    c = cards[cards["complete"]].copy()
    c["Cadena"] = c["chain"].map(chain_name)
    c["dominada"] = [
        bool(((c["basket_eur"] <= r.basket_eur) & (c["distance_km"] <= r.distance_km)
              & ((c["basket_eur"] < r.basket_eur) | (c["distance_km"] < r.distance_km))).any())
        for r in c.itertuples()
    ]
    c["Cesta"] = c["basket_eur"].map(eur)
    c["Distancia"] = c["distance_km"].map(km_text)
    scale = alt.Scale(domain=[chain_name(k) for k in CHAIN_COLORS], range=list(CHAIN_COLORS.values()))
    base = alt.Chart(c).encode(
        x=alt.X("distance_km:Q", title="Distancia a la tienda (km)", scale=alt.Scale(zero=True)),
        y=alt.Y("basket_eur:Q", title="Precio de la cesta (€)", scale=alt.Scale(zero=False)),
        color=alt.Color("Cadena:N", scale=scale, legend=None),
        opacity=alt.condition("datum.dominada", alt.value(0.28), alt.value(1)),
        tooltip=["Cadena", "Cesta", "Distancia"],
    )
    points = base.mark_circle(size=230)
    labels = base.mark_text(align="left", dx=11, fontSize=12, fontWeight=700).encode(text="Cadena")
    return styled((points + labels).properties(height=280))


def sensitivity_chart(plans):
    km_costs = np.round(np.arange(0, 1.01, 0.05), 2)
    sens = opt.sensitivity(plans, km_costs, st.session_state.cost_per_stop)
    long = pd.concat([
        pd.DataFrame({"€/km": sens["cost_per_km"], "Total": sens["best_total"], "Plan": "Mejor plan", "z": 0}),
        pd.DataFrame({"€/km": sens["cost_per_km"], "Total": sens["single_total"], "Plan": "Una sola tienda", "z": 1}),
    ])
    lines = alt.Chart(long).mark_line(strokeWidth=3).encode(
        order="z:Q",
        x=alt.X("€/km:Q", title="Coste de desplazarte (€/km)"),
        y=alt.Y("Total:Q", title="Coste total (€)", scale=alt.Scale(zero=False)),
        color=alt.Color("Plan:N", scale=alt.Scale(domain=["Una sola tienda", "Mejor plan"], range=["#9AA7AE", BEST]),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("Plan:N", scale=alt.Scale(domain=["Una sola tienda", "Mejor plan"], range=[[7, 5], [1, 0]]), legend=None),
    )
    now = alt.Chart(pd.DataFrame({"x": [st.session_state.cost_per_km]})).mark_rule(color=INK, strokeDash=[2, 3]).encode(x="x:Q")
    split = sens[sens["best_n_stores"] > 1]
    if split.empty:
        text = "Con tu coste por parada, ir a una sola tienda es siempre lo mejor."
    elif len(split) == len(sens):
        text = "Repartir la compra compensa incluso a 1 €/km."
    else:
        text = f"Repartir la compra compensa hasta unos {split['cost_per_km'].max():.2f} €/km; a partir de ahí, mejor una sola tienda.".replace(".", ",", 1)
    return styled((lines + now).properties(height=280)), text + " La línea punteada vertical es tu ajuste actual."


# ============================================================
# PÁGINA: MAPA
# ============================================================

def page_map():
    header()
    html('<div class="hero">Tu cesta, tienda por tienda</div>')
    html(f'<div class="lede">Cada chincheta muestra lo que te costaría la cesta en esa tienda. '
         f'El círculo marca tu distancia máxima ({km_text(st.session_state.radius)}).</div>')
    if nearest.empty:
        st.write("No hay supermercados en tu radio. Amplía la distancia máxima en la barra de arriba.")
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
    folium.Marker(HOME, tooltip="Tu dirección", icon=folium.DivIcon(
        html=f'<div style="background:{INK};color:#fff;font:700 12px Archivo,sans-serif;padding:5px 9px;border-radius:14px;'
             'white-space:nowrap;transform:translate(-50%,-50%);display:inline-block">Casa</div>'), z_index_offset=1000).add_to(m)

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
            html(f'<div class="small" style="margin-bottom:8px">La línea verde es tu mejor ruta: '
                 f'{" y ".join(chain_name(c) for c in ranked.iloc[0]["chains"])}.</div>')
        rows = []
        for i, s in enumerate(shown.itertuples()):
            sel = " sel" if i == picked else ""
            rows.append(
                f'<div class="store-row{sel}" style="--chain:{CHAIN_COLORS.get(s.supermarket, INK)}">'
                f'<div><span class="dot"></span><b>{chain_name(s.supermarket)}</b> '
                f'<span class="small">{km_text(s.distance_km)}</span>'
                f'<div class="addr">{s.address or "Ver en el mapa"}</div></div>'
                f'<div><b>{pin_text(s.supermarket)}</b></div></div>'
            )
        html('<div class="panel" style="padding:0;max-height:560px;overflow-y:auto">' + "".join(rows) + "</div>")
        st.caption("Toca una chincheta para resaltarla en la lista.")


# ============================================================
# PÁGINA: TENDENCIAS (pendiente de decidir, ver §2.3 del plan)
# ============================================================

def page_trends():
    header()
    html('<div class="hero">Tendencias de precios</div>')
    html('<div class="lede">Esta sección está pendiente. Ahora mismo tenemos una sola toma de precios, '
         "y una tendencia necesita varias. Preferimos no enseñar números inventados.</div>")
    html(
        '<div class="placeholder">'
        '<div class="verdict">Decisión pendiente del grupo</div>'
        '<p><b>Opción A.</b> Guardar la fecha de cada recogida y repetir cada semana las APIs estables '
        "(BM, Consum, Mercadona). Con 3 o 4 tomas ya se puede enseñar:</p>"
        "<ul><li>la evolución del coste de tu cesta en cada cadena,</li>"
        "<li>qué productos han subido o bajado desde la última toma,</li>"
        "<li>el histórico de precio en la ficha de cada producto.</li></ul>"
        "<p><b>Opción B.</b> Quitar esta sección y explicar en la memoria que los datos son una foto fija "
        "y que seguir los precios en el tiempo es trabajo futuro.</p></div>"
    )


# ============================================================
# NAVEGACIÓN
# ============================================================

PAGES = {
    "basket": st.Page(page_basket, title="Cesta", icon=":material/shopping_basket:", url_path="cesta", default=True),
    "compare": st.Page(page_compare, title="Comparar", icon=":material/compare_arrows:", url_path="comparar"),
    "map": st.Page(page_map, title="Mapa", icon=":material/map:", url_path="mapa"),
    "trends": st.Page(page_trends, title="Tendencias", icon=":material/trending_up:", url_path="tendencias"),
}
st.navigation(list(PAGES.values()), position="top").run()

st.markdown("---")
st.caption("Tiendas: © OpenStreetMap contributors. Precios: recogidos por web scraping y APIs de cada cadena.")
