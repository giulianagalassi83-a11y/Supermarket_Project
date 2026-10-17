# PARA LANZARLO PONER EN LA TERMINAL: streamlit run dashboard/app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import requests
import re
import folium
from streamlit_folium import st_folium


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Optimizador Inteligente de Supermercados",
    layout="wide"
)

st.title("🛒 Optimizador Inteligente de Supermercados")

st.markdown(
    "Calcula el coste de tu cesta y encuentra automáticamente "
    "los supermercados más cercanos a tu dirección."
)


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path.cwd()

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# Comprobamos que existe la carpeta de datos necesarios

if not PROCESSED_DIR.exists():

    st.error(
        f"No se encuentra la carpeta de datos en: "
        f"{PROCESSED_DIR}"
    )

    st.stop()


# ============================================================
# CARGAR DATOS DE PRODUCTOS
# ============================================================

@st.cache_data
def load_data():

    files = {

        "dia":
            PROCESSED_DIR / "dia_products.csv",

        "mercadona":
            PROCESSED_DIR / "mercadona_products.csv",

        "aldi":
            PROCESSED_DIR / "aldi_products.csv",

        "alcampo":
            PROCESSED_DIR / "alcampo_products.csv",

        "ahorramas":
            PROCESSED_DIR / "ahorramas_products.csv",

        "bm":
            PROCESSED_DIR / "bm_products.csv",

        "consum":
            PROCESSED_DIR / "consum_products.csv"

    }

    dfs = []


    # --------------------------------------------------------
    # LEER CSV DE CADA SUPERMERCADO
    # --------------------------------------------------------

    for name, path in files.items():

        if path.exists():

            try:

                df = pd.read_csv(path)


                # Si el CSV no tiene columna supermarket,
                # usamos el nombre del fichero.

                if "supermarket" not in df.columns:

                    supermarket_names = {

                        "dia":
                            "Dia",

                        "mercadona":
                            "Mercadona",

                        "aldi":
                            "Aldi",

                        "alcampo":
                            "Alcampo",

                        "ahorramas":
                            "Ahorramás",

                        "bm":
                            "Bm",

                        "consum":
                            "Consum"

                    }

                    df["supermarket"] = (
                        supermarket_names.get(
                            name,
                            name.title()
                        )
                    )


                dfs.append(df)


            except Exception as e:

                st.warning(
                    f"No se pudo leer {path.name}: {e}"
                )


    if not dfs:

        return pd.DataFrame()


    # --------------------------------------------------------
    # LIMPIEZA
    # --------------------------------------------------------

    for df in dfs:

        if "supermarket" in df.columns:

            df["supermarket"] = (
                df["supermarket"]
                .astype(str)
                .str.strip()
                .str.title()
            )


        if "searched_item" in df.columns:

            df["searched_item"] = (
                df["searched_item"]
                .astype(str)
                .str.strip()
                .str.lower()
            )


        if "price_per_unit" in df.columns:

            df["price_per_unit"] = pd.to_numeric(
                df["price_per_unit"],
                errors="coerce"
            )


    # --------------------------------------------------------
    # COLUMNAS COMUNES
    # --------------------------------------------------------

    COMMON_COLUMNS = [

        "supermarket",
        "searched_item",
        "name",
        "price",
        "price_per_unit",
        "unit"

    ]


    valid_dfs = []


    for df in dfs:

        available_columns = [

            c
            for c in COMMON_COLUMNS
            if c in df.columns

        ]


        valid_dfs.append(
            df[available_columns].copy()
        )


    all_products = pd.concat(
        valid_dfs,
        ignore_index=True
    )


    # --------------------------------------------------------
    # COMPROBAR PRICE_PER_UNIT
    # --------------------------------------------------------

    if "price_per_unit" not in all_products.columns:

        st.error(
            "Los CSV no contienen la columna "
            "'price_per_unit'."
        )

        return pd.DataFrame()


    all_products["price_per_unit"] = pd.to_numeric(
        all_products["price_per_unit"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # ELIMINAR VALORES IMPOSIBLES
    # --------------------------------------------------------

    clean_products = all_products[
        (all_products["price_per_unit"] > 0.05)
        &
        (all_products["price_per_unit"] <= 50.0)
    ].copy()


    return clean_products


df = load_data()


if df.empty:

    st.error(
        "No hay datos cargados en la carpeta "
        "data/processed."
    )

    st.stop()


# ============================================================
# GEOCODIFICAR DIRECCIÓN CON NOMINATIM (DEFINIDA ANTES DE USARSE)
# ============================================================

@st.cache_data(ttl=3600)
def geocode_address(address):

    url = (
        "https://nominatim.openstreetmap.org/search"
    )

    params = {

        "q": address,

        "format": "jsonv2",

        "limit": 1,

        "countrycodes": "es"

    }

    headers = {

        "User-Agent":
            "UC3M-Supermarket-Project/1.0"

    }

    try:

        response = requests.get(

            url,

            params=params,

            headers=headers,

            timeout=15

        )

        response.raise_for_status()

        results = response.json()

        if not results:

            return None

        result = results[0]

        return {

            "lat":
                float(result["lat"]),

            "lon":
                float(result["lon"]),

            "display_name":
                result["display_name"]

        }

    except Exception as e:

        st.error(
            f"Error al buscar la dirección: {e}"
        )

        return None


# ============================================================
# CARGAR SUPERMERCADOS DINÁMICAMENTE SEGÚN LA UBICACIÓN
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_stores_dynamic(user_lat, user_lon, radius_deg=0.15):
    """
    Consulta Overpass API dinámicamente en un radio alrededor 
    de la ubicación del usuario (válido para toda España sin timeouts).
    """
    south = user_lat - radius_deg
    north = user_lat + radius_deg
    west = user_lon - radius_deg
    east = user_lon + radius_deg

    overpass_url = "https://overpass-api.de/api/interpreter"
    
    query = f"""
    [out:json][timeout:60];
    (
        nwr["shop"="supermarket"]["brand"~"Mercadona|DIA|ALDI|Alcampo|Ahorramás|Ahorramas|BM|Consum",i]({south},{west},{north},{east});
        nwr["shop"="supermarket"]["name"~"Mercadona|DIA|ALDI|Alcampo|Ahorramás|Ahorramas|BM|Consum",i]({south},{west},{north},{east});
    );
    out center tags;
    """

    try:
        response = requests.post(
            overpass_url,
            data=query.encode("utf-8"),
            headers={"User-Agent": "UC3M-Supermarket-Project/1.0"},
            timeout=70
        )
        response.raise_for_status()
        data = response.json()
        elements = data.get("elements", [])
        
        stores = []
        for element in elements:
            tags = element.get("tags", {})
            if tags.get("shop", "") != "supermarket":
                continue
                
            name = tags.get("name") or tags.get("brand") or tags.get("operator")
            if not name:
                continue
                
            # Identificar cadena
            name_lower = str(name).lower()
            chain = None
            if "mercadona" in name_lower: chain = "Mercadona"
            elif "ahorramas" in name_lower: chain = "Ahorramás"
            elif "alcampo" in name_lower: chain = "Alcampo"
            elif re.search(r"\baldi\b", name_lower): chain = "Aldi"
            elif re.search(r"\bdia\b", name_lower): chain = "Dia"
            elif re.search(r"\bbm\b", name_lower) or "supermercados bm" in name_lower: chain = "Bm"
            elif "consum" in name_lower: chain = "Consum"
            
            if chain is None:
                continue
                
            # Coordenadas
            if "lat" in element and "lon" in element:
                lat, lon = element["lat"], element["lon"]
            elif "center" in element:
                lat, lon = element["center"]["lat"], element["center"]["lon"]
            else:
                continue
                
            # Dirección limpia (evita NaN)
            street = tags.get("addr:street", "")
            house_number = tags.get("addr:housenumber", "")
            postcode = tags.get("addr:postcode", "")
            city = tags.get("addr:city", "")
            
            address_parts = [p for p in [street, house_number, postcode, city] if p]
            address = ", ".join(str(x).strip() for x in address_parts)
            
            if not address:
                address = f"Establecimiento {name} (Dirección no especificada)"

            stores.append({
                "supermarket": chain,
                "name": name,
                "address": address,
                "lat": float(lat),
                "lon": float(lon)
            })
            
        df_stores = pd.DataFrame(stores)
        if df_stores.empty:
            return pd.DataFrame()
            
        df_stores = df_stores.drop_duplicates(subset=["supermarket", "lat", "lon"]).reset_index(drop=True)
        return df_stores

    except Exception:
        return pd.DataFrame()


# ============================================================
# NORMALIZAR NOMBRES DE CADENAS Y HAVERSINE
# ============================================================

def normalize_chain_name(name):

    name = (
        str(name)
        .lower()
        .strip()
    )

    name = (
        name
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )

    return name


def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):
    R = 6371.0

    dlat = np.radians(
        lat2 - lat1
    )

    dlon = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(
            np.radians(lat1)
        )
        *
        np.cos(
            np.radians(lat2)
        )
        *
        np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return R * c


# ============================================================
# FUNCIONES DE CÁLCULO DE TIENDAS CERCANAS
# ============================================================

def get_nearest_stores(
    user_lat,
    user_lon,
    stores
):
    nearest_stores = []

    for supermarket in stores[
        "supermarket"
    ].unique():

        chain_stores = stores[
            stores["supermarket"] == supermarket
        ].copy()

        if chain_stores.empty:
            continue

        chain_stores["distance_km"] = chain_stores.apply(
            lambda row:
            haversine(
                user_lat,
                user_lon,
                row["lat"],
                row["lon"]
            ),
            axis=1
        )

        nearest = chain_stores.loc[
            chain_stores[
                "distance_km"
            ].idxmin()
        ]

        nearest_stores.append({
            "supermarket":
                supermarket,
            "name":
                nearest.get(
                    "name",
                    supermarket
                ),
            "address":
                nearest.get(
                    "address",
                    ""
                ),
            "lat":
                nearest["lat"],
            "lon":
                nearest["lon"],
            "distance_km":
                nearest["distance_km"]
        })

    if not nearest_stores:
        return pd.DataFrame()

    result = pd.DataFrame(
        nearest_stores
    )

    result = result.sort_values(
        "distance_km"
    ).reset_index(
        drop=True
    )

    return result


def get_nearest_stores_overall(user_lat, user_lon, stores, n=10):
    stores = stores.copy()

    stores["distance_km"] = haversine(
        user_lat,
        user_lon,
        stores["lat"],
        stores["lon"]
    )

    nearest = (
        stores
        .sort_values("distance_km")
        .head(n)
        .copy()
    )

    return nearest


supermarket_colors = {
    "Dia": "red",
    "Mercadona": "green",
    "Aldi": "blue",
    "Alcampo": "orange",
    "Ahorramás": "purple",
    "Bm": "beige",        
    "Consum": "lightgray"    
}


# ============================================================
# NAVEGACIÓN Y BARRA LATERAL
# ============================================================

st.sidebar.title("🛒 Supermarket Optimizer")

page = st.sidebar.radio(
    "Navegación",
    [
        "🏠 Overview",
        "📊 Visualizations",
        "📍 Stores"
    ]
)

st.sidebar.markdown("---")

max_distance = st.sidebar.slider(
    "📏 Distancia máxima",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

st.sidebar.caption(
    f"Mostrando supermercados hasta {max_distance:.1f} km"
)

st.sidebar.header(
    "📍 Tu Dirección"
)

user_address_input = st.sidebar.text_input(
    "Escribe tu calle y localidad:",
    value=(
        "Avenida de la Universidad, "
        "30, Leganés"
    )
)

st.sidebar.markdown("---")

st.sidebar.header(
    "📝 Tu Lista de la Compra"
)

all_items = sorted(
    df[
        "searched_item"
    ]
    .dropna()
    .unique()
)

selected_items = st.sidebar.multiselect(
    "Selecciona los productos:",
    options=all_items,
    default=(
        all_items[:3]
        if len(all_items) >= 3
        else all_items
    )
)


# ============================================================
# GEOCODIFICACIÓN Y CARGA DINÁMICA DE TIENDAS
# ============================================================

location = geocode_address(
    user_address_input
)

if location is None:
    st.error(
        "❌ No hemos podido encontrar esa dirección."
    )
    st.info(
        "Prueba escribiendo la calle, número "
        "y localidad. Por ejemplo: "
        "'Avenida de la Universidad, 30, Leganés'."
    )
    st.stop()

user_lat = location["lat"]
user_lon = location["lon"]

st.sidebar.success(
    f"📍 Ubicación encontrada:\n\n"
    f"{location['display_name']}"
)

# Carga dinámica usando las coordenadas del usuario
with st.spinner("Searching for your nearest supermarkets... 🔎"):
    stores_df = load_stores_dynamic(user_lat, user_lon)

if stores_df.empty:
    st.error(
        "⚠️ No se han encontrado supermercados en las inmediaciones de esta dirección en OpenStreetMap. Prueba con otra ubicación."
    )
    st.stop()


# ============================================================
# BUSCAR TIENDAS MÁS CERCANAS
# ============================================================

nearest_stores = get_nearest_stores(
    user_lat,
    user_lon,
    stores_df
)

nearest_stores_map = get_nearest_stores_overall(
    user_lat,
    user_lon,
    stores_df,
    n=10
)

# Filtrar por la distancia máxima seleccionada
nearest_stores = nearest_stores[
    nearest_stores["distance_km"] <= max_distance
].copy()

nearest_stores_map = nearest_stores_map[
    nearest_stores_map["distance_km"] <= max_distance
].copy()

if nearest_stores.empty:
    st.error(
        "❌ No se han encontrado supermercados dentro de la distancia máxima seleccionada."
    )
    st.stop()


# ============================================================
# COMPROBAR PRODUCTOS SELECCIONADOS
# ============================================================

if not selected_items:
    st.warning(
        "⚠️ Selecciona al menos un producto "
        "en la barra lateral."
    )
    st.stop()


# ============================================================
# CALCULAR PRECIO DE LA CESTA
# ============================================================

shopping_df = df[
    df[
        "searched_item"
    ].isin(
        selected_items
    )
].copy()

basket_costs = (
    shopping_df
    .groupby(
        [
            "supermarket",
            "searched_item"
        ]
    )[
        "price_per_unit"
    ]
    .median()
    .reset_index()
)

pivot_basket = basket_costs.pivot(
    index="searched_item",
    columns="supermarket",
    values="price_per_unit"
)


# ============================================================
# CREAR RESULTADOS
# ============================================================

results_data = []

for _, store in nearest_stores.iterrows():
    supermarket = store["supermarket"]
    distance = store["distance_km"]
    address = store["address"]
    store_name = store["name"]

    if supermarket in pivot_basket.columns:
        prices = pivot_basket[supermarket]
    else:
        prices = pd.Series(dtype=float)

    available_products = (
        prices
        .dropna()
        .shape[0]
    )

    total_products = len(
        selected_items
    )

    coverage = (
        available_products
        /
        total_products
        *
        100
    )

    if available_products > 0:
        basket_cost = prices.sum()
    else:
        basket_cost = np.nan

    if (
        not np.isnan(basket_cost)
        and
        available_products
        == total_products
    ):
        score = (
            basket_cost
            +
            distance * 0.50
        )
    else:
        score = np.nan

    results_data.append({
        "supermarket":
            supermarket,
        "store_name":
            store_name,
        "basket_cost":
            basket_cost,
        "distance_km":
            distance,
        "coverage":
            coverage,
        "address":
            address,
        "score":
            score
    })

res_df = pd.DataFrame(
    results_data
)


# ============================================================
# ORDENAR RESULTADOS
# ============================================================

df_by_distance = (
    res_df
    .sort_values(
        "distance_km"
    )
)

df_by_price = (
    res_df
    .dropna(
        subset=[
            "basket_cost"
        ]
    )
    .sort_values(
        "basket_cost"
    )
)

df_by_smart = (
    res_df
    .dropna(
        subset=[
            "score"
        ]
    )
    .sort_values(
        "score"
    )
)


# ============================================================
# PÁGINA: OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown("---")

    st.subheader(
        "🏆 Recomendación Principal"
    )

    if df_by_smart.empty:

        st.warning(
            "No hay ninguna cadena con datos de precios "
            "para todos los productos seleccionados."
        )

    else:

        best_option = df_by_smart.iloc[0]

        st.success(
            f"🌟 **{best_option['supermarket']}** "
            f"es la opción recomendada.\n\n"
            f"🏪 **Tienda:** "
            f"{best_option['store_name']}\n\n"
            f"📍 **Dirección:** "
            f"{best_option['address']}\n\n"
            f"📏 **Distancia:** "
            f"{best_option['distance_km']:.2f} km\n\n"
            f"💶 **Precio de la cesta:** "
            f"{best_option['basket_cost']:.2f} €\n\n"
            f"📦 **Cobertura de productos:** "
            f"{best_option['coverage']:.0f}%\n\n"
            f"⭐ **Score:** "
            f"{best_option['score']:.2f}"
        )

    st.markdown("---")

    st.subheader(
        "💶 Comparación por precio"
    )

    if df_by_price.empty:

        st.info(
            "No hay datos suficientes para comparar "
            "los precios."
        )

    else:

        display_price = (
            df_by_price[
                [
                    "supermarket",
                    "store_name",
                    "basket_cost",
                    "distance_km",
                    "coverage"
                ]
            ]
            .rename(columns={
                "supermarket":
                    "Supermercado",
                "store_name":
                    "Tienda",
                "basket_cost":
                    "Precio cesta (€)",
                "distance_km":
                    "Distancia (km)",
                "coverage":
                    "Cobertura (%)"
            })
        )

        display_price[
            "Precio cesta (€)"
        ] = display_price[
            "Precio cesta (€)"
        ].round(2)

        display_price[
            "Distancia (km)"
        ] = display_price[
            "Distancia (km)"
        ].round(2)

        display_price[
            "Cobertura (%)"
        ] = display_price[
            "Cobertura (%)"
        ].round(0)

        st.dataframe(
            display_price,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader(
        "📍 Comparación por distancia"
    )

    display_distance = (
        df_by_distance[
            [
                "supermarket",
                "store_name",
                "distance_km",
                "basket_cost",
                "address"
            ]
        ]
        .rename(columns={
            "supermarket":
                "Supermercado",
            "store_name":
                "Tienda",
            "distance_km":
                "Distancia (km)",
            "basket_cost":
                "Precio cesta (€)",
            "address":
                "Dirección"
        })
    )

    display_distance[
        "Distancia (km)"
    ] = display_distance[
        "Distancia (km)"
    ].round(2)

    display_distance[
        "Precio cesta (€)"
    ] = display_distance[
        "Precio cesta (€)"
    ].round(2)

    st.dataframe(
        display_distance,
        use_container_width=True
    )


# ============================================================
# PÁGINA: VISUALIZATIONS
# ============================================================

elif page == "📊 Visualizations":

    st.markdown("---")

    st.subheader(
        "💶 Price Analysis"
    )

    st.markdown(
        "Comparison of the selected shopping basket "
        "across supermarkets."
    )

    if df_by_price.empty:

        st.info(
            "No hay datos suficientes para realizar "
            "la comparación."
        )

    else:

        chart_data = df_by_price[
            [
                "supermarket",
                "basket_cost"
            ]
        ].copy()

        chart_data = chart_data.set_index(
            "supermarket"
        )

        st.bar_chart(
            chart_data[
                "basket_cost"
            ]
        )

    st.markdown("---")

    st.subheader(
        "🔍 Price by Product"
    )

    st.markdown(
        "Unit price comparison for the selected products."
    )

    display_products = (
        pivot_basket
        .copy()
        .round(2)
    )

    st.dataframe(
        display_products,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(
        "🏪 Store Analysis"
    )

    st.markdown(
        "Distribution of stores available for each "
        "supermarket chain."
    )

    store_counts = (
        stores_df[
            "supermarket"
        ]
        .value_counts()
    )

    st.bar_chart(
        store_counts
    )

    st.markdown("---")

    st.subheader(
        "📍 Distance to the nearest store"
    )

    distance_chart = (
        nearest_stores[
            [
                "supermarket",
                "distance_km"
            ]
        ]
        .copy()
    )

    distance_chart = distance_chart.set_index(
        "supermarket"
    )

    st.bar_chart(
        distance_chart[
            "distance_km"
        ]
    )


# ============================================================
# PÁGINA: STORES
# ============================================================

elif page == "📍 Stores":

    st.markdown("---")

    st.subheader(
        "📍 Supermarkets near your location"
    )

    st.markdown(
        "Map showing your location and the 10 nearest supermarkets."
    )

    supermarket_map = folium.Map(
        location=[user_lat, user_lon],
        zoom_start=14
    )

    folium.Marker(
        location=[user_lat, user_lon],
        popup="📍 Your location",
        tooltip="Your location",
        icon=folium.Icon(
            color="black",
            icon="home"
        )
    ).add_to(supermarket_map)

    for _, store in nearest_stores_map.iterrows():

        supermarket_name = store["supermarket"]

        marker_color = supermarket_colors.get(
            supermarket_name,
            "gray"
        )

        popup_text = (
            f"<b>{store['name']}</b><br>"
            f"Supermarket: {store['supermarket']}<br>"
            f"Distance: {store['distance_km']:.2f} km<br>"
            f"Address: {store['address']}"
        )

        folium.Marker(
            location=[
                store["lat"],
                store["lon"]
            ],
            popup=folium.Popup(
                popup_text,
                max_width=300
            ),
            tooltip=(
                f"{store['supermarket']} "
                f"{store['distance_km']:.2f} km"
            ),
            icon=folium.Icon(
                color=marker_color,
                icon="shopping-cart",
                prefix="fa"
            )
        ).add_to(supermarket_map)

    st.markdown(
        """
        **Leyenda**

        🔴 Dia &nbsp;&nbsp;
        🟢 Mercadona &nbsp;&nbsp;
        🔵 Aldi &nbsp;&nbsp;
        🟠 Alcampo &nbsp;&nbsp;
        🟣 Ahorramás &nbsp;&nbsp;
        🟡 BM &nbsp;&nbsp;
        ⚪ Consum
        """
    )

    st_folium(
        supermarket_map,
        width=None,
        height=550
    )

    st.markdown("---")

    st.subheader(
        "🏪 Nearest supermarkets"
    )

    display_nearest = (
        nearest_stores_map[
            [
                "supermarket",
                "name",
                "distance_km",
                "address"
            ]
        ]
        .rename(columns={
            "supermarket": "Supermarket",
            "name": "Store",
            "distance_km": "Distance (km)",
            "address": "Address"
        })
    )

    display_nearest["Distance (km)"] = (
        display_nearest["Distance (km)"].round(2)
    )

    st.dataframe(
        display_nearest,
        use_container_width=True
    )


# ============================================================
# INFORMACIÓN TÉCNICA Y PIE DE PÁGINA
# ============================================================

with st.expander(
    "ℹ️ ¿Cómo funciona este cálculo?"
):

    st.markdown(
        """
        **1. Dirección**
        La dirección introducida se convierte en coordenadas mediante Nominatim.

        **2. Localización de supermercados**
        La aplicación busca dinámicamente en tiempo real mediante OpenStreetMap (Overpass API) los establecimientos de las cadenas analizadas alrededor de la ubicación introducida, permitiendo cobertura en toda España.

        **3. Selección de establecimientos**
        Para cada cadena se calcula la distancia entre la dirección del usuario y todas sus tiendas disponibles en el radio consultado, seleccionando automáticamente la más cercana.

        **4. Precios**
        Los precios proceden de los datos obtenidos mediante web scraping y APIs.

        **5. Distancia**
        La distancia geográfica se calcula mediante la fórmula de Haversine.

        **6. Recomendación**
        Cuando una cadena dispone de todos los productos seleccionados, se calcula:
        `score = precio_cesta + 0.50 × distancia_km`

        **7. Cobertura**
        Porcentaje de los productos seleccionados con datos disponibles para cada supermercado.
        """
    )

st.markdown("---")

st.caption(
    "Datos de localización: OpenStreetMap contributors. "
    "Precios: datos obtenidos mediante web scraping y APIs."
)