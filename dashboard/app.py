import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import requests
import re


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

STORES_FILE = (
    BASE_DIR
    / "data"
    / "stores_madrid.csv"
)


# Comprobamos que existen las carpetas/archivos necesarios

if not PROCESSED_DIR.exists():

    st.error(
        f"No se encuentra la carpeta de datos en: "
        f"{PROCESSED_DIR}"
    )

    st.stop()


if not STORES_FILE.exists():

    st.error(
        f"No se encuentra el archivo de supermercados: "
        f"{STORES_FILE}"
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
            PROCESSED_DIR / "ahorramas_products.csv"

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
                            "Ahorramás"

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
# CARGAR SUPERMERCADOS DE OPENSTREETMAP
# ============================================================

@st.cache_data
def load_stores():

    stores = pd.read_csv(
        STORES_FILE
    )


    # --------------------------------------------------------
    # NORMALIZAR NOMBRES
    # --------------------------------------------------------

    if "supermarket" in stores.columns:

        stores["supermarket"] = (
            stores["supermarket"]
            .astype(str)
            .str.strip()
        )


    # --------------------------------------------------------
    # CONVERTIR COORDENADAS A NUMÉRICAS
    # --------------------------------------------------------

    stores["lat"] = pd.to_numeric(
        stores["lat"],
        errors="coerce"
    )

    stores["lon"] = pd.to_numeric(
        stores["lon"],
        errors="coerce"
    )


    # Eliminamos establecimientos
    # sin coordenadas válidas.

    stores = stores.dropna(
        subset=[
            "lat",
            "lon"
        ]
    ).copy()


    return stores


stores_df = load_stores()


if stores_df.empty:

    st.error(
        "El archivo stores_madrid.csv está vacío "
        "o no contiene coordenadas válidas."
    )

    st.stop()


# ============================================================
# NORMALIZAR NOMBRES DE CADENAS
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


# ============================================================
# HAVERSINE
# ============================================================

def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):

    """
    Calcula la distancia aproximada en kilómetros
    entre dos puntos geográficos.
    """

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
# GEOCODIFICAR DIRECCIÓN CON NOMINATIM
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
# BUSCAR LA TIENDA MÁS CERCANA DE CADA CADENA
# ============================================================

def get_nearest_stores(
    user_lat,
    user_lon,
    stores
):

    nearest_stores = []


    # --------------------------------------------------------
    # RECORREMOS CADA CADENA
    # --------------------------------------------------------

    for supermarket in stores[
        "supermarket"
    ].unique():

        chain_stores = stores[
            stores["supermarket"] == supermarket
        ].copy()


        if chain_stores.empty:

            continue


        # ----------------------------------------------------
        # CALCULAR DISTANCIA A TODAS LAS TIENDAS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # ELEGIR LA MÁS CERCANA
        # ----------------------------------------------------

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


# ============================================================
# SIDEBAR
# ============================================================

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
# GEOCODIFICAR DIRECCIÓN
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


# ============================================================
# BUSCAR TIENDAS MÁS CERCANAS
# ============================================================

nearest_stores = get_nearest_stores(

    user_lat,

    user_lon,

    stores_df

)


if nearest_stores.empty:

    st.error(
        "❌ No se han encontrado supermercados."
    )

    st.stop()


# ============================================================
# MOSTRAR TIENDAS MÁS CERCANAS
# ============================================================

st.subheader(
    "📍 Supermercados más cercanos"
)


display_nearest = (

    nearest_stores[

        [
            "supermarket",
            "name",
            "distance_km",
            "address"

        ]

    ]

    .rename(columns={

        "supermarket":
            "Cadena",

        "name":
            "Tienda",

        "distance_km":
            "Distancia (km)",

        "address":
            "Dirección"

    })

)


display_nearest[
    "Distancia (km)"
] = display_nearest[
    "Distancia (km)"
].round(2)


st.dataframe(

    display_nearest,

    use_container_width=True

)


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


# Para cada supermercado y producto
# utilizamos la mediana del precio unitario.

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


# ============================================================
# TABLA PRODUCTO × SUPERMERCADO
# ============================================================

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

    supermarket = store[
        "supermarket"
    ]


    distance = store[
        "distance_km"
    ]


    address = store[
        "address"
    ]


    store_name = store[
        "name"
    ]


    # --------------------------------------------------------
    # PRECIOS DE LA CADENA
    # --------------------------------------------------------

    if supermarket in pivot_basket.columns:

        prices = pivot_basket[
            supermarket
        ]

    else:

        prices = pd.Series(
            dtype=float
        )


    # --------------------------------------------------------
    # PRODUCTOS DISPONIBLES
    # --------------------------------------------------------

    available_products = (
        prices
        .dropna()
        .shape[0]
    )


    total_products = len(
        selected_items
    )


    # --------------------------------------------------------
    # COBERTURA
    # --------------------------------------------------------

    coverage = (

        available_products

        /

        total_products

        *

        100

    )


    # --------------------------------------------------------
    # COSTE DE LA CESTA
    # --------------------------------------------------------

    if available_products > 0:

        basket_cost = prices.sum()

    else:

        basket_cost = np.nan


    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

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
# RECOMENDACIÓN PRINCIPAL
# ============================================================

st.markdown("---")


st.subheader(
    "🏆 Recomendación Principal"
)


if df_by_smart.empty:

    st.warning(

        "No hay ninguna cadena con datos de precios "
        "para todos los productos seleccionados."

    )

    st.info(

        "Puedes consultar igualmente las tablas "
        "de precios y cobertura."

    )

else:

    best_option = (
        df_by_smart.iloc[0]
    )


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


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(

    [

        "📊 Comparativa por Precio",

        "📍 Comparativa por Cercanía",

        "🔍 Desglose por Productos"

    ]

)


# ============================================================
# TAB 1 — PRECIO
# ============================================================

with tab1:

    st.markdown(
        "### 💶 Supermercados ordenados por precio"
    )


    if df_by_price.empty:

        st.info(
            "No hay datos suficientes."
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
                    "Coste Cesta (€)",

                "distance_km":
                    "Distancia (km)",

                "coverage":
                    "Cobertura (%)"

            })

        )


        display_price[
            "Coste Cesta (€)"
        ] = display_price[
            "Coste Cesta (€)"
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


# ============================================================
# TAB 2 — DISTANCIA
# ============================================================

with tab2:

    st.markdown(
        "### 📍 Supermercados ordenados por cercanía"
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
                "Coste Cesta (€)",

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
        "Coste Cesta (€)"
    ] = display_distance[
        "Coste Cesta (€)"
    ].round(2)


    st.dataframe(

        display_distance,

        use_container_width=True

    )


# ============================================================
# TAB 3 — PRODUCTOS
# ============================================================

with tab3:

    st.markdown(
        "### 🔍 Precios unitarios por producto"
    )


    display_products = pivot_basket.copy()


    display_products = (
        display_products.round(2)
    )


    st.dataframe(

        display_products,

        use_container_width=True

    )


# ============================================================
# INFORMACIÓN TÉCNICA
# ============================================================

with st.expander(
    "ℹ️ ¿Cómo funciona este cálculo?"
):

    st.markdown(

        """
        **1. Dirección**

        La dirección introducida se convierte en
        coordenadas mediante Nominatim.

        **2. Localización de supermercados**

        La aplicación utiliza un archivo previamente
        obtenido mediante OpenStreetMap que contiene
        establecimientos de las cadenas analizadas
        en la Comunidad de Madrid.

        **3. Selección de establecimientos**

        Para cada cadena se calcula la distancia
        entre la dirección del usuario y todas sus
        tiendas disponibles.

        Se selecciona automáticamente la tienda
        físicamente más cercana.

        **4. Precios**

        Los precios proceden de los datos obtenidos
        mediante web scraping.

        **5. Distancia**

        La distancia geográfica se calcula mediante
        la fórmula de Haversine.

        **6. Recomendación**

        Cuando una cadena dispone de todos los
        productos seleccionados, se calcula:

        `score = precio_cesta + 0.50 × distancia_km`

        Cuanto menor sea el score, mejor es la
        combinación entre precio y proximidad.

        **7. Cobertura**

        También mostramos qué porcentaje de los
        productos seleccionados tiene datos disponibles
        para cada supermercado.
        """

    )


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.markdown("---")


st.caption(

    "Datos de localización: OpenStreetMap contributors. "
    "Precios: datos obtenidos mediante web scraping."

)