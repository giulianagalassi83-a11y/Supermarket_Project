import requests
import pandas as pd
from pathlib import Path
import re
import time


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_FILE = BASE_DIR / "data" / "stores_madrid.csv"


OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]


# ============================================================
# OBTENER SUPERMERCADOS DE OPENSTREETMAP
# ============================================================

def get_madrid_supermarkets():

    print("🔎 Buscando supermercados en Madrid...")

    # --------------------------------------------------------
    # IMPORTANTE:
    # Solo buscamos elementos etiquetados como
    # shop=supermarket.
    #
    # Esto evita encontrar:
    # - parkings
    # - gasolineras
    # - centros de distribución
    # - centros de día
    # - calles
    # etc.
    # --------------------------------------------------------

    query = """
    [out:json][timeout:120];

    (
        nwr["shop"="supermarket"]["brand"~"Mercadona|DIA|ALDI|Alcampo|Ahorramás|Ahorramas",i](40.20,-4.00,40.60,-3.40);

        nwr["shop"="supermarket"]["name"~"Mercadona|DIA|ALDI|Alcampo|Ahorramás|Ahorramas",i](40.20,-4.00,40.60,-3.40);
    );

    out center tags;
    """

    # --------------------------------------------------------
    # Probar los servidores de Overpass
    # --------------------------------------------------------

    for server in OVERPASS_SERVERS:

        print("\n🌐 Probando servidor:")
        print(server)

        try:

            response = requests.post(
                server,
                data=query.encode("utf-8"),
                headers={
                    "User-Agent": "UC3M-Supermarket-Project/1.0"
                },
                timeout=150
            )

            response.raise_for_status()

            data = response.json()

            elements = data.get("elements", [])

            print(
                f"✅ Servidor correcto. "
                f"Elementos encontrados: {len(elements)}"
            )

            return elements

        except Exception as e:

            print("❌ Este servidor ha fallado:")
            print(e)

            time.sleep(2)

    return None


# ============================================================
# IDENTIFICAR CADENA
# ============================================================

def identify_chain(name):

    if not name:
        return None

    name = str(name).lower()

    # Normalizamos acentos
    name = (
        name
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )

    # --------------------------------------------------------
    # Mercadona
    # --------------------------------------------------------

    if "mercadona" in name:
        return "Mercadona"

    # --------------------------------------------------------
    # Ahorramás
    # --------------------------------------------------------

    if "ahorramas" in name:
        return "Ahorramás"

    # --------------------------------------------------------
    # Alcampo
    # --------------------------------------------------------

    if "alcampo" in name:
        return "Alcampo"

    # --------------------------------------------------------
    # Aldi
    # --------------------------------------------------------

    if re.search(r"\baldi\b", name):
        return "Aldi"

    # --------------------------------------------------------
    # DIA
    # --------------------------------------------------------

    if re.search(r"\bdia\b", name):
        return "Dia"

    return None


# ============================================================
# PROCESAR ELEMENTOS
# ============================================================

def process_elements(elements):

    stores = []

    for element in elements:

        tags = element.get("tags", {})

        # ----------------------------------------------------
        # SOLO SUPERMERCADOS
        # ----------------------------------------------------

        shop_type = tags.get("shop", "")

        if shop_type != "supermarket":
            continue

        # ----------------------------------------------------
        # NOMBRE
        # ----------------------------------------------------

        name = (
            tags.get("name")
            or tags.get("brand")
            or tags.get("operator")
        )

        if not name:
            continue

        # ----------------------------------------------------
        # IDENTIFICAR CADENA
        # ----------------------------------------------------

        chain = identify_chain(name)

        # Si el nombre no permite identificarla,
        # probamos con brand.

        if chain is None:

            brand = tags.get("brand", "")

            chain = identify_chain(brand)

        if chain is None:
            continue

        # ----------------------------------------------------
        # COORDENADAS
        # ----------------------------------------------------

        if "lat" in element and "lon" in element:

            lat = element["lat"]
            lon = element["lon"]

        elif "center" in element:

            lat = element["center"]["lat"]
            lon = element["center"]["lon"]

        else:

            continue

        # ----------------------------------------------------
        # DIRECCIÓN
        # ----------------------------------------------------

        street = tags.get("addr:street", "")
        house_number = tags.get("addr:housenumber", "")
        postcode = tags.get("addr:postcode", "")
        city = tags.get("addr:city", "")

        address_parts = [
            street,
            house_number,
            postcode,
            city
        ]

        address = ", ".join(
            str(x).strip()
            for x in address_parts
            if x
        )

        # ----------------------------------------------------
        # GUARDAR SUPERMERCADO
        # ----------------------------------------------------

        stores.append({
            "supermarket": chain,
            "name": name,
            "address": address,
            "lat": lat,
            "lon": lon
        })

    return stores


# ============================================================
# LIMPIAR DATOS
# ============================================================

def clean_stores(stores):

    df = pd.DataFrame(stores)

    if df.empty:
        return df

    # --------------------------------------------------------
    # Convertir coordenadas a numéricas
    # --------------------------------------------------------

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce"
    )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce"
    )

    # Eliminar filas sin coordenadas
    df = df.dropna(
        subset=["lat", "lon"]
    )

    # --------------------------------------------------------
    # Eliminar duplicados
    #
    # Dos elementos de OSM pueden representar la misma tienda.
    # Si tienen misma cadena + mismas coordenadas,
    # nos quedamos con uno.
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "supermarket",
            "lat",
            "lon"
        ]
    )

    # --------------------------------------------------------
    # Ordenar
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "supermarket",
            "name"
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "🛒 OBTENCIÓN DE SUPERMERCADOS "
        "DE LA COMUNIDAD DE MADRID"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Obtener datos de OpenStreetMap
    # --------------------------------------------------------

    elements = get_madrid_supermarkets()

    if elements is None:

        print("\n❌ No se pudo conectar con Overpass.")
        print("No se ha creado el archivo.")

        return

    # --------------------------------------------------------
    # Procesar
    # --------------------------------------------------------

    stores = process_elements(elements)

    print(
        f"\n📦 Supermercados válidos encontrados: "
        f"{len(stores)}"
    )

    # --------------------------------------------------------
    # Limpiar
    # --------------------------------------------------------

    df = clean_stores(stores)

    if df.empty:

        print(
            "\n❌ No se encontraron "
            "supermercados válidos."
        )

        return

    # --------------------------------------------------------
    # Crear carpeta si no existe
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Guardar CSV
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("✅ ARCHIVO CREADO")
    print("=" * 60)

    print(OUTPUT_FILE)

    print("\n📊 RESUMEN POR CADENA:")

    print(
        df["supermarket"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\n🏪 Total de establecimientos: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Direcciones disponibles
    # --------------------------------------------------------

    addresses_available = (
        df["address"]
        .fillna("")
        .str.strip()
        .ne("")
        .sum()
    )

    addresses_missing = (
        len(df) - addresses_available
    )

    print(
        f"\n📍 Con dirección: "
        f"{addresses_available}"
    )

    print(
        f"❓ Sin dirección: "
        f"{addresses_missing}"
    )

    # --------------------------------------------------------
    # Primeros resultados
    # --------------------------------------------------------

    print("\n🔎 PRIMEROS RESULTADOS:")

    print(
        df.head(15).to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("🎉 PROCESO TERMINADO")
    print("=" * 60)


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":
    main()