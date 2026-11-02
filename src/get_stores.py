# Downloads the supermarkets of the Comunidad de Madrid from OpenStreetMap.
# TO GET THE STORES, RUN IN THE TERMINAL: python src/get_stores.py
import requests
import pandas as pd
from pathlib import Path
import re
import time


# ============================================================
# SETTINGS
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
# GET SUPERMARKETS FROM OPENSTREETMAP
# ============================================================

def get_madrid_supermarkets():

    print("🔎 Searching for supermarkets in Madrid...")

    # --------------------------------------------------------
    # IMPORTANT:
    # We only search for elements tagged shop=supermarket.
    #
    # This avoids finding:
    # - car parks
    # - petrol stations
    # - distribution centres
    # - day centres
    # - streets
    # etc.
    # --------------------------------------------------------

    query = """
    [out:json][timeout:120];

    (
        nwr["shop"="supermarket"]["brand"~"Mercadona|DIA|ALDI|Alcampo|Ahorramás|Ahorramas|BM|Consum",i](40.20,-4.00,40.60,-3.40);

        nwr["shop"="supermarket"]["name"~"Mercadona|DIA|ALDI|Alcampo|Ahorramás|Ahorramas|BM|Consum",i](40.20,-4.00,40.60,-3.40);
    );

    out center tags;
    """

    # --------------------------------------------------------
    # Try the Overpass servers
    # --------------------------------------------------------

    for server in OVERPASS_SERVERS:

        print("\n🌐 Trying server:")
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
                f"✅ Server OK. "
                f"Elements found: {len(elements)}"
            )

            return elements

        except Exception as e:

            print("❌ This server failed:")
            print(e)

            time.sleep(2)

    return None


# ============================================================
# IDENTIFY THE CHAIN
# ============================================================

def identify_chain(name):

    if not name:
        return None

    name = str(name).lower()

    # Normalize accents
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


    # --------------------------------------------------------
    # BM Supermercados
    # --------------------------------------------------------
    if re.search(r"\bbm\b", name) or "supermercados bm" in name:
        return "Bm"

    # --------------------------------------------------------
    # Consum
    # --------------------------------------------------------
    if "consum" in name:
        return "Consum"

    return None


# ============================================================
# PROCESS ELEMENTS
# ============================================================

def process_elements(elements):

    stores = []

    for element in elements:

        tags = element.get("tags", {})

        # ----------------------------------------------------
        # SUPERMARKETS ONLY
        # ----------------------------------------------------

        shop_type = tags.get("shop", "")

        if shop_type != "supermarket":
            continue

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        name = (
            tags.get("name")
            or tags.get("brand")
            or tags.get("operator")
        )

        if not name:
            continue

        # ----------------------------------------------------
        # IDENTIFY THE CHAIN
        # ----------------------------------------------------

        chain = identify_chain(name)

        # If the name doesn't identify it,
        # try the brand.

        if chain is None:

            brand = tags.get("brand", "")

            chain = identify_chain(brand)

        if chain is None:
            continue

        # ----------------------------------------------------
        # COORDINATES
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
        # ADDRESS
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
        # SAVE THE SUPERMARKET
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
# CLEAN THE DATA
# ============================================================

def clean_stores(stores):

    df = pd.DataFrame(stores)

    if df.empty:
        return df

    # --------------------------------------------------------
    # Convert coordinates to numbers
    # --------------------------------------------------------

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce"
    )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce"
    )

    # Drop rows without coordinates
    df = df.dropna(
        subset=["lat", "lon"]
    )

    # --------------------------------------------------------
    # Drop duplicates
    #
    # Two OSM elements can represent the same store.
    # With the same chain and coordinates, we keep one.
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "supermarket",
            "lat",
            "lon"
        ]
    )

    # --------------------------------------------------------
    # Sort
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
        "🛒 SUPERMARKETS OF THE "
        "COMUNIDAD DE MADRID"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Get the data from OpenStreetMap
    # --------------------------------------------------------

    elements = get_madrid_supermarkets()

    if elements is None:

        print("\n❌ Could not connect to Overpass.")
        print("The file was not created.")

        return

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    stores = process_elements(elements)

    print(
        f"\n📦 Valid supermarkets found: "
        f"{len(stores)}"
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = clean_stores(stores)

    if df.empty:

        print(
            "\n❌ No valid "
            "supermarkets found."
        )

        return

    # --------------------------------------------------------
    # Create the folder if it doesn't exist
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save the CSV
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("✅ FILE CREATED")
    print("=" * 60)

    print(OUTPUT_FILE)

    print("\n📊 SUMMARY BY CHAIN:")

    print(
        df["supermarket"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\n🏪 Total stores: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Available addresses
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
        f"\n📍 With address: "
        f"{addresses_available}"
    )

    print(
        f"❓ Without address: "
        f"{addresses_missing}"
    )

    # --------------------------------------------------------
    # First results
    # --------------------------------------------------------

    print("\n🔎 FIRST RESULTS:")

    print(
        df.head(15).to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("🎉 FINISHED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()