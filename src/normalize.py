import pandas as pd


def normalize_units(df):
    df = df.copy()

    unit_mapping = {
        "L": "LITRO",
        "LTR": "LITRO",
        "LITRO": "LITRO",
        "LITROS": "LITRO",

        "KG": "KILO",
        "KGS": "KILO",
        "KILO": "KILO",
        "KILOS": "KILO",

        "UD": "UNIDAD",
        "UDS": "UNIDAD",
        "UNIDAD": "UNIDAD",
        "UNIDADES": "UNIDAD",

        "DOCENA": "DOCENA",
        "DOCENAS": "DOCENA",

        "LAVADO": "LAVADO",
        "METRO": "METRO",
        "CAPSULA": "CAPSULA",
        "BOLSITA": "BOLSITA",
        "PASTILLA": "PASTILLA",
        "MONODOSIS": "MONODOSIS",
        "PANUELO": "PANUELO",
        "TOALLITA": "TOALLITA",

        "ROLLO": "ROLLO",
        "SERVICIO": "SERVICIO",
        "SOBRE": "SOBRE",
    }

    df["unit"] = (
        df["unit"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(unit_mapping)
    )

    return df