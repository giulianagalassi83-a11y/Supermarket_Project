# 🛒 Supermarket Price Analytics — Comunidad de Madrid

Proyecto analítico de web scraping, normalización y comparación de precios de productos de gran consumo en los principales supermercados de la Comunidad de Madrid (**AhorraMás, Alcampo, Aldi, DIA y Mercadona**).

---

## 📂 Estructura del Proyecto

```text
supermarket_project/
│
├── data/
│   ├── raw/                  # Datos crudos extraídos directamente de los web scrapers
│   │   ├── dia_products_raw.csv
│   │   ├── mercadona_products_raw.csv
│   │   └── ...
│   ├── processed/            # Datos limpios, deduplicados y normalizados
│   │   ├── dia_products.csv[cite: 16]
│   │   ├── mercadona_products.csv[cite: 17]
│   │   ├── aldi_products.csv
│   │   ├── alcampo_products.csv
│   │   └── ahorramas_products.csv
│   └── stores_madrid.csv     # Localizaciones de establecimientos físicas (OSM)
│
├── src/
│   ├── scrapers/             # Scripts automatizados de extracción (Selenium / BeautifulSoup)
│   ├── normalize.py          # Módulo central de estandarización de unidades
│   └── get_stores.py         # Extractor de ubicaciones vía Overpass API (OpenStreetMap)
│
├── api_clients/
│   ├── bm_consum_api.ipynb   # Extracción vía API REST JSON (BM y Consum), sin navegador
│   └── mercadona_api.ipynb   # Alternativa vía API al scraper Selenium de Mercadona (catálogo completo)
│
└── README.md
```

## ⚙️ Flujo de Trabajo (Pipeline)
El proyecto sigue una metodología rigurosa dividida en fases de ingeniería de datos:

Web Scraping Automatizado:

Se realizan búsquedas parametrizadas sobre un catálogo común de 20 productos esenciales (leche, huevos, aceite, arroz, pasta, detergente, etc.).

Se maneja la carga dinámica de elementos mediante scroll infinito, control de tiempos de espera y gestión de reintentos automáticos.

Capa RAW:

Las apariciones extraídas de cada cadena se consolidan en archivos CSV sin procesar ubicados en data/raw/.

Procesamiento y Deduplicación:

Se eliminan registros duplicados basándose en identificadores de producto o combinaciones de nombre y formato.

Se filtran filas incompletas que carezcan de precios o unidades válidas.

Normalización de Unidades (normalize.py):

Se estandarizan las magnitudes a un formato común (KILO, LITRO, UNIDAD, DOCENA, etc.) para permitir comparaciones homogéneas entre diferentes marcas y supermercados.

Capa PROCESSED:

Se almacenan los datasets limpios en data/processed/ listos para el análisis conjunto.

## 🛠️ Componentes Principales
1. Scrapers por Supermercado
DIA: Automatización basada en Selenium con scroll interno para la carga completa de tarjetas de producto y extracción de precios por unidad/kilo.

Mercadona: Simulación de navegación introduciendo el código postal de referencia (28911 - Alcorcón / Madrid) para asegurar la disponibilidad local del stock y precios del almacén correspondiente.

Aldi, Alcampo y AhorraMás: Scrapers adaptados a sus respectivas estructuras web para unificar el formato tabular de salida.

BM y Consum (api_clients/bm_consum_api.ipynb): En lugar de scraping, se consulta la API REST JSON que usan sus propias webs (ambas sobre la plataforma Aktios, por lo que un único cliente sirve para las dos). Devuelve ID, EAN, marca, categoría, precio normal, precio en oferta y precio por unidad ya estructurados, con paginación (`limit`/`offset`, máximo 500 resultados por búsqueda).

Mercadona vía API (api_clients/mercadona_api.ipynb): alternativa opcional al scraper Selenium. Su API no tiene buscador, así que descarga el catálogo completo por categorías (~150 peticiones) y asigna cada producto a uno de los 20 términos por coincidencia de palabras en el nombre. Genera `mercadona_api_products_raw.csv` y `mercadona_api_products.csv` (mismas columnas que `mercadona_products.csv` más `product_id`, `url`, `category` y `previous_price`).

Nota ética: el `robots.txt` de BM, Consum y Mercadona incluye `Disallow: /api/`. Lo declaramos como limitación: uso académico y no comercial, ~120-150 peticiones por ejecución, 1 s de pausa entre peticiones y un `User-Agent` que identifica el proyecto.

2. Módulo de Normalización (src/normalize.py)
Módulo encargado de mapear variantes de unidades de medida heterogéneas (como L, LTR, litros o KGS, kilos) hacia una nomenclatura unificada en mayúsculas (LITRO, KILO, UNIDAD, etc.).

3. Localizador Geográfico (src/get_stores.py)
Utiliza la API de Overpass (OpenStreetMap) para extraer las coordenadas geográficas (lat, lon) y direcciones postales de los establecimientos físicos de las 5 cadenas en la Comunidad de Madrid, filtrando exclusivamente los nodos etiquetados como supermercados comerciales.

## 📊 Análisis y Comparativa Conjunta
Una vez unificados los datasets, el sistema permite realizar consultas analíticas avanzadas como:

Comparativa de precios por producto y unidad entre competidores.

Identificación de los productos más económicos por categoría o de forma global.

Análisis de cestas de la compra y dispersión de precios en Madrid.

🚀 Requisitos e Instalación
Asegúrate de tener instalado Python 3.10 o superior y las librerías necesarias: (HAY QIE AÑADIRLAS)