# 🛒 Supermarket Price Analytics: Comunidad de Madrid

Web scraping, normalization and price comparison of everyday products in the main supermarket chains of the Comunidad de Madrid: **Ahorramás, Alcampo, Aldi, BM, Consum, Dia and Mercadona**.

---

## 📂 Project structure

```text
supermarket_project/
│
├── data/
│   ├── raw/                     # Data as returned by the scrapers and APIs
│   ├── processed/               # Clean per-chain files
│   ├── history/                 # Older snapshots, recovered from git
│   ├── taxonomy/                # Product types
│   ├── labels/                  # Hand labels for the classifier
│   ├── products_normalized.csv  # Output of the pipeline
│   ├── price_changes.csv        # Price changes between collection dates
│   └── stores_madrid.csv        # Store locations (OpenStreetMap)
│
├── src/
│   ├── pipeline/                # Normalization and product-type classification
│   ├── analysis/                # Prices, optimizer, price history and insights
│   ├── normalize.py             # Unit standardization used by the collectors
│   └── get_stores.py            # Store locations from the Overpass API
│
├── scrapers/                    # Selenium notebooks, one per chain
├── api_clients/                 # REST API notebooks (BM, Consum and Mercadona)
│
├── dashboard/
│   ├── app.py                   # Main dashboard
│   ├── charts.py                # Altair charts
│   └── app_legacy.py            # Previous dashboard, deprecated
│
└── README.md
```

## ⚙️ Pipeline

**Scraping and API collection:** parameterized searches over 20 essential products (milk, eggs, oil, rice, pasta, detergent and others). The scrapers handle infinite scroll, waiting times and automatic retries. Carrefour is blocked by Cloudflare and is not included.

**Raw layer:** rows from each chain are stored as CSV files in `data/raw/`.

**Normalization (`src/pipeline/`):** run with `python -m src.pipeline.build`. It removes duplicates, standardizes units and classifies each product into one of 70 product types by text similarity, so chains can be compared on equivalent products.

**Output:** `data/supermarkets.db` and `data/products_normalized.csv`, which the dashboard reads.

**Price changes:** run with `python -m src.analysis.history`. It compares the two dates held for Aldi, Dia and Mercadona and writes `data/price_changes.csv`.

## 🛠️ Main components

### 1. Collectors

- **Dia:** Selenium with internal scrolling to load every product card.
- **Mercadona:** Selenium, entering a reference postcode (28911) so stock and prices match the local warehouse.
- **Aldi, Alcampo and Ahorramás:** scrapers adapted to each website, with a shared output format.
- **BM and Consum (`api_clients/bm_consum_api.ipynb`):** the REST JSON API behind both websites, which share the Aktios platform. It returns ID, barcode, brand, category, regular price, offer price and unit price, with up to 500 results per search.
- **Mercadona API (`api_clients/mercadona_api.ipynb`):** downloads the whole catalogue by category (about 150 requests) and assigns each product to a search term by the words in its name.

Ethics note: the `robots.txt` of BM, Consum and Mercadona includes `Disallow: /api/`. We declare this as a limitation: academic, non-commercial use, about 120 to 150 requests per run, a 1 s pause between requests and a `User-Agent` that identifies the project.

### 2. Store locator (`src/get_stores.py`)

Uses the Overpass API (OpenStreetMap) to get the coordinates and addresses of stores in the Comunidad de Madrid.

### 3. Dashboard (`dashboard/app.py`)

Streamlit interface with five pages. Product names stay in Spanish, as the chains sell them.

To run it: `streamlit run dashboard/app.py`

- **Basket:** search the product types and set quantities.
- **Compare:** price per chain, a product × chain matrix, the best route and two charts.
- **Map:** what the basket costs at each nearby store, with the best route.
- **Trends:** price changes between the dates we hold, biggest moves, change by category and current offers. A trend line needs more dates.
- **Analysis:** chain index, cheapest chain per category, own brand against national brands, product against store choice, same-barcode prices, route savings and data coverage.

**Optimizer (`src/analysis/optimizer.py`):** solves a small Travelling Purchaser Problem. It tries up to 3 stores (at most one per chain, from the nearest of each chain) and every visiting order, adding product cost, travel (Haversine distance with a 1.3 detour factor and a configurable cost per km) and a cost per extra stop.

**Fault tolerance:** if Overpass or Nominatim do not respond, the app uses `data/stores_madrid.csv` and a saved address.

The previous dashboard is kept, deprecated, in `dashboard/app_legacy.py`.

## 🚀 Requirements

Python 3.10 or later. Install the dependencies with:

```bash
pip install -r requirements.txt
```
