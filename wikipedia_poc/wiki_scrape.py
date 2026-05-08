  
"""
TFG - Wikipedia Snapshot Builder
---------------------------------
Construye una tabla denormalizada de snapshots de páginas
a partir de revisiones históricas.

Fuente: Wikipedia API
"""

import requests
import json
import os
import time
import hashlib
from datetime import datetime


import logging
from tfg.logging_config import setup_logging, timed

setup_logging(level="DEBUG")
logger = logging.getLogger("wiki_poc.wiki_scrape_data")

API = "https://es.wikipedia.org/w/api.php"
PAGES_SIZE=1000
PREFIX_FILE="data/raw/wiki"
PAGES_FILE = f"{PREFIX_FILE}_pages_{PAGES_SIZE}.json"

# =========================
# CONFIGURACIÓN
# =========================

API_URL = "https://en.wikipedia.org/w/api.php"

SNAPSHOT_DATE_1 = "2024-01-01T00:00:00Z"
SNAPSHOT_DATE_2 = "2024-02-01T00:00:00Z"

PAGE_LIMIT = 1000
REQUEST_SLEEP = 0.2

PAGES_FILE = f"{PREFIX_FILE}_pages_{PAGE_LIMIT}.json"
OUTPUT_FILE = f"{PREFIX_FILE}_page_snapshot_wide.json"

headers = { "User-Agent": "Mozilla/5.0 (Xavi TFG scraper)"}   

def fetch_random_pages(limit=1000):
    """
    Obtiene páginas aleatorias desde Wikipedia.
    """
    pages = []

    while len(pages) < limit:
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnlimit": 500,
            "rnnamespace": 0
        }

        try:
            response = requests.get(API_URL, params=params, headers=headers)
            response.raise_for_status()
            if response.status_code != 200:
                raise RuntimeError(f"HTTP error {response.status_code}: {response.text[:200]}")
            data = response.json()
        
        except Exception as e:
            logger.exception("Request failed")

        logger.debug("STATUS:", response.status_code)
        logger.debug("CONTENT-TYPE:", response.headers.get("Content-Type"))
        logger.debug("TEXT:", response.text[:500])
        
        data = response.json()
        pages.extend(data["query"]["random"])

    return [
        {"pageid": p["id"], "title": p["title"]}
        for p in pages[:limit]
    ]


def load_or_create_pages():
    """
    Queremos reproducibilidad del experimento.
    """
    if os.path.exists(PAGES_FILE):
        logger.info("Cargando páginas existentes desde {PAGES_FILE}")
        with open(PAGES_FILE, "r") as f:
            return json.load(f)
    logger.info("No se ha encontrando fichero de páginas {PAGES_FILE}")
    with timed (logger, "Generando nuevas páginas..."):
        pages = fetch_random_pages(PAGE_LIMIT)

    logger.debug("Guardando fichero de págincas en {PAGES_FILE}")
    with open(PAGES_FILE, "w") as f:
        json.dump(pages, f, indent=2)

    return pages


def get_page_snapshot(pageid, snapshot_date):
    """
    Obtiene el estado de una página en una fecha dada.
    """

    params = {
        "action": "query",
        "format": "json",
        "pageids": pageid,
        "prop": "info|revisions|categories|links",

        "inprop": "url",

        "rvlimit": 1,
        "rvstart": snapshot_date,
        "rvprop": "ids|timestamp|user|comment|size|content",
        "rvslots": "main",

        "cllimit": 50,
        "pllimit": 50
    }

    try:
        r = requests.get(API_URL, params=params, headers=headers)
        
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", 5))
            logger.warning(f"Rate limit hit. Sleeping {retry_after}s...")
            time.sleep(retry_after)
            return get_page_snapshot(pageid, snapshot_date)
        elif r.status_code != 200:
            raise RuntimeError(f"HTTP error {r.status_code}: {r.text[:200]}")
        r = r.json()
    except Exception as e:
        logger.exception("Request failed")
    
    page = r["query"]["pages"].get(str(pageid), {})

    revision = (page.get("revisions") or [{}])[0]

    content = revision.get("slots", {}).get("main", {}).get("*", "")

    return {
        "pageid": pageid,
        "title": page.get("title"),
        "revision_id": revision.get("revid"),
        "timestamp": revision.get("timestamp"),
        "user": revision.get("user"),
        "comment": revision.get("comment"),
        "size": revision.get("size"),
        "page_length": page.get("length"),

        "categories": [
            c.get("title") for c in page.get("categories", [])
        ],

        "links": [
            l.get("title") for l in page.get("links", [])
        ],

        "content_hash": hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest() if content else None
    }

def build_snapshot_dataset(pages, snapshot_date):
    """
    Construye dataset wide para una fecha.
    """

    dataset = []

    for i, p in enumerate(pages):
        pageid = p["pageid"]

        data = get_page_snapshot(pageid, snapshot_date)

        row = {
            "snapshot_date": snapshot_date,
            "pageid": data["pageid"],
            "title": data["title"],

            "revision_id": data["revision_id"],
            "revision_timestamp": data["timestamp"],

            "user": data["user"],
            "comment": data["comment"],

            "size": data["size"],
            "page_length": data["page_length"],

            "categories": "|".join(data["categories"] or []),
            "links": "|".join(data["links"] or []),

            "content_hash": data["content_hash"],

            # métricas derivadas
            "num_categories": len(data["categories"] or []),
            "num_links": len(data["links"] or [])
        }

        dataset.append(row)

        if i % 100 == 0:
            logger.debug(f"Procesadas {i} [ {(i/PAGES_SIZE)*100}% ] páginas...")
            time.sleep(REQUEST_SLEEP)

    return dataset


def save_dataset(dataset, filename):
    """
    Guarda dataset en disco.
    """
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=2)

def main():
    logger.info("===================================")
    logger.info("Wikipedia Snapshot ETL - TFG")
    logger.info("===================================\n")

    pages = load_or_create_pages()

    logger.info(f"Páginas cargadas: {len(pages)}")

    logger.info("\nConstruyendo snapshot 1...")
    snapshot_1 = build_snapshot_dataset(pages, SNAPSHOT_DATE_1)

    logger.info("\n[STEP] Construyendo snapshot 2...")
    snapshot_2 = build_snapshot_dataset(pages, SNAPSHOT_DATE_2)

    logger.info("\n[STEP] Guardando resultados...")

    save_dataset(snapshot_1, f"{PREFIX_FILE}_snapshot_{PAGE_SIZES}_1.json")
    save_dataset(snapshot_2, f"{PREFIX_FILE}_snapshot_{PAGE_SIZES}_2.json")

    logger.info("\n[OK] Proceso completado")
    logger.debug(f" - {PREFIX_FILE}_snapshot_{PAGE_SIZES}_1.json")
    logger.debug(f" - {PREFIX_FILE}_snapshot_{PAGE_SIZES}_2.json")



if __name__ == "__main__":
    main()