"""
Construye snapshots históricos de páginas de Wikipedia via API URL de Wikipedia
"""

import requests
import json
import os
import time
import hashlib
from datetime import datetime

import logging
from tfg.logging_config import setup_logging, timed
setup_logging(level="DEBUG", log_file="logs/wiki_snapshot.log")


logger = logging.getLogger("tfg.wikipedia_poc.wiki_scrape")

# ===================== CONFIG =====================
API_URL = "https://es.wikipedia.org/w/api.php"   
PAGE_LIMIT = 10000
REQUEST_SLEEP = 0.3
PREFIX_FILE = "data/raw/wiki"

SNAPSHOT_DATE_1 = "2024-01-01T00:00:00Z"
SNAPSHOT_DATE_2 = "2024-02-01T00:00:00Z"

PAGES_FILE = f"{PREFIX_FILE}_pages_{PAGE_LIMIT}.json"
headers = {"User-Agent": "TFG-Wikipedia-Snapshot-Builder (xavi.tfg@example.com)"}



def fetch_random_pages(limit=PAGE_LIMIT):
    """Obtiene páginas aleatorias (reproducible)."""
    pages = []
    while len(pages) < limit:
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnlimit": 500,
            "rnnamespace": 0
        }

        response = requests.get(API_URL, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        pages.extend(data["query"]["random"])
        
        if i % 500 == 0 and i > 0:
            logger.debug(f"Procesadas {i:5,}/{len(pages):,} páginas ({i/len(pages):.1%})")

    return [{"pageid": p["id"], "title": p["title"]} for p in pages[:limit]]


def load_or_create_pages():
    """Carga páginas o genera nuevas (para reproducibilidad)."""
    if os.path.exists(PAGES_FILE):
        logger.info(f"Cargando páginas desde {PAGES_FILE}")
        with open(PAGES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Generando nuevo conjunto de páginas aleatorias...")
    pages = fetch_random_pages(PAGE_LIMIT)

    #os.makedirs(os.path.dirname(PAGES_FILE), exist_ok=True)
    with open(PAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    logger.info(f"Guardadas {len(pages)} páginas en {PAGES_FILE}")
    return pages


def get_page_snapshot(pageid: int, snapshot_date: str):
    """Obtiene el estado de una página en una fecha concreta."""
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
        "pllimit": 50,
        "redirects": 1
    }

    try:
        r = requests.get(API_URL, params=params, headers=headers, timeout=20)
        
        if r.status_code == 429:
            retry = int(r.headers.get("Retry-After", 8))
            logger.warning(f"Rate limit → esperando {retry}s")
            time.sleep(retry)
            return get_page_snapshot(pageid, snapshot_date)

        r.raise_for_status()
        data = r.json()

        page = data["query"]["pages"].get(str(pageid), {})
        if not page or "missing" in page:
            return {"pageid": pageid, "title": None, "error": "missing"}

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
            "categories": [c.get("title") for c in page.get("categories", [])],
            "links": [l.get("title") for l in page.get("links", [])],
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None,
        }

    except Exception as e:
        logger.error(f"Error en pageid {pageid}: {e}")
        return {"pageid": pageid, "title": None, "error": str(e)}


def build_snapshot_dataset(pages, snapshot_date):
    """Construye snapshot para una fecha."""
    dataset = []
    
    for i, p in enumerate(pages):
        data = get_page_snapshot(p["pageid"], snapshot_date)

        row = {
            "snapshot_date": snapshot_date,
            "pageid": data["pageid"],
            "title": data.get("title"),
            "revision_id": data.get("revision_id"),
            "revision_timestamp": data.get("timestamp"),
            "user": data.get("user"),
            "comment": data.get("comment"),
            "size": data.get("size"),
            "page_length": data.get("page_length"),
            "categories": "|".join(data.get("categories", [])),
            "links": "|".join(data.get("links", [])),
            "content_hash": data.get("content_hash"),
            "num_categories": len(data.get("categories", [])),
            "num_links": len(data.get("links", [])),
            "error": data.get("error")
        }

        dataset.append(row)

        if i % 100 == 0 and i > 0:
            logger.info(f"Procesadas {i}/{len(pages)} páginas ({i/len(pages):.1%})")
            time.sleep(REQUEST_SLEEP)

    return dataset


def save_dataset(dataset, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    logger.info(f"Guardado: {filename} ({len(dataset)} registros)")


def main():
    logger.info("=== Wikipedia Snapshot Scraper ===")
    logger.info(f"Páginas a procesar: {PAGE_LIMIT:,}")

    with timed (logger, "Carga o crea páginas") :
     pages = load_or_create_pages()

    with timed (logger, f"Creando snapshot 1: {SNAPSHOT_DATE_1}") :
        snapshot_1 = build_snapshot_dataset(pages, SNAPSHOT_DATE_1)

    with timed (logger, f"\nConstruyendo snapshot 2: {SNAPSHOT_DATE_2}"):
        snapshot_2 = build_snapshot_dataset(pages, SNAPSHOT_DATE_2)

    save_dataset(snapshot_1, f"{PREFIX_FILE}_snapshot_{PAGE_LIMIT}_2024-01.json")
    save_dataset(snapshot_2, f"{PREFIX_FILE}_snapshot_{PAGE_LIMIT}_2024-02.json")

    logger.info(f"Procesos de scraping wikipedia de {PAGE_LIMIT} páginas finalizado")
    logger.info(f"Total páginas procesadas: {len(pages):,}")


if __name__ == "__main__":
    main()