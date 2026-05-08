import os
import json
import logging
from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    MetaData,
    Table
)

# -------------------
# CONFIG
# -------------------
#DB1_URL = "sqlite:///data/db/wikipedia_DB1.db"
#DB1_URL = os.getenv("WIKIPEDIA_PGSQL_URL") "postgresql+psycopg2://user:password@localhost:5432/wikipedia_poc"
DB1_URL = os.getenv("WIKIPEDIA_MYSQL_URL")
DB2_URL = os.getenv("WIKIPEDIA_PGSQL_URL")

SNAPSHOT_DIR = "data/raw/"

SNAPSHOT_LIST = (
    "wiki_snapshot_1000_2024-01.json",
    "wiki_snapshot_1000_2024-02.json",
)

import logging
from tfg.logging_config import setup_logging, timed
setup_logging(level="DEBUG", log_file="logs/wiki_load_to_db.log")


logger = logging.getLogger("tfg.wikipedia_poc.wiki_load_to_db")

# -------------------
# DB SETUP
# -------------------
engine1 = create_engine(DB1_URL, echo=False)
engine2 = create_engine(DB2_URL, echo=False)
metadata = MetaData()

# -------------------
# UTILS
# -------------------
def parse_datetime(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def get_table_name(json_file):
    name = os.path.basename(json_file)
    name = name.replace(".json", "")
    return name


# -------------------
# TABLE FACTORY
# -------------------
def create_snapshot_table(table_name):
    table = Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("snapshot_date", DateTime),
        Column("pageid", Integer, index=True),
        Column("title", String(512)),
        Column("revision_id", Integer),
        Column("revision_timestamp", DateTime),
        Column("user", String(255)),
        Column("comment", Text),
        Column("size", Integer),
        Column("page_length", Integer),
        Column("categories", Text),
        Column("links", Text),
        Column("content_hash", String(128)),
        Column("num_categories", Integer),
        Column("num_links", Integer),
        Column("error", Text),
    )

    table.create(engine1, checkfirst=True)
    table.create(engine2, checkfirst=True)
    return table


# -------------------
# EXISTENCE CHECK
# -------------------
def is_table_populated(engine, table):
    with engine.connect() as conn:
        result = conn.execute(table.select().limit(1)).fetchone()
        return result is not None
    


# -------------------
# IMPORT FUNCTION
# -------------------
def import_json_dynamic(file_path):
    table_name = get_table_name(file_path)

    logger.info(f"Preparando tabla: {table_name}")

    table = create_snapshot_table(table_name)

    # Evitar recargar si ya tiene datos
    is_engine1_populated = is_table_populated(engine1, table)
    if is_engine1_populated:
        logger.info(f"Engine 1 ( {engine1.dialect.name} ). Tabla {table_name} ya contiene datos. Se omite.")
        
    is_engine2_populated = is_table_populated(engine2, table)    
    if is_engine2_populated:
        logger.info(f"Engine 2 ( {engine2.dialect.name} ). Tabla {table_name} ya contiene datos. Se omite.")
           

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        logger.warning(f"JSON vacío: {file_path}")
        return

    rows = []

    for item in data:
        rows.append({
            "snapshot_date": parse_datetime(item.get("snapshot_date")),
            "pageid": item.get("pageid"),
            "title": item.get("title"),
            "revision_id": item.get("revision_id"),
            "revision_timestamp": parse_datetime(item.get("revision_timestamp")),
            "user": item.get("user"),
            "comment": item.get("comment"),
            "size": item.get("size"),
            "page_length": item.get("page_length"),
            "categories": item.get("categories"),
            "links": item.get("links"),
            "content_hash": item.get("content_hash"),
            "num_categories": item.get("num_categories"),
            "num_links": item.get("num_links"),
            "error": item.get("error"),
        })

    # Insert batch
    if not is_engine1_populated : 
        logger.debug(f"Poblando filas en engine1 ( {engine1.dialect.name} ).")
        with engine1.begin() as conn:
            conn.execute(table.insert(), rows)
    
    if not is_engine2_populated:
        logger.debug(f"Poblando filas en engine2 ( {engine2.dialect.name} ).")
        with engine2.begin() as conn:
            conn.execute(table.insert(), rows)

    logger.info(f"Insertados {len(rows)} registros en {table_name}")


# -------------------
# MAIN
# -------------------
if __name__ == "__main__":

    logger.info("=== Wikipedia Snapshot Loader ===")

    for json_file in SNAPSHOT_LIST:
        path = os.path.join(SNAPSHOT_DIR, json_file)

        if not os.path.exists(path):
            logger.warning(f"No existe: {path}")
            continue

        logger.info(f"Importando fichero {json_file}")
        import_json_dynamic(path)

    logger.info("Proceso finalizado")