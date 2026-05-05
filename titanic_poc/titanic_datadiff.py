###################################
# Escenario: 
#   Seguimos pipeline de comparación 
#   pivotando sobre DiffClassifier
#   y usando la API data-diff como 
#   herramienta de comparación.
###################################

from .titanic_utils import Config
from data_diff import connect_to_table, diff_tables
from sqlalchemy import inspect 

from tfg.datadiff_classifier.classifier import DiffClassifier
from tfg.datadiff_classifier.models import SegmentStructure

import logging
from tfg.logging_config import setup_logging, timed

logger = logging.getLogger("tfg.titanic_poc.titanic_datadiff")
setup_logging(level="DEBUG")
logger.debug("Cargando locales y configuración")

cfg = Config()

###################################
# Conectamos con los datasets
###################################
with timed(logger, "Conexión a tablas y mapeo", level="INFO"):
    #Mapeamos las tablas para conectarlas y compararlas con data-diff
    table_raw = Config.DATASET["raw"]["table"]
    table_modified = Config.DATASET["modified"]["table"]
    primary_key = "PassengerId"

    mysql_metadata = inspect(cfg.mysql_engine)
    mysql_all_columns=[col["name"] for col in mysql_metadata.get_columns(table_raw)]

    table_mysql = connect_to_table(
        #f"{Config.MYSQL["dialect"]}://{Config.mysql_engine_url}",
        cfg.getConnectionString(Config.MYSQL,datadiff=True ),
        table_name=table_raw,
        key_columns=primary_key,
        extra_columns = mysql_all_columns
    )

    table_pg = connect_to_table(
        #f"{Config.MYSQL["dialect"]}://{Config.postgres_engine_url}",
        cfg.getConnectionString(Config.POSTGRES,datadiff=True ),
        table_name=table_modified,
        key_columns=("PassengerId"),
        extra_columns = mysql_all_columns
    )

###################################
# Búsqueda de  diferencias (API data-diff)
###################################
with timed(logger, "Búsqueda de diferencias con data-diff", level="INFO"):
    metadata = SegmentStructure(columnas=mysql_all_columns, pk=primary_key)
    diff_results = diff_tables(table_mysql, table_pg)

    #Creamos el objeto de clasificación y asociamos resultados de data-diff a DiffRows
    clasificador = DiffClassifier()
    diffrows = clasificador.parse_to_diffrows(metadata=metadata, diffs=diff_results)
    
###################################
# Clasificación de las diferencias
###################################
with timed(logger, "Clasificación de diferencias", level="INFO"):
    clasificaciones = []
    clasificaciones = clasificador.classify_row_by_row(diffrows,30)

###################################
# Reportamos las diferencias
###################################

logger.info(f"Total de clasificaciones obtenidas: {len(clasificaciones)}")

# print("\nEjemplos de 5 clasificaciones:\n") 
# for c in clasificaciones[:5] :
#     print(c.to_json())

clasificador.report(clasificaciones)

clasificador.report_details(clasificaciones)

