from .titanic_utils import Config
from data_diff import connect_to_table, diff_tables
from sqlalchemy import inspect 

from tfg.datadiff_classifier.classifier import DiffClassifier
from tfg.datadiff_classifier.models import DiffRow, SegmentStructure

cfg = Config()

###################################
# Conectamos con los datasets
###################################

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

metadata = SegmentStructure(columnas=mysql_all_columns, pk=primary_key)
diff_results = diff_tables(table_mysql, table_pg)

#Creamos el objeto de clasificación y asociamos resultados de data-diff a DiffRows
clasificador = DiffClassifier()
diffrows = clasificador.parse_diff_results(diffs=diff_results, metadata=metadata)
 
###################################
# Clasificación de las diferencias
###################################
clasificaciones = []
clasificaciones = clasificador.classify_row_by_row(diffrows)