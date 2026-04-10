from titanic_utils import mysql_engine, mysql_engine_url, postgres_engine_url, dataset 
from data_diff import connect_to_table, diff_tables, disable_tracking
from sqlalchemy import inspect

disable_tracking()

table_raw = dataset["raw"]["table"]
table_modified = dataset["modified"]["table"]

mysql_metadata = inspect(mysql_engine)
mysql_all_columns=[col["name"] for col in mysql_metadata.get_columns(table_raw)]
#mysql_all_columns = []

table_mysql = connect_to_table(
    f"mysql://{mysql_engine_url}",
    table_name=table_raw,
    key_columns=("PassengerId"),
    extra_columns = mysql_all_columns
)

table_pg = connect_to_table(
    f"postgresql://{postgres_engine_url}",
    table_name=table_modified,
    key_columns=("PassengerId"),
    extra_columns = mysql_all_columns
)

added = 0
removed = 0
total = 0
#for diff in diff_tables(table_mysql, table_pg, extra_columns=mysql_all_columns):
for diff in diff_tables(table_mysql, table_pg):
   total+=1
   if diff[0] == '+' :
      added+=1
   elif diff[0] == '-' :
      removed+=1

   print(diff)

print("Cambios detectados en total:", total)
print("Insertados:", added)
print("Eliminados:", removed)
