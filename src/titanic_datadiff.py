from titanic_utils import mysql_engine, mysql_engine_url, postgres_engine_url, dataset 
from data_diff import connect_to_table, diff_tables, disable_tracking
from sqlalchemy import inspect

disable_tracking()

src_table = dataset["raw"]["table"]
dst_table = dataset["modified"]["table"]

table_mysql = connect_to_table(
    f"mysql://{mysql_engine_url}",
    table_name=src_table,
    key_columns=("PassengerId")
)

table_pg = connect_to_table(
    f"postgresql://{postgres_engine_url}",
    table_name=dst_table,
    key_columns=("PassengerId")
)

mysql_metadata = inspect(mysql_engine)

mysql_all_columns=[col["name"] for col in mysql_metadata.get_columns(src_table)]
print(mysql_all_columns)

for diff in diff_tables(table_mysql, table_pg, extra_columns=mysql_all_columns):
    print(diff)

#diff_result = diff_tables(table_mysql, table_pg) 