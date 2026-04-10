import pandas as pd
import numpy as np
from sqlalchemy.types import Integer, String, Float, Numeric
from sqlalchemy import create_engine, Table, Column, MetaData
from sqlalchemy import Integer, Float, String, Numeric

from tfg.titanic_poc.titanic_utils import mysql_engine, postgres_engine, dataset

load_method = 'pandas' # pandas | orm 

#Detectamos automáticamente las definiciones de df de Pandas
def orm_autotype(series):

    if pd.api.types.is_integer_dtype(series):
        return Integer()

    if pd.api.types.is_float_dtype(series):
        #return Float()
        return Numeric(10,3)

    if pd.api.types.is_numeric_dtype(series):
        return Numeric()

    return String(255)

#Recrea tabla e inserta datos
def orm_to_db(df, table_name, engine):

    metadata = MetaData()

    columns = []

    for col in df.columns:

        col_type = orm_autotype(df[col])

        if col == "PassengerId":
            columns.append(Column(col, col_type, primary_key=True))
        else:
            columns.append(Column(col, col_type))

    table = Table(table_name, metadata, *columns)

    metadata.drop_all(engine, [table])
    metadata.create_all(engine)

    # Reemplazar NaN de Pandas por None
    df_clean = df.replace({np.nan: None})
    records = df_clean.to_dict(orient="records")

    with engine.begin() as conn:
        conn.execute(table.insert(), records)

    print(f"{table_name} cargada en {engine.url.database}")

#Método para hacer la carga con Pandas
def pandas_to_db(df, table_name, engine, dtype_mapping):
    try:
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False, dtype=dtype_mapping)
        print(f"{table_name} cargada correctamente en {engine.url.database}")
        
    except Exception as e:
        print(f"Error cargando {table_name} en {engine.url.database}: {e}")

# Leer dataset Titanic
df = pd.read_csv(dataset["raw"]["file"])
table_raw = dataset["raw"]["table"]

df_modified = pd.read_csv(dataset["modified"]["file"])
table_modified = dataset["modified"]["table"]

if load_method == 'orm':
    # Cargar en MySQL
    orm_to_db(df, table_raw, mysql_engine)
    orm_to_db(df, table_modified, mysql_engine)

    # Cargar en PostgreSQL
    orm_to_db(df, table_raw, postgres_engine)
    orm_to_db(df, table_modified, postgres_engine)

elif load_method == 'pandas' :
    
    # Definición explícita de tipos de columna para SQL en Pandas
    dtype_mapping = {
        'PassengerId': Integer(),
        'Survived': Integer(),
        'Pclass': Integer(),
        'Name': String(255),
        'Sex': String(10),
        'Age': Float(),
        'SibSp': Integer(),
        'Parch': Integer(),
        'Ticket': String(50),
        'Fare': Numeric(10,3),
        'Cabin': String(50),
        'Embarked': String(5)
    }
    # Cargar en MySQL
    pandas_to_db(df, table_raw, mysql_engine, dtype_mapping)
    pandas_to_db(df_modified, table_modified, mysql_engine, dtype_mapping)

    # Cargar en PostgreSQL
    pandas_to_db(df, table_raw, postgres_engine, dtype_mapping)
    pandas_to_db(df_modified, table_modified, postgres_engine, dtype_mapping)
