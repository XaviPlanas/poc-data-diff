import pandas as pd
from sqlalchemy.types import Integer, String, Float

# Configuración de conexiones
from titanic_utils import mysql_engine, postgres_engine

# Leer dataset Titanic
df = pd.read_csv("data/raw/titanic.csv")

# Mapear tipos de columna para SQL
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
    'Fare': Float(),
    'Cabin': String(50),
    'Embarked': String(5)
}

def load_to_db(df, table_name, engine, dtype_mapping):
    try:
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False, dtype=dtype_mapping)
        print(f"{table_name} cargada correctamente en {engine.url.database}")
    except Exception as e:
        print(f"Error cargando {table_name} en {engine.url.database}: {e}")

# Cargar en MySQL
load_to_db(df, "titanic", mysql_engine, dtype_mapping)

# Cargar en PostgreSQL
load_to_db(df, "titanic", postgres_engine, dtype_mapping)