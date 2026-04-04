from sqlalchemy import create_engine

dataset = {'raw' : {
                'file' : 'data/raw/titanic.csv',
                'table': 'titanic' } ,
            'modified' : {
                'file' : 'data/modified/titanic_modified.csv',
                'table': 'titanic_modified'
            }
        }
# ---------------------------
# Configuración MySQL (hardcodeada)
# ---------------------------
MYSQL_USER = "poc"
MYSQL_PASSWORD = "poc3306"
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
MYSQL_DB = "pocdb"

mysql_engine_url = f"{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
mysql_engine = create_engine(f"mysql+mysqlconnector://{mysql_engine_url}")

# ---------------------------
# Configuración PostgreSQL (hardcodeada)
# ---------------------------
PG_USER = "poc"
PG_PASSWORD = "poc5432"
PG_HOST = "localhost"
PG_PORT = "5432"
PG_DB = "tfgdb"

postgres_engine_url = f"{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
postgres_engine = create_engine(f"postgresql+psycopg2://{postgres_engine_url}")

# ---------------------------
# Función de prueba de conexión
# ---------------------------
def test_connections():
    try:
        with mysql_engine.connect() as conn:
            print(f"Conexión MySQL OK: {MYSQL_DB} en {MYSQL_HOST}")
        with postgres_engine.connect() as conn:
            print(f"Conexión PostgreSQL OK: {PG_DB} en {PG_HOST}")
    except Exception as e:
        print(f"Error de conexión: {e}")