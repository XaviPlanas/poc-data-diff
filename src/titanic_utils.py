from sqlalchemy import create_engine

# ---------------------------
# Configuración MySQL (hardcodeada)
# ---------------------------
MYSQL_USER = "poc"
MYSQL_PASSWORD = "poc3306"
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
MYSQL_DB = "pocdb"

mysql_engine = create_engine(
    f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)

# ---------------------------
# Configuración PostgreSQL (hardcodeada)
# ---------------------------
PG_USER = "poc"
PG_PASSWORD = "poc5432"
PG_HOST = "localhost"
PG_PORT = "5432"
PG_DB = "tfgdb"

postgres_engine = create_engine(
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

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