from sqlalchemy import create_engine
from data_diff import disable_tracking
class Config:
    DEBUG = True
    # Dataset
    DATASET = {
        'raw': {
            'file': '../data/raw/titanic.csv', # desde  ./src
            'table': 'titanic'
        },
        'modified': {
            'file': '../data/modified/titanic_modified.csv',
            'table': 'titanic_modified'
        }
    }
    # MySQL
    MYSQL = {
        "user": "poc",
        "password": "poc3306",
        "host": "localhost",
        "port": "3306",
        "db": "pocdb",
        "dialect" : 'mysql',
        "driver" : 'mysqlconnector'
    }
    # PostgreSQL
    POSTGRES = {
        "user": "poc",
        "password": "poc5432",
        "host": "localhost",
        "port": "5432",
        "db": "tfgdb",
        "dialect" : "postgresql",
        "driver" : "psycopg2"
    }

    def __init__(self):
        disable_tracking()

        self.mysql_engine = self._create_engine(self.MYSQL)
        self.postgresql_engine = self._create_engine(self.POSTGRES)

    @staticmethod
    def get_url(cfg: dict) -> str:
        return f"{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['db']}"

    @staticmethod
    def getConnectionString(cfg: dict, datadiff=False):
        if datadiff:
            return f"{cfg['dialect']}://{Config.get_url(cfg)}"
        return f"{cfg['dialect']}+{cfg['driver']}://{Config.get_url(cfg)}"

    def _create_engine(self, cfg: dict):
        return create_engine(self.getConnectionString(cfg))

    # ---------------------------
    # Test conexiones
    # ---------------------------
    @staticmethod
    def test_connections(self):
        try:
            with self.mysql_engine.connect():
                print(f"Conexión MySQL OK: {self.MYSQL['db']} en {self.MYSQL['host']}")
            with self.postgresql_engin.connect():
                print(f"Conexión PostgreSQL OK: {self.POSTGRES['db']} en {self.POSTGRES['host']}")
        except Exception as e:
            print(f"Error de conexión: {e}")