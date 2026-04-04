import pandas as pd
import random
import uuid
from datetime import timedelta
from titanic_utils import dataset

### Constantes 
random.seed(27) # para que sea reproducible emtre ejecuciones

# porcentaje de cambios
UPDATE_RATE = 0.05
DELETE_RATE = 0.03
INSERT_RATE = 0.04

### Funciones
def mutate_value(v):
    """Modifica un valor respetando su tipo."""
    #TODO: incluir funciones normalización?
    
    if pd.isna(v):
        return v

    # strings
    if isinstance(v, str):
        return v + "_mod"

    # enteros
    if isinstance(v, int):
        return v + random.randint(-5, 5)

    # floats
    if isinstance(v, float):
        return round(v + random.uniform(-1.0, 1.0), 3)

    # booleanos
    if isinstance(v, bool):
        return not v

    # fechas
    if isinstance(v, pd.Timestamp):
        return v + timedelta(days=random.randint(-10, 10))

    return v

dataset_input = dataset["raw"]
dataset_output = dataset["modified"]

df = pd.read_csv(dataset_input)
initial_rows = len(df)

# -------------------
# DELETE filas
# -------------------

delete_count = int(len(df) * DELETE_RATE)
delete_idx = random.sample(list(df.index), delete_count)
df = df.drop(delete_idx)

# -------------------
# UPDATE filas
# -------------------

update_count = int(len(df) * UPDATE_RATE)
update_idx = random.sample(list(df.index), update_count)

for i in update_idx:
    col = random.choice(df.columns[1:])  # evita modificar PassengerId
    df.at[i, col] = mutate_value(df.at[i, col])

# -------------------
# INSERT filas nuevas
# -------------------

insert_count = int(len(df) * INSERT_RATE)

for _ in range(insert_count):
    new_row = df.sample(1).iloc[0].copy()
    new_row[df.columns[0]] = str(uuid.uuid4())[:8]  # nuevo id
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df.to_csv(dataset_output, index=False)

print("Filas originales:", initial_rows)
print("Filas finales:", len(df))
print("Deletes:", delete_count)
print("Updates:", update_count)
print("Inserts:", insert_count)
