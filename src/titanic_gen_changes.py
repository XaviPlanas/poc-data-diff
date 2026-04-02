import pandas as pd

df = pd.read_csv("data/raw/titanic.csv")

df_mod = df.copy()

# Cambios simulados
df_mod.loc[0, "Age"] = 99
df_mod.loc[1, "Name"] = "Modified Passenger"

# eliminar fila
df_mod = df_mod.drop(index=[2])

# añadir fila
new_row = df.iloc[3].copy()
new_row["PassengerId"] = 9999
df_mod = pd.concat([df_mod, pd.DataFrame([new_row])])

df_mod.to_csv("data/modified/titanic_modified.csv", index=False)
