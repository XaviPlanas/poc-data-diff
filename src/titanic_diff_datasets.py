import pandas as pd

df1 = pd.read_csv("data/raw/titanic.csv")
df2 = pd.read_csv("data/modified/titanic_modified.csv")

merged = df1.merge(df2, on="PassengerId", how="outer", indicator=True)

added = merged[merged["_merge"] == "right_only"]
removed = merged[merged["_merge"] == "left_only"]

print("Filas añadidas:", len(added))
print("Filas eliminadas:", len(removed))
