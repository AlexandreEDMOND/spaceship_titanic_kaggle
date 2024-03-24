import pandas as pd

nom_fichier_analyse = "data/train.csv"

data = pd.read_csv(nom_fichier_analyse)

for colonne in data.columns:
    print(data[colonne].value_counts(dropna=False))

data_filtre = data.dropna()

for colonne in data_filtre.columns:
    print(data_filtre[colonne].value_counts(dropna=False))