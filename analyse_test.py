import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def show_correlation_class(df, colonne_1, colonne_2='Transported'):
    count_value = df[colonne_1].value_counts()

    element_colonne = []
    for element in df[colonne_1]:
        if element not in element_colonne:
            element_colonne.append(element)
    
    print("\nTableau de proportion")
    for element in element_colonne:
        colonnes_a_garder = df.loc[df[colonne_1] == element, [colonne_1, colonne_2]]
        nmbre = count_value[element]
        nmbre_true = colonnes_a_garder[colonne_2].value_counts()[True]
        print(f"{element} : {nmbre_true/nmbre}")


def calcul_correlation(df, colonne_1, colonne_2='Transported'):
    type_donnees = type(df[colonne_1][0])

    if type_donnees == str:
        print("La colonne", colonne_1, "contient des string.")


    elif type_donnees == np.int64 or type_donnees == np.float64:
        print("La colonne", colonne_1, "contient des int.")
        correlation = df[colonne_1].corr(df[colonne_2])
        print(f"Corrélation entre {colonne_1} et {colonne_2} :", correlation)
        #return correlation

    elif type_donnees == np.bool_:
        print("La colonne", colonne_1, "contient des booléens.")
        correlation = df[colonne_1].corr(df[colonne_2])
        print(f"Corrélation entre {colonne_1} et {colonne_2} :", correlation)

    else:
        print("La colonne", colonne_1, "contient d'autres types de données : ", type_donnees)

#####################################

nom_fichier_analyse = "data/test.csv"

data = pd.read_csv(nom_fichier_analyse)
print(data)
data = data.dropna()
print(data)

#####################################
