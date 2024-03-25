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

nom_fichier_analyse = "data/train.csv"

data = pd.read_csv(nom_fichier_analyse)

data = data.dropna()

#####################################

colonne_id_groupe = []
for passenger in data["PassengerId"]:
    colonne_id_groupe.append(passenger[:4])



data["GroupId"] = colonne_id_groupe

print(data['GroupId'].value_counts().value_counts())
data['GroupId'] = pd.to_numeric(data['GroupId'])

calcul_correlation(data, 'GroupId')

show_correlation_class(data, 'HomePlanet')
mapping = {'Europa': 2, 'Earth': 0, 'Mars': 1}
data['HomePlanet_numeric'] = data['HomePlanet'].map(mapping)

calcul_correlation(data, 'HomePlanet_numeric')

data['CryoSleep'] = data['CryoSleep'].astype(bool)
show_correlation_class(data, 'CryoSleep')
calcul_correlation(data, 'CryoSleep')

cabin_letter = []
for lettre in data["Cabin"]:
    cabin_letter.append(lettre[-1])
data["Cabin_letter"] = cabin_letter
show_correlation_class(data, 'Cabin_letter')

cabin_letter = []
for lettre in data["Cabin"]:
    cabin_letter.append(lettre[0])
data["Cabin_deck"] = cabin_letter
show_correlation_class(data, 'Cabin_deck')

mapping = {'P': 0, 'S': 1}

data['Cabin_numeric'] = data['Cabin_letter'].map(mapping)

calcul_correlation(data, 'Cabin_numeric')

show_correlation_class(data, 'Destination')
show_correlation_class(data, 'VIP')

tranche_age = 5
list_age = []
for age in data['Age']:
    list_age.append(int(age/tranche_age))
data["Tranche_age"] = list_age
#show_correlation_class(data, 'Tranche_age')

limit_age = 15
list_age = []
for age in data['Age']:
    if age < limit_age:
        list_age.append(0)
    else:
        list_age.append(1)
data["Limit_age"] = list_age
show_correlation_class(data, 'Limit_age')


name_add = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
somme_individus = [0 for _ in range(len(data))]
for name_colomn in name_add:
    for index, element in enumerate(data[name_colomn]):
        somme_individus[index] += element

rich_list = []
for argent in somme_individus:
    if argent <= 0:
        rich_list.append(0)
    else:
        rich_list.append(1)
data["Depense"] = rich_list
print("Dépense")
show_correlation_class(data, 'Depense')
