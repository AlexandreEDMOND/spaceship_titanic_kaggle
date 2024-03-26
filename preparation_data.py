import pandas as pd

def supprimer_colonne(dataset, list_nom_colonne_a_supprimer):

    for nom_colonne in list_nom_colonne_a_supprimer:
        dataset = dataset.drop(nom_colonne, axis=1)
    
    return dataset

def creation_jeune(dataset, limit_age=15):

    list_age = []
    for age in dataset['Age']:
        if age < limit_age:
            list_age.append(True)
        else:
            list_age.append(False)
    dataset['Jeune'] = list_age

    dataset = supprimer_colonne(dataset, ['Age'])

    return dataset

def creation_deck_et_side(dataset):

    deck_letter = []
    for lettre in dataset["Cabin"]:
        deck_letter.append(lettre[0])
    dataset['Deck'] = deck_letter

    side_letter = []
    for lettre in dataset["Cabin"]:
        if lettre[-1] == 'S':
            side_letter.append(True)
        else:
            side_letter.append(False)
    dataset['Side_S'] = side_letter

    dataset = supprimer_colonne(dataset, ['Cabin'])

    return dataset

def creation_somme_depense(dataset):

    colonne_a_sommer = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    somme_individus = [0 for _ in range(len(dataset))]

    # On remplit les valeurs des colonnes vide par des 0
    for name_colomn in colonne_a_sommer:
        dataset[name_colomn] = dataset[name_colomn].fillna(0)

    # On fait la somme des valeurs des colonne à sommer
    for name_colomn in colonne_a_sommer:
        for index, element in enumerate(dataset[name_colomn]):
            somme_individus[index] += element

    # On créer la colonne à rajouter
    a_depenser = []
    for argent in somme_individus:
        if argent <= 0:
            a_depenser.append(False)
        else:
            a_depenser.append(True)
    dataset['Somme_depense'] = a_depenser

    # On supprime les colonnes à sommer
    dataset = supprimer_colonne(dataset, colonne_a_sommer)

    return dataset

nom_fichier_analyse = "data/test.csv"

df = pd.read_csv(nom_fichier_analyse)

df = supprimer_colonne(df, ['PassengerId', 'Name'])
df = creation_somme_depense(df)
print(df.head(5))

# Dataset avec les lignes où il n'y a pas de valeurs manquantes
dataset_sans_nan = df.dropna()
dataset_sans_nan = creation_jeune(dataset_sans_nan)
dataset_sans_nan = creation_deck_et_side(dataset_sans_nan)
print(dataset_sans_nan.head(5))

# Dataset avec les lignes où il y a des valeurs manquantes
dataset_avec_nan = df[df.isnull().any(axis=1)]
dataset_avec_nan['Transported'] = True
print(dataset_avec_nan.head(5))

# Sauvegarder le DataFrame modifié dans un fichier CSV
#dataset_sans_nan.to_csv("nouveau_fichier.csv", index=False)