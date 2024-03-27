import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from network import *

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


def prepera_data_training(nom_fichier_analyse="data/train.csv"):

    df = pd.read_csv(nom_fichier_analyse)

    df = supprimer_colonne(df, ['PassengerId', 'Name'])
    df = creation_somme_depense(df)

    dataset_sans_nan = df.dropna()

    dataset_sans_nan = creation_jeune(dataset_sans_nan)
    dataset_sans_nan = creation_deck_et_side(dataset_sans_nan)

    dataset_sans_nan['CryoSleep'] = dataset_sans_nan['CryoSleep'].astype(bool)
    dataset_sans_nan['VIP'] = dataset_sans_nan['VIP'].astype(bool)

    dataset_sans_nan = pd.get_dummies(dataset_sans_nan[:])

    print(dataset_sans_nan.head(5))

    dataset_sans_nan.to_csv("train_clean.csv")

    return dataset_sans_nan

def train_network(df_train):

    # Séparer les caractéristiques et la cible
    X = df_train.drop('Transported', axis=1)
    y = df_train['Transported']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir en tenseurs PyTorch
    X_train_tensor = torch.tensor(X_train.values).float()
    y_train_tensor = torch.tensor(y_train.values).float().view(-1, 1)  # Remodeler pour correspondre aux dimensions attendues par PyTorch
    X_test_tensor = torch.tensor(X_test.values).float()
    y_test_tensor = torch.tensor(y_test.values).float().view(-1, 1)

    model = Net()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Boucle d'entraînement
    epochs = 1000
    list_loss = []
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        list_loss.append(loss.item())
        
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predicted_classes = predictions.round()  # Arrondir pour obtenir 0 ou 1
        accuracy = (predicted_classes.eq(y_test_tensor).sum() / float(y_test_tensor.shape[0])).item()
        print(f'Accuracy: {accuracy:.4f}')
    
    # Enregistrer l'état du modèle
    torch.save(model.state_dict(), f'network_train/network_spaceship_{accuracy:.4f}.pth')

    plt.plot(list_loss)
    plt.show()



data_traing = prepera_data_training()
train_network(data_traing)
# Dataset avec les lignes où il n'y a pas de valeurs manquantes



# Dataset avec les lignes où il y a des valeurs manquantes
# dataset_avec_nan = df[df.isnull().any(axis=1)]
# dataset_avec_nan['Transported'] = True
# print(dataset_avec_nan.head(5))

# Sauvegarder le DataFrame modifié dans un fichier CSV
#dataset_sans_nan.to_csv("nouveau_fichier.csv", index=False)