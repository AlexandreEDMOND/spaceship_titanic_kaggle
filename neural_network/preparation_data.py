import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from network import *

# Fonction permettant de supprimer des colonnes d'un DataFrame à partir d'une liste des noms des colonnes
def supprimer_colonne(df, list_nom_colonne_a_supprimer):

    # Pour chaque noms de la liste, on supprime la colonne
    for nom_colonne in list_nom_colonne_a_supprimer:
        df = df.drop(nom_colonne, axis=1)
    
    return df

# Fonction permettant des créer la colonne 'Jeune' à partir d'une colonne 'Age'
# Converti ainsi une valeur numérique en booléen
def creation_jeune(df, limit_age=15):

    # Initialisation de la colonne 'Jeune' à False
    df['Jeune'] = False
    
    for index, row in df.iterrows():
        if row['Age'] < limit_age:
            df.at[index, 'Jeune'] = True

    df = supprimer_colonne(df, ['Age'])

    return df

# Fonction permettant des créer la colonne 'Deck' et 'Side_S' à partir d'une colonne 'Deck'
# Converti ainsi une string en classe et booléen
def creation_deck_et_side(df):

    # Initialisation de la colonne à ''
    df['Deck'] = ''

    # Mise à jour de la colonne 'Deck' en fonction de la 1er lettre de 'Cabin'
    for index, row in df.iterrows():
        cabin = row['Cabin']
        df.at[index, 'Deck'] = cabin[0]

    # Initialisation de la colonne à True
    df['Side_S'] = True

    for index, row in df.iterrows():
        cabin = row['Cabin']
        df.at[index, 'Side_S'] = (cabin[-1] == 'S')

    # On supprime la colonne 'Cabin'
    df = supprimer_colonne(df, ['Cabin'])

    return df

# Fonction permettant des créer la colonne 'Somme_depense' à partir des colonnes 'colonne_a_sommer'
# Converti ainsi plusieurs colonnes numériques en booléen
def creation_somme_depense(df, colonne_a_sommer=["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]):

    # Initialisation des sommes dépensées de chaque individu
    somme_individus = [0 for _ in range(len(df))]

    # On remplit les valeurs des colonnes vide par des 0
    for name_colomn in colonne_a_sommer:
        df[name_colomn] = df[name_colomn].fillna(0)

    # On fait la somme des valeurs des colonnes à sommer
    for name_colomn in colonne_a_sommer:
        for index, element in enumerate(df[name_colomn]):
            somme_individus[index] += element

    # On initialise la colonne à 0
    df['Somme_depense'] = False

    # En fonction des dépenses, on actualise la colonne 'Somme_depense'
    for index, argent in enumerate(somme_individus):
        if argent > 0:
            df.loc[index, 'Somme_depense'] = True

    # On supprime les colonnes à sommer
    df = supprimer_colonne(df, colonne_a_sommer)

    return df

def make_prediction(neural_network_path, data_test_path="data/test.csv"):

    # Convertion du fichier CSV de test en DataFrame
    df = pd.read_csv(data_test_path)

    # On enlève les colonnes inutiles
    df = supprimer_colonne(df, ['Name'])

    # On créer la colonne 'Somme_depense'
    df = creation_somme_depense(df)

    # On isole la partie du DataFrame qui n'a pas de Nan
    # Cette partie permet d'obtenir une prédiction du Netwok
    df_sans_nan = df.dropna()

    df_sans_nan = creation_jeune(df_sans_nan)
    df_sans_nan = creation_deck_et_side(df_sans_nan)

    df_sans_nan['CryoSleep'] = df_sans_nan['CryoSleep'].astype(bool)
    df_sans_nan['VIP'] = df_sans_nan['VIP'].astype(bool)

    df_sans_nan_sans_id = supprimer_colonne(df_sans_nan, ['PassengerId'])
    df_sans_nan_sans_id = pd.get_dummies(df_sans_nan_sans_id[:])

    X = torch.tensor(df_sans_nan_sans_id.values).float()

    # On créer un modèle avec les bonnes dimensions
    model = Net(X.shape[1])
    # On charge le modèle sur lequel on veut faire les prédictions
    model.load_state_dict(torch.load(neural_network_path))
    # On met le modèle en mode évaluation
    model.eval()

    # On effectue les prédictions
    with torch.no_grad():
        predictions = model(X)
        predicted_classes = predictions.round()     # On arrondit les prédictions pour avoir 0 ou 1

    # Convertir le tenseur en DataFrame pandas
    df_predicted_classes = pd.DataFrame(predicted_classes.numpy(), columns=['Transported'])
    df_sans_nan = df_sans_nan['PassengerId'].reset_index(drop=True)
    merged_df = pd.concat([df_sans_nan, df_predicted_classes], axis=1)

    # On isole la partie du DataFrame quia des Nan
    # Cette partie ne permet pas d'obtenir une prédiction du Netwok
    # On mets tous à 'True'
    df_avec_nan = df[df.isnull().any(axis=1)]
    df_avec_nan['Transported'] = True
    df_nan = df_avec_nan[['PassengerId', 'Transported']]

    # On créer le DataFrame final avec tous les valeurs prédites et compléter
    final_df = pd.concat([merged_df, df_nan], axis = 0)

    # Conversion des valeurs 0 et 1 en valeurs booléennes
    # Demandée par Kaagle
    final_df['Transported'] = final_df['Transported'].astype(bool)

    # On enregistre le DataFrame en fichier CSV pour la submission de Kaagle
    final_df.to_csv("neural_network/submission.csv", index=False)

# Fonction permettant de créer un DataFrame adapté pour entrainer le réseaux de neurone
# Prend en entrée le chemin  vers le fichier CSV de train
# Renvoie le DataFrame adapté
def prepera_data_for_training(chemin_fichier_train="data/train.csv"):

    # On import le CSV
    df = pd.read_csv(chemin_fichier_train)

    # On supprime les colonnes qui sont inutiles à l'entraînement
    df = supprimer_colonne(df, ['PassengerId', 'Name'])

    # On créer une colonne 'Somme_depense'
    df = creation_somme_depense(df)

    # On garde uniquement les lignes sans aucunes valeurs manquantes
    df_sans_nan = df.dropna()

    # On créer la colonne 'Jeune'
    df_sans_nan = creation_jeune(df_sans_nan)
    # On créer la colonne 'Deck' et 'Side_S'
    df_sans_nan = creation_deck_et_side(df_sans_nan)

    # On converti les colonne 'CryoSleep' et 'VIP' en booléen
    df_sans_nan['CryoSleep'] = df_sans_nan['CryoSleep'].astype(bool)
    df_sans_nan['VIP'] = df_sans_nan['VIP'].astype(bool)

    # On converti les variables catégorielles en dummies
    df_sans_nan = pd.get_dummies(df_sans_nan[:])

    # On enregistre le DataFrame sous format CSV
    df_sans_nan.to_csv("neural_network/dataframe/dataframe_for_training.csv")

    return df_sans_nan

# Fonction permettant d'entraîner un réseaux de neurones
# Prend en entrée le DataFrame pour entraîner le Network
# Ne renvoie rien mais enregistre sous forme d'un fichier PTH le Network
def train_network(df_train):

    # Séparer les caractéristiques et la cible
    X = df_train.drop('Transported', axis=1)
    y = df_train['Transported']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Convertir en tenseurs PyTorch
    X_train_tensor = torch.tensor(X_train.values).float()
    y_train_tensor = torch.tensor(y_train.values).float().view(-1, 1)  # Remodeler pour correspondre aux dimensions attendues par PyTorch
    X_test_tensor = torch.tensor(X_test.values).float()
    y_test_tensor = torch.tensor(y_test.values).float().view(-1, 1)

    # Création du Network
    model = Net(X_train.shape[1])
    # On définit la fonction de perte
    # Ici, BCELoss est adapté pour une classification binaire
    criterion = nn.BCELoss()
    # On choisit le modèle d'optimisation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Boucle d'entraînement
    epochs = 100
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
    torch.save(model.state_dict(), f'neural_network/network_train/network_spaceship_{accuracy:.4f}.pth')

    # On affiche l'évolution de la fonction de perte pendant l'entraînement
    plt.plot(list_loss)
    plt.show()



data_traning = prepera_data_for_training()
train_network(data_traning)
make_prediction("neural_network/network_train/network_spaceship_0.7518.pth")