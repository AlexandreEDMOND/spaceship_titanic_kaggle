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

def data_test(file_path, nom_fichier_test="data/test.csv"):

    df = pd.read_csv(nom_fichier_test)

    df = supprimer_colonne(df, ['Name'])

    df['Somme_depense'] = False
    df['Jeune'] = False
    df['Deck'] = 'A'
    df['Side_S'] = False
    df['Prediction'] = False

    # Définir les colonnes à vérifier pour les valeurs manquantes
    columns_to_check = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP']
    columns_to_complete = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    selected_column_fro_network = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Somme_depense', 'Jeune', 'Deck', 'Side_S']


    model = Net(19)
    model.load_state_dict(torch.load(file_path))
    model.eval()

    # Parcourir chaque ligne du DataFrame
    for index, row in df.iterrows():
        need_network_prediction = True
        for col in columns_to_check:
            if pd.isnull(row[col]): 
                #print(f"Valeur manquante trouvée à l'id {row['PassengerId']} dans la colonne '{col}'.")
                need_network_prediction = False
        if need_network_prediction:
            # On remplit les valeurs des colonnes vide par des 0
            for name_colomn in columns_to_complete:
                if pd.isnull(row[name_colomn]):
                    df.loc[index, name_colomn] = 0
            
            somme = 0
            for name_colomn in columns_to_complete:
                somme += row[name_colomn]
            
            if somme > 0:
                df.loc[index, 'Somme_depense'] = True
            
            if row['Age'] <= 15:
                df.loc[index, 'Jeune'] = True
            
            df.loc[index, 'Deck'] = row['Cabin'][0]

            if row['Cabin'][-1] == 'S':
                df.loc[index, 'Side_S'] = True
            
            row_selected = row[selected_column_fro_network]
            print(row_selected)
            row_selected = pd.get_dummies(row_selected, columns=['HomePlanet', 'Destination', 'Deck'])
            print(row_selected)
            X = torch.tensor(row_selected.values).float()
            predictions = model(X)
            predicted_classes = predictions.round()
            print(predicted_classes)
            
    
    df.to_csv("train_preclean.csv")
            


            

        

def prepare_data_test(file_path, nom_fichier_test="data/test.csv"):

    df = pd.read_csv(nom_fichier_test)
    print("Taille du dataset : ", df.shape)

    df = supprimer_colonne(df, ['Name'])
    df = creation_somme_depense(df)

    # Partie avec le dataset sans nan
    dataset_sans_nan = df.dropna()

    dataset_sans_nan = creation_jeune(dataset_sans_nan)
    dataset_sans_nan = creation_deck_et_side(dataset_sans_nan)

    dataset_sans_nan['CryoSleep'] = dataset_sans_nan['CryoSleep'].astype(bool)
    dataset_sans_nan['VIP'] = dataset_sans_nan['VIP'].astype(bool)

    dataset_sans_nan_2 = supprimer_colonne(dataset_sans_nan, ['PassengerId'])
    dataset_sans_nan_2 = pd.get_dummies(dataset_sans_nan_2[:])

    X = torch.tensor(dataset_sans_nan_2.values).float()

    model = Net(X.shape[1])
    model.load_state_dict(torch.load(file_path))
    model.eval()  # Mettez le modèle en mode évaluation

    # 4. Effectuer des prédictions
    with torch.no_grad():
        predictions = model(X)
        predicted_classes = predictions.round()

    # Afficher les prédictions
    print(predicted_classes.shape)
    print(dataset_sans_nan['PassengerId'])
    # Convertir le tenseur en DataFrame pandas
    tensor_df = pd.DataFrame(predicted_classes.numpy(), columns=['Transported'])
    passenger_id_series = dataset_sans_nan['PassengerId'].reset_index(drop=True)
    merged_df = pd.concat([passenger_id_series, tensor_df], axis=1)
    print(merged_df)

    # Partie avec le dataset avec nan
    dataset_avec_nan = df[df.isnull().any(axis=1)]
    dataset_avec_nan['Transported'] = True
    df_nan = dataset_avec_nan[['PassengerId', 'Transported']]
    print(df_nan)

    final_df = pd.concat([merged_df, df_nan], axis = 0)
    print(final_df)

    # Conversion des valeurs 0 et 1 en valeurs booléennes
    final_df['Transported'] = final_df['Transported'].astype(bool)

    final_df.to_csv("first_submission.csv", index=False)


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

    dataset_sans_nan.to_csv("train_clean.csv")

    return dataset_sans_nan

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

    model = Net(X_train.shape[1])
    criterion = nn.BCELoss()
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
    torch.save(model.state_dict(), f'network_train/network_spaceship_{accuracy:.4f}.pth')

    plt.plot(list_loss)
    plt.show()



#data_traing = prepera_data_training()
#train_network(data_traing)
# Dataset avec les lignes où il n'y a pas de valeurs manquantes
#prepare_data_test()

prepare_data_test("network_train/network_spaceship_0.7518.pth")
#data_test("network_train/network_spaceship_0.7518.pth")