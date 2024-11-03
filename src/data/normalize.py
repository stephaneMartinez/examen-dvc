import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

input_folderpath =  'data/processed/'
output_folderpath =  'data/processed/'

# Chargement des données
X_train = pd.read_csv(input_folderpath + "X_train.csv")
X_test = pd.read_csv(input_folderpath + "X_test.csv")

# Créer un objet StandardScaler
scaler = StandardScaler()

# Sélectionner uniquement les colonnes numériques (exclure 'date')
numeric_columns = X_test.select_dtypes(include=['float64', 'int']).columns

# Normaliser les colonnes numériques
X_train_scaled = pd.DataFrame (scaler.fit_transform(X_train[numeric_columns]), columns=numeric_columns)
X_test_scaled = pd.DataFrame (scaler.transform(X_test[numeric_columns]), columns=numeric_columns)

# Saving the dataframes to their respective output file paths
for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
    output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
    file.to_csv(output_filepath, index=False)

print("\nETAPE Normalisation réalisée avec succès\n")
