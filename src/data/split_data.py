import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from check_structure import check_existing_file, check_existing_folder
import os

input_folderpath =  './data/raw_data/'
output_folderpath =  './data/processed/'
target_column = 'silica_concentrate'

# Chargement des données
df = pd.read_csv(input_folderpath+"raw.csv", sep=',')

# Convertie les colonnes en numérique
for col in df.columns:
    if col != 'date':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Supprime la date jugée inutile pour la suite         
df.drop(['date'], axis=1)

# Features | target
features = df.drop([target_column], axis = 1)
target = df[target_column]

# Split des data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Create folder if necessary 
if check_existing_folder(output_folderpath) :
    os.makedirs(output_folderpath)

# Saving the dataframes to their respective output file paths
for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
    output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
    file.to_csv(output_filepath, index=False)

print("\nETAPE Split du jeu de données réalisé avec succès\n")