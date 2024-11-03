import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib  # Pour la sauvegarde et le chargement des modèles
import os

print("joblib version = ",joblib.__version__)

input_folderpath =  'data/processed/'
models_folderpath = 'models/'

# Chargement des données
X_train_scaled = pd.read_csv(input_folderpath + "X_train_scaled.csv")
X_test_scaled = pd.read_csv(input_folderpath + "X_test_scaled.csv")
y_train = pd.read_csv(input_folderpath + "y_train.csv")
y_test = pd.read_csv(input_folderpath + "y_test.csv")
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Chargement du modèle
loaded_model = joblib.load(models_folderpath + 'best_params.pkl')
loaded_model.fit(X_train_scaled, y_train)

# Sauvegarde du modèle entrainé
joblib.dump(loaded_model, models_folderpath + 'trained_model.pkl')
    
print("\nETAPE Entrainement du modèle réalisée avec succès\n")
