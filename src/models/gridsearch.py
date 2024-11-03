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

input_folderpath =  'data/processed/'
output_folderpath =  'data/processed/'
models_folderpath = 'models/'

# 1. Chargement des données
X_train_scaled = pd.read_csv(input_folderpath + "X_train_scaled.csv")
X_test_scaled = pd.read_csv(input_folderpath + "X_test_scaled.csv")
y_train = pd.read_csv(input_folderpath + "y_train.csv")
y_test = pd.read_csv(input_folderpath + "y_test.csv")
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# 2. Définir les modèles et leurs hyperparamètres
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regressor': SVR()
}

param_grids = {
    'Linear Regression': {},
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
    #    'min_samples_split': [2, 5, 10]
    },
    'Support Vector Regressor': {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.5]
    }
}

best_model = None
best_params = None
best_mse = float('inf')
best_model_name = ""  # Variable pour stocker le nom du meilleur modèle

# 3. Configurer Grid Search
for name, model in models.items():
    print (f"\n{name} - Démarrage analyse du modèle...")
    grid_search = GridSearchCV(model, param_grids[name], scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(X_train_scaled, y_train)
    
    # Évaluer le modèle avec les meilleures hyperparamètres sur les données de test
    best_model_for_this = grid_search.best_estimator_
    best_params_for_this = grid_search.best_params_
    predictions = best_model_for_this.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    
    print(f"{name} - Best Parameters: {grid_search.best_params_}, MSE: {mse:.4f}")

    # Comparer pour trouver le meilleur modèle global
    if mse < best_mse:
        best_mse = mse
        best_model = best_model_for_this
        best_params = best_params_for_this
        best_model_name = name  # Mettre à jour le nom du meilleur modèle

# 4. Sauvegarder le meilleur modèle global
if best_model is not None:
    joblib.dump(best_model, models_folderpath + 'best_params.pkl')
    print(f"\nBest model: {best_model_name} saved with MSE: {best_mse:.4f} - params={best_params}")

print("\nETAPE Sélection meilleur modèle réalisée avec succès\n")
