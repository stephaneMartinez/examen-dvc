
import sklearn
import pandas as pd 
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib, json
import numpy as np
data_folderpath = 'data/processed/'
model_filename = 'models/trained_model.pkl'
metrics_folderpath = 'metrics/'

print("joblib version = ",joblib.__version__)

X_test = pd.read_csv(data_folderpath + 'X_test_scaled.csv')
y_test = pd.read_csv(data_folderpath + 'y_test.csv')
y_test = np.ravel(y_test)

#--charge le modèle
loaded_model = joblib.load(model_filename)

#--Evaluate the model
y_pred = loaded_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
metrics_to_write = {
    "mse": mse,
    "mae": mae,
    "r2" : r2
}

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (r2):", r2)

# Sauvegarde des prédictions y_pred
pd.DataFrame(y_pred, columns=['target']).to_csv(data_folderpath + "predictions.csv")

# Écrire chaque résultat dans un fichier JSON 
with open(metrics_folderpath + "metrics.json", "w") as f:
    json.dump(metrics_to_write, f)

print("\nETAPE Evaluation du modèle réalisée avec succès.\n")
