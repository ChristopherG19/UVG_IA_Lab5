import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization
from pandas_profiling import ProfileReport
from quickda.clean_data import *

# Leer archivo csv
df = pd.read_csv('./dataset_phishing.csv')

#Se realiza la codificación de la columna status
data_encoded = pd.get_dummies(df, columns=['status'], prefix='', prefix_sep='')
df = data_encoded

df["legitimate"] = df["legitimate"].astype('int64')
df["phishing"] = df["phishing"].astype('int64')
to_categoric = ["url"]
df = clean(df, method = 'dtypes', columns = to_categoric, dtype='category')

# eliminar la columna legitamate para no tener información repetida
df = df.drop('legitimate', axis = 1)

# Obtención de valores
X = df.iloc[:, 2:-1].values
y = df.iloc[:, -1].values

print('x')
print(X)
print('y')
print(y)

## Separación de datos de entrenamieno y prueba
from sklearn.model_selection import train_test_split
X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size = 0.2, random_state = 100)

# balanceo de datos

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred) / len(y_true)
    return accuracy

# Task 1.1
from KNN import *

ks = range(1, 10)
accs = [] # Todas las accuracies
best = (0, 0) # El k con mejor accuracie
for k in ks:
    if k % 2 != 0:
        # de esta manera solo obtenemos los números impares
        knn = KNN(k=k)
        knn.fit(X_entreno, y_entreno)
        y_pred = knn.predict(X_prueba)
        acc = accuracy(y_prueba, y_pred)
        accs.append(acc)
        if (acc > best[1]):
            best = (k, acc)

# Task 1.1 impresión resultados
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# obtener mejores accuracies sin librerias
print('KNN Accuracy y mejor knn: ', best)
print("\nMatrix de confusión: ")
print(confusion_matrix(y_prueba, y_pred))
print(classification_report(y_prueba, y_pred))
print()

# Con librerías
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_entreno, y_entreno)
pred = knn.predict(X_prueba)
print("Matrix de confusión: ")
print(confusion_matrix(y_prueba, pred))
print(classification_report(y_prueba, pred))
