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

# Task 1.2 Support Vector Machines
from SVM import *
clf = SVM(n_iters=1000)
clf.fit(X_entreno, y_entreno)
predictions = clf.predict(X_prueba)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred) / len(y_true)
    return accuracy

# Task 1.2 impresión resultados
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Sin librerías
print("SVM Accuracy: ", accuracy(y_prueba, predictions))
mat_conf = confusion_matrix(y_prueba, predictions)
print("\nMatriz de confusión: ")
print(mat_conf)
print("\nReporte de clasificación")
print(classification_report(y_prueba, predictions))
print()

# Con librerías
clasificador = SVC(kernel = 'rbf', random_state = 2)
clasificador.fit(X_entreno, y_entreno)
y_pred = clasificador.predict(X_prueba)
np.concatenate((y_pred.reshape(len(y_pred),1), y_prueba.reshape(len(y_prueba),1)),1)
mat_conf = confusion_matrix(y_prueba, y_pred)
print("\nMatriz de confusión: ")
print(mat_conf)
print("\nReporte de clasificación")
print(classification_report(y_prueba, y_pred))
print()
