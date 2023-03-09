import numpy as np

# basado en: https://towardsdatascience.com/create-your-own-k-nearest-vecinos-algorithm-in-python-eb7093fc6339

def elegir_cat(lst):
    return max(set(lst), key=lst.count)

def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))
class KNN:
    def __init__(self, k = 5, metodo_distinción = euclidean):
        self.k = k # por default es 5
        self.metodo_distinción = metodo_distinción # Almacena el método completo para el caso en el que se desee utilizar
        # otro método distinto a el auclidaino

    def fit(self, X_entreno, y_entreno):
        # Simplemente almacena esta información, no es necesario hacer mucho
        # más sobre ella
        self.X_entreno = X_entreno
        self.y_entreno = y_entreno
        

    def predict(self, X_prueba):
        vecinos = [] 
        for x in X_prueba:
            distancias = self.metodo_distinción(x, self.X_entreno)
            y_sorted = [y for _, y in sorted(zip(distancias, self.y_entreno))]
            vecinos.append(y_sorted[:self.k])
        return list(map(elegir_cat, vecinos))

    
