import numpy as np
from scipy.stats import mode

def most_common(lst):
    return max(set(lst), key=lst.count)

def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))
class KNN:
    def __init__(self, k = 5, metodo_dis = euclidean):
        self.k = k
        self.metodo_dis = metodo_dis

    def fit(self, X_entreno, y_entreno):
        self.X_entreno = X_entreno
        self.y_entreno = y_entreno
        

    def predict(self, X_prueba):
        neighbors = []
        for x in X_prueba:
            distances = self.metodo_dis(x, self.X_entreno)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_entreno))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))

    
