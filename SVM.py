import pandas as pd
import numpy as np

# basado en: https://towardsdatascience.com/implementing-svm-from-scratch-784e4ad0bc6a

class SVM:
    def __init__(self, learning_rate=1e-3, param_lambda=1e-2, n_iters=1000):
        self.lr = learning_rate
        self.param_lambda = param_lambda
        self.n_iters = n_iters
        self.w = None # Se obtiene después
        self.b = None # Se obttiene después

    def _init_weights_bias(self, X):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

    def _get_cls_map(self, y):
        return np.where(y <= 0, -1, 1)

    def _satisfy_constraint(self, x, idx):
        modelo_lineal = np.dot(x, self.w) + self.b 
        return self.cls_map[idx] * modelo_lineal >= 1
    
    def _get_gradients(self, constrain, x, idx):
        if constrain:
            dw = self.param_lambda * self.w
            db = 0
            return dw, db
        
        dw = self.param_lambda * self.w - np.dot(self.cls_map[idx], x)
        db = - self.cls_map[idx]
        return dw, db
    
    def _update_weights_bias(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db
    
    # Método fit para que el modelo, dado un set de datos X y y, genere el modelo SVM
    def fit(self, X, y):
        self._init_weights_bias(X)
        self.cls_map = self._get_cls_map(y)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                constrain = self._satisfy_constraint(x, idx)
                dw, db = self._get_gradients(constrain, x, idx)
                self._update_weights_bias(dw, db)
    
    # Dado los valores, predecir 'y'
    def predict(self, X):
        estimacion = np.dot(X, self.w) + self.b
        prediccion = np.sign(estimacion)
        return np.where(prediccion == -1, 0, 1)
    