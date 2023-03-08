import pandas as pd
import numpy as np

class SVM:
    def __init__(self, C = 1.0, kernel = 'linear', gamma = 'auto'):
        self.C = C 
        self.kernel = kernel 
        self.gamma = gamma 
        self.alpha = None 
        self.b = None 

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        tol = 1e-5
        max_passes = 5
        passes = 0

        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(n_samples):
                Ei = self.predict(X[i])