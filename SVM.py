import pandas as pd
import numpy as np

# class SVM:
#     def __init__(self, C = 1.0, kernel = 'linear', gamma = 'auto'):
#         self.C = C 
#         self.kernel = kernel 
#         self.gamma = gamma 
#         self.alpha = None 
#         self.b = None 

#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.alpha = np.zeros(n_samples)
#         self.b = 0.0
#         tol = 1e-5
#         max_passes = 5
#         passes = 0

#         while passes < max_passes:
#             num_changed_alphas = 0
#             for i in range(n_samples):
#                 Ei = self.predict(X[i])
#                 if (y[i] * Ei < -tol and self.alpha[i] < self.C) or (y[i] * Ei > tol and self.alpha[i] > 0):
#                     j = np.random.choice(list(range(i)) + list(range(i+1, n_samples)))
#                     Ej = self.predict()


#     def predict(self, X):
#         pass 

import numpy as np

class SVM:
    def __init__(self, kernel='linear', C=1.0, gamma=None, degree=3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.alpha = None
        self.support_vectors = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize alpha and b vectors
        self.alpha = np.zeros(n_samples)
        self.intercept = 0.0

        # Compute the kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel_function(X[i], X[j])

        # Train the SVM model
        for epoch in range(100):
            for i in range(n_samples):
                E_i = np.sum(self.alpha * y * K[:,i]) + self.intercept - y[i]
                if ((y[i]*E_i < -1*self.gamma and self.alpha[i] < self.C) or (y[i]*E_i > self.gamma and self.alpha[i] > 0)):
                    j = np.random.choice([x for x in range(n_samples) if x != i])
                    E_j = np.sum(self.alpha * y * K[:,j]) + self.intercept - y[j]
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                    if y[i] == y[j]:
                        L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                        H = min(self.C, self.alpha[j] + self.alpha[i])
                    else:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    if L == H:
                        continue
                    eta = 2 * K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0:
                        continue
                    self.alpha[j] -= (y[j] * (E_i - E_j)) / eta
                    self.alpha[j] = max(self.alpha[j], L)
                    self.alpha[j] = min(self.alpha[j], H)
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    b1 = self.intercept - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i,i] - y[j] * (self.alpha[j] - alpha_j_old) * K[i,j]
                    b2 = self.intercept - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i,j] - y[j] * (self.alpha[j] - alpha_j_old) * K[j,j]
                    if 0 < self.alpha[i] and self.alpha[i] < self.C:
                        self.intercept = b1
                    elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                        self.intercept = b2
                    else:
                        self.intercept = (b1 + b2) / 2

        # Save support vectors and their corresponding labels
        idx = self.alpha > 1e-5
        self.alpha = self.alpha[idx]
        self.support_vectors = X[idx]
        # self.support

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i, sample in enumerate(X):
            prediction = self.intercept
            for alpha, sv, label in zip(self.alpha, self.support_vectors, y):
                prediction += alpha * label * self.kernel_function(sample, sv)
            y_pred[i] = np.sign(prediction)
        return y_pred
