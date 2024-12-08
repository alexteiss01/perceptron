import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs


class Perceptron:
    def __init__(self, learning_rate=0.1, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.W = None
        self.b = None
        self.losses = []

    def initialize_weights(self, n_features):
        self.W = np.random.randn(n_features, 1)
        self.b = np.random.randn(1)

    def model(self, X):
        Z = X.dot(self.W) + self.b
        A = 1 / (1 + np.exp(-Z))
        return A

    def log_loss(self, A, y):
        return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

    def gradients(self, A, X, y):
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return dW, db

    def update_weights(self, dW, db):
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def fit(self, X, y):
        n_features = X.shape[1]
        self.initialize_weights(n_features)

        for i in range(self.n_iter):
            A = self.model(X)
            self.losses.append(self.log_loss(A, y))
            dW, db = self.gradients(A, X, y)
            self.update_weights(dW, db)

        plt.plot(self.losses)
        plt.title("Loss over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Log Loss")
        plt.show()

    def predict(self, X):
        A = self.model(X)
        return (A >= 0.5).astype(int)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

