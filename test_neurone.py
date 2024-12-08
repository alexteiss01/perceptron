import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
from neurone import perceptron

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

# Création et entraînement du perceptron
perceptron = Perceptron(learning_rate=0.1, n_iter=100)
perceptron.fit(X, y)

# Prédiction sur les données
y_pred = perceptron.predict(X)

# Affichage de l'exactitude sur l'ensemble de données
accuracy = perceptron.accuracy(X, y)
print(f"Exactitude sur les données générées : {accuracy * 100:.2f}%")