import numpy as np
from sklearn.datasets import fetch_openml
from src.neural_network import NeuralNetwork
from src.losses import categorical_cross_entropy

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False,
    parser="liac-arff",
    data_home="./data"
)

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)
X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]
model = NeuralNetwork()
sample_images = X_train[:5]
probabilities = model.forward(sample_images)
predictions = np.argmax(probabilities, axis=1)
loss = categorical_cross_entropy(y_train[:5], probabilities)

print("output shape:", probabilities.shape)
print("predictions:", predictions)
print("actual labels:", y_train[:5])
print("probability sums:", probabilities.sum(axis=1))
print("Categorical cross-entropy loss:", loss)