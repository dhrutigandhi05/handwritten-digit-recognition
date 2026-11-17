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
X_batch = X_train[:64]
y_batch = y_train[:64]
predictions = model.forward(X_batch)
loss_before = categorical_cross_entropy(y_batch, predictions)
model.backward(X_batch, y_batch)
model.update_parameters(learning_rate=0.1)
new_predictions = model.forward(X_batch)
loss_after = categorical_cross_entropy(y_batch, new_predictions)

print("Loss before:", loss_before)
print("Loss after:", loss_after)