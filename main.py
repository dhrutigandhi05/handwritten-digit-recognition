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
sample_labels = y_train[:5]
probabilities = model.forward(sample_images)
predictions = np.argmax(probabilities, axis=1)
loss = categorical_cross_entropy(sample_labels, probabilities)
model.backward(sample_images, sample_labels)

print("Predictions:", predictions)
print("Actual labels:", sample_labels)
print("Loss:", loss)
print("\nGradient shapes:")
print("dW1:", model.dW1.shape)
print("db1:", model.db1.shape)
print("dW2:", model.dW2.shape)
print("db2:", model.db2.shape)
print("\nSample gradients:")
print(model.dW2[:3, :3])