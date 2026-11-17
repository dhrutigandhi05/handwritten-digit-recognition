import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
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

epochs = 10
batch_size = 64
learning_rate = 0.1
model = NeuralNetwork()
rng = np.random.default_rng(42)
loss_history = []
accuracy_history = []

for epoch in range(epochs):
    indices = rng.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        probabilities = model.forward(X_batch)
        model.backward(X_batch, y_batch)
        model.update_parameters(learning_rate)

    train_probs = model.forward(X_train)
    loss = categorical_cross_entropy(y_train, train_probs)
    predictions = np.argmax(train_probs, axis=1)
    accuracy = np.mean(predictions == y_train)
    loss_history.append(loss)
    accuracy_history.append(accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"Loss: {loss:.4f} - "
        f"Accuracy: {accuracy:.4f}"
    )

plt.plot(range(1, epochs + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

Path("models").mkdir(exist_ok=True)
model.save("models/mnist_model.npz")
print("model saved")