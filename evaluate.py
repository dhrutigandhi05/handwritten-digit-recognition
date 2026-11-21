import numpy as np
from sklearn.datasets import fetch_openml
from src.neural_network import NeuralNetwork
from src.losses import categorical_cross_entropy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
model.load("models/mnist_model.npz")
train_probs = model.forward(X_train)
train_predictions = np.argmax(train_probs, axis=1)
train_accuracy = np.mean(train_predictions == y_train)

test_probs = model.forward(X_test)
test_predictions = np.argmax(test_probs, axis=1)
test_accuracy = np.mean(test_predictions == y_test)
test_loss = categorical_cross_entropy(y_test, test_probs)

print(f"training accuracy: {train_accuracy:.4f}")
print(f"test accuracy: {test_accuracy:.4f}")
print(f"test loss: {test_loss:.4f}")


cm = confusion_matrix(y_test, test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap="Blues")
plt.title("MNIST Confusion Matrix")
plt.tight_layout()
plt.show()

incorrect_indices = np.where(test_predictions != y_test)[0]
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    if i >= len(incorrect_indices):
        ax.axis("off")
        continue

    index = incorrect_indices[i]
    image = X_test[index].reshape(28, 28)

    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"Actual: {y_test[index]}\n"
        f"Predicted: {test_predictions[index]}"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()

accuracy_gap = train_accuracy - test_accuracy
print(f"Accuracy Gap: {accuracy_gap * 100:.2f} percentage points")