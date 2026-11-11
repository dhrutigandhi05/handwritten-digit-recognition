from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml(
    "mnist_784", 
    version=1, 
    as_frame=False, 
    parser="liac-arff",
    data_home="./data"
)

print("data downloaded")
# print("data shape:", mnist.data.shape)
# print("target shape:", mnist.target.shape)

X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)
X = X / 255.0

print("X shape:", X.shape)
print("y shape:", y.shape)
print("pixel value range: ", X.min(), "-", X.max())
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    image = X[i].reshape(28, 28)
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

print("training images:", X_train.shape)
print("training labels:", y_train.shape)
print("testing images:", X_test.shape)
print("testing labels:", y_test.shape)