from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
print("data downloaded")
print("data shape:", mnist.data.shape)
print("target shape:", mnist.target.shape)