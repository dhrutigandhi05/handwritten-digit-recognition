import numpy as np
from src.activations import relu, softmax, relu_derivative

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=32, output_size=10):
        randomNumber = np.random.default_rng(42)

        #input -> hidden layer
        self.W1 = randomNumber.standard_normal((input_size, hidden_size)) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        #hidden -> output layer
        self.W2 = randomNumber.standard_normal((hidden_size, output_size)) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        #hidden layer
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        #output layer
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2
    
    def backward(self, X, y):
        m = X.shape[0]
        Y = np.eye(self.W2.shape[1])[y]
        dZ2 = (self.A2 - Y) / m
        self.dW2 = self.A1.T @ dZ2
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        self.dW1 = X.T @ dZ1
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def update_parameters(self, learning_rate):
        # update hidden layer
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

        # update output layer
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
    
    def print_shapes(self):
        print("W1 shape:", self.W1.shape)
        print("b1 shape:", self.b1.shape)
        print("W2 shape:", self.W2.shape)
        print("b2 shape:", self.b2.shape)