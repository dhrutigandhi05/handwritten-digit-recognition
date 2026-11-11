import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=32, output_size=10):
        randomNumber = np.random.default_rng(42)

        #input -> hidden layer
        self.W1 = randomNumber.standard_normal((input_size, hidden_size)) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        #hidden -> output layer
        self.W2 = randomNumber.standard_normal((hidden_size, output_size)) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def print_shapes(self):
        print("W1 shape:", self.W1.shape)
        print("b1 shape:", self.b1.shape)
        print("W2 shape:", self.W2.shape)
        print("b2 shape:", self.b2.shape)