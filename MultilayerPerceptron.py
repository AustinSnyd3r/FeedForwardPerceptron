# Define class Perceptron
import numpy as np

import util

# On the forward pass, we dont update weights, just make a prediction with the current,
# and this value will serve as an input to the next
class Perceptron:
    def __init__(self, input_size, learning_rate=1.0, a_function=1):
        self.weights = np.random.rand(input_size).tolist()
        self.learning_rate = learning_rate
        self.a_function = a_function

        if a_function != 1 and a_function != -1 and a_function != 0:
            raise ValueError("Invalid activation function: Bipolar = -1, Unipolar = 1, Linear = 0")

    def activation_function(self, x):
        if self.a_function == 1:
            return util.unipolar_sigmoid(x)
        elif self.a_function == -1:
            return util.bipolar_sigmoid(x)
        else:
            return util.linear_activation(x)

    def activation_derivative(self, x):
        if self.a_function == 1:
            return util.unipolar_sigmoid_derivative(x)
        elif self.a_function == -1:
            return util.bipolar_sigmoid_derivative(x)
        else:
            return util.linear_activation_derivative(x)

    # We will use this on the forward pass, this will predict with existing weights and inputs
    # then we will pass this on to the next layer.
    def predict(self, input_vector):
        x = input_vector
        w = self.weights

        # Dot product of inputs and weights
        dot_product = np.dot(x, w)

        # Activation function: Bipolar = -1, Unipolar = 1, Linear = 0
        if self.a_function == 1:
            return util.unipolar_sigmoid(dot_product)
        elif self.a_function == -1:
            return util.unipolar_sigmoid(dot_product)
        else:
            return util.linear_activation(dot_product)

