# Define class Perceptron
import numpy as np

import util

# On the forward pass, we dont update weights, just make a prediction with the current,
# and this value will serve as an input to the next
class Perceptron:
    def __init__(self, weights, a_function='sigmoid'):
        self.weights = np.array(weights)
        self.a_function = a_function

        # Store activation and pre_activation for backprop
        self.activation = None
        self.pre_activation = None
        self.gradients = []

        if a_function != 'sigmoid' and a_function != 'bipolar_sigmoid' and a_function != 'linear':
            raise ValueError("Invalid activation function.")

    def activation_function(self, x):
        if self.a_function == 'sigmoid':
            return util.unipolar_sigmoid(x)
        elif self.a_function == 'bipolar_sigmoid':
            return util.bipolar_sigmoid(x)
        else:
            return util.linear_activation(x)

    def activation_derivative(self, x):
        if self.a_function == 'sigmoid':
            return util.unipolar_sigmoid_derivative(x)
        elif self.a_function == 'bipolar_sigmoid':
            return util.bipolar_sigmoid_derivative(x)
        else:
            return util.linear_activation_derivative(x)

    # We will use this on the forward pass, this will predict with existing weights and inputs
    # then we will pass this on to the next layer.
    def predict(self, input_vector):
        x = np.append(input_vector, -1)  # Add bias term to input
        w = self.weights

        # Dot product of inputs and weights
        dot_product = np.dot(x, w)

        # Store the weighted sum (z^L)
        self.pre_activation = dot_product

        # Activation function: Bipolar = -1, Unipolar = 1, Linear = 0
        if self.a_function == 1:
            return util.unipolar_sigmoid(dot_product)
        elif self.a_function == -1:
            return util.unipolar_sigmoid(dot_product)
        else:
            return util.linear_activation(dot_product)

class MultilayerPerceptron:
    def __init__(self, num_layers, layer_sizes, activations, learning_rate):
        self.learning_rate = learning_rate
        self.num_hidden_layers = num_layers
        self.layer_size = layer_sizes
        self.activations = None
        # We need to make the layers

        # List( List(p1, ..., pn), List(p1, ..., pn))

        # We assume we have this [2, x, x, 1]
        # make a list that will hold each of our layers
        self.layers = []
        for i in range(1, len(layer_sizes)):
            weights = np.random.rand(layer_sizes[i-1] + 1)

            layer = [
                Perceptron(weights=weights, a_function=activations[i-1])
                for _ in range(layer_sizes[i])
            ]

            self.layers.append(layer)

    def forward(self, inputs):
        # Start the inputs as the x, y
        layer_inputs = inputs
        activations = []

        # Go through each layer, calculate the activation of each perceptron
        for layer in self.layers:
            layer_output = []
            for perceptron in layer:
                perceptron.activation = perceptron.predict(layer_inputs)
                # Build up the outputs of this layer
                layer_output.append(perceptron.activation)
            print("Layer output: ", layer_output)
            # Make the inputs of next layer, the outputs of this layer
            layer_inputs = layer_output
            activations.append(layer_output)

        # Save the activations of each node for back-propogation
        self.activations = activations
    def backprop(self, inputs, target_map):
        for layer in self.layers:
            for node in layer:
                node.gradients.clear()

        for i in range(len(self.layers) - 1, -1, -1):
            # Get the current layer we are on
            curr_layer = self.layers[i]

            # If we are at the output layer
            if i == len(self.layers) - 1:
                # The activation of the output node is the prediction
                prediction = self.layers[i][0].activation
                # MSE' = y actual - y predicted
                error = target_map[inputs] - prediction

                # Go through each of the weights connecting to the output layer
                # find the gradient for each one of these
                for j in range(len(self.layers[i][0].weights) - 1):
                    print("In J loop", j)
                    gradient = self.layers[i-1][j].activation * error
                    self.layers[i][0].gradients.append(gradient)
            else:
                # Go through each node in this layer => curr_node
                for j, curr_node in enumerate(self.layers[i]):
                    # calculate the propogated error for the layer
                    propogated_error = 0

                    for k, next_node in enumerate(self.layers[i+1]):
                        propogated_error += next_node.weights[j] * next_node.gradients[j]

                    derivative = curr_node.activation_derivative(curr_node.pre_activation)
                    propogated_error *= derivative
                    # go through each weight connecting to curr_node from n-1 layer
                    # find the gradient of each of these weights
                    if i == 0:
                        for k in range(len(curr_node.weights) - 1):
                            gradient = inputs[k] * propogated_error
                            curr_node.gradients.append(gradient)
                    else:
                        for k in range(len(curr_node.weights) - 1):
                            gradient = self.layers[i - 1][k].activation * propogated_error
                            curr_node.gradients.append(gradient)
    def update_weights(self):
        # Go through each node in th
        for layer in self.layers:
            for node in layer:
                for j in range(len(node.weights) - 1):
                    node.weights[j] -= self.learning_rate * node.gradients[j]


p = MultilayerPerceptron(4, [2, 4, 4, 1], ["sigmoid", "sigmoid", "linear"], 0.5)
p.forward([5, 2])

map = {(5, 2): 7.6}

for i in range(5):
    p.forward([5, 2])

    p.backprop((5, 2), map)

    p.update_weights()


print(p.layers[-1][0].activation)

