# Define class Perceptron
import numpy as np

import util

# On the forward pass, we dont update weights, just make a prediction with the current,
# and this value will serve as an input to the next
class Perceptron:
    def __init__(self, weights, a_function='sigmoid'):
        self.weights = np.array(weights)
        # Random bias
        self.bias = np.random.rand()
        self.a_function = a_function

        # Store activation and pre_activation for backprop
        self.activation = None
        self.pre_activation = None
        self.gradients = []

        if a_function not in ['sigmoid', 'bipolar_sigmoid', 'linear']:
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

        x = np.append(input_vector, 1)  # Add bias term to input
        w = np.append(self.weights, self.bias)

        # Dot product of inputs and weights
        dot_product = np.dot(x, w)

        # Store the weighted sum (z^L)
        self.pre_activation = dot_product

        return self.activation_function(dot_product)

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
            weights = np.random.rand(layer_sizes[i-1])

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

    """
     We want to go through and see how much each weight from a neuron in the n-1 layer
     played into the error we have. We will then propogate this error backwards through the network
     and continuely "blame" the nodes that caused the error at each point.
        
     We use a partial derivative of cost function with respect to each weight to find which way
     a weight move the cost function when changed (the gradient). We will use the -gradient to adjust
     the weights after backpropogation. 
     
    """
    def backprop(self, inputs, target_map):
        # clear the gradients from the previous pass
        for layer in self.layers:
            for node in layer:
                node.gradients.clear()

        # we will go backwards through our network
        for i in range(len(self.layers) - 1, -1, -1):
            curr_layer = self.layers[i]

            # If we are at the output layer
            if i == len(self.layers) - 1:
                prediction = curr_layer[0].activation
                error = prediction - target_map[inputs]

                # Calculate gradients for the output layer
                for j in range(len(curr_layer[0].weights)):
                    # case for finding gradient of weights
                    if j < len(curr_layer[0].weights) - 1:
                        gradient = self.activations[i - 1][j] * error
                    else:
                        gradient = error

                    # add the gradients for the current node
                    curr_layer[0].gradients.append(gradient)
            else:
                # go through each node in the layer.
                for j, curr_node in enumerate(curr_layer):

                    # we find propagated error by SUM(forward_weight_i * gradient_i)
                    propagated_error = 0
                    for k, next_node in enumerate(self.layers[i + 1]):
                        propagated_error += next_node.weights[j] * next_node.gradients[j]

                    # derivative of the activation(z^L)
                    derivative = curr_node.activation_derivative(curr_node.pre_activation)
                    propagated_error *= derivative

                    # Gradients for weights and bias
                    if i == 0:
                        # Need this because input layer is just numbers, not perceptrons
                        for k in range(len(curr_node.weights)):
                            if k < len(curr_node.weights) - 1:
                                gradient = inputs[k] * propagated_error
                            else:
                                gradient = propagated_error
                            curr_node.gradients.append(gradient)
                    else:
                        # Go through each weight
                        for k in range(len(curr_node.weights)):
                            # case for weights
                            if k < len(curr_node.weights) - 1:
                                # A^L-1 * propagated_error = gradient
                                gradient = self.layers[i - 1][k].activation * propagated_error
                            else:
                                # case for the gradient
                                gradient = propagated_error
                            curr_node.gradients.append(gradient)

    def update_weights(self):
        for layer in self.layers:
            for node in layer:
                for j in range(len(node.weights)):
                    # Update the weights and biases based on LR and gradients
                    if j < len(node.weights) - 1:
                        node.weights[j] -= self.learning_rate * node.gradients[j]
                    else:
                        node.bias -= self.learning_rate * node.gradients[j]


p = MultilayerPerceptron(4, [2, 4, 4, 1], ["sigmoid", "sigmoid", "linear"], 0.1)
p.forward([5, 2])

map = {(5, 2): 7.6}

for i in range(9000):
    p.forward([5, 2])

    p.backprop((5, 2), map)

    p.update_weights()
    print(p.layers[-1][0].activation)

# IT IS WOKRINGINIGNIGNG
print(p.layers[-1][0].activation)

