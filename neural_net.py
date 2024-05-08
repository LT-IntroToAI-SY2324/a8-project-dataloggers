import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def d_tanh(y):
    return 1 - y * y

def make_random_array(rows, cols):
    return np.random.rand(rows, cols) - 0.5  # Centered around zero

def make_zero_array(rows, cols):
    return np.zeros((rows, cols))

class NeuralNet:
    """A simple implementation of a neural network with one hidden layer."""

    def __init__(self, n_input, n_hidden, n_output):
        self.num_input = n_input + 1  # Including bias node
        self.num_hidden = n_hidden + 1  # Including bias node
        self.num_output = n_output
        self.input_layer = np.ones(self.num_input)
        self.hidden_layer = np.ones(self.num_hidden)
        self.output_layer = np.ones(self.num_output)
        self.ih_weights = make_random_array(self.num_input, n_hidden)  # Fixed dimensions
        self.ho_weights = make_random_array(self.num_hidden, n_output)  # Fixed dimensions
        self.ih_weights_changes = make_zero_array(self.num_input, n_hidden)  # Fixed dimensions
        self.ho_weights_changes = make_zero_array(self.num_hidden, n_output)  # Fixed dimensions
        self.act_function_is_sigmoid = True
        self.act_function = sigmoid
        self.dact_function = d_sigmoid

    def compute_one_layer(self, inputs, weights, add_bias=True):
        # The bias is already included in inputs; no need to add again
        return self.act_function(np.dot(inputs, weights))

    def evaluate(self, inputs):
        """Evaluates the neural network on the given inputs."""
        if inputs.ndim == 1:  # Single instance
            inputs = inputs.reshape(1, -1)
        if inputs.shape[1] != self.num_input - 1:
            raise ValueError("Wrong number of inputs")

        outputs = []
        for input in inputs:
            self.input_layer[:-1] = input  # Set inputs, excluding bias
            self.hidden_layer[:-1] = self.compute_one_layer(self.input_layer, self.ih_weights)  # Compute hidden layer activations
            output = self.compute_one_layer(self.hidden_layer, self.ho_weights, False)  # Compute output layer activations
            outputs.append(output)

        return np.array(outputs)

    
    def back_propagate(self, inputs, expected, lr=0.1, momentum=0.1):
        self.evaluate(inputs)
        output_errors = expected - self.output_layer
        output_deltas = self.dact_function(self.output_layer) * output_errors

        hidden_errors = np.dot(output_deltas, self.ho_weights.T)
        # Make sure to exclude the bias unit from the hidden layer when calculating hidden_deltas
        hidden_deltas = self.dact_function(self.hidden_layer[:-1]) * hidden_errors[:-1]

        # Update output weights
        self.ho_weights += lr * np.outer(self.hidden_layer, output_deltas) + momentum * self.ho_weights_changes
        self.ho_weights_changes = lr * np.outer(self.hidden_layer, output_deltas)

        # Update input weights
        self.ih_weights += lr * np.outer(self.input_layer, hidden_deltas) + momentum * self.ih_weights_changes
        self.ih_weights_changes = lr * np.outer(self.input_layer, hidden_deltas)

        return np.mean(output_errors**2)


    def train(self, data, labels, epochs=100, lr=0.1, momentum=0.1):
        for epoch in range(epochs):
            for inputs, expected in zip(data, labels):
                self.back_propagate(inputs, expected, lr, momentum)
            if epoch % 10 == 0:
                loss = np.mean([self.back_propagate(inputs, expected, lr, momentum) for inputs, expected in zip(data, labels)])
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, inputs):
        outputs = self.evaluate(inputs)
        return outputs
    
    
