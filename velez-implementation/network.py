#!/usr/bin/env python3

import numpy as np


def phi(x):
    return 2/(1 + np.exp(-32*x)) - 1


def g(x, max_distance=1.5, sigma=0.5):
    if x <= max_distance:
        return np.exp(-2)/np.sqrt(2*sigma**2*np.pi)*np.exp(-x**2/(2*sigma**2))
    return 0


def polynomial_mutation(x, lower, upper, eta):
    """Adjusts the value of x in accordance with polynomial mutation.

    Args:
        x (float): The value to be mutated.
        lower (float): The minimum value of x.
        upper (float): The maximum value of x.
        eta (int): The distribution index.

    Returns:
        float: The mutated value.
    """
    d1 = (x-lower)/(upper-lower)
    d2 = (upper-x)/(upper-lower)
    r = np.random.rand()
    if r <= 0.5:
        dq = (2*r + (1-2*r)*(1-d1)**(eta+1))**(1/(eta+1))-1
    else:
        dq = 1 - (2*(1-r) + 2*(r-0.5)*(1-d2)**(eta+1))**(1/(eta+1))
    return x + dq*(upper-lower)


class Neuron():
    def __init__(self, layer, coords):
        self.possible_input_neurons = layer
        self.input_neurons = layer
        self.weights = [np.random.rand()*2-1 for _ in range(len(layer))]
        self.bias = np.random.rand()*2-1
        self.coords = coords

    def toggle_connection(self, neuron):
        if neuron in self.input_neurons:
            index = self.input_neurons.index(neuron)
            self.input_neurons.pop(index)
            self.weights.pop(index)
        else:
            self.input_neurons.append(neuron)
            self.weights.append(np.random.rand()*2-1)

    def activation(self):
        if len(self.input_neurons) == 0:
            return 0
        return phi(
            sum([w*a for w, a in zip(self.weights, self.input_neurons)])
            + self.bias)

    def update_weights(self, sources, eta=0.002):
        m = phi(sum([s.activation()*g(self.distance(s)) for s in sources]))
        for i in range(len(self.input_neurons)):
            self.weights[i] += eta * m * self.input_neurons[i].activation() \
                * self.activation()

    def distance(self, source):
        return np.sqrt(sum([
            (self.coords[i] - source.coords[i])**2
            for i in range(len(self.coords))]))


class InputNeuron(Neuron):
    def __init__(self, coords):
        self.value = 0
        self.coords = coords

    def set_value(self, value):
        self.value = value

    def activation(self):
        return self.value


class DiffusionSource(InputNeuron):
    def __init__(self, x, y):
        super().__init__(x, y)


class Network():
    def __init__(self, layer_config, source_config):
        """Creates a Network instance.
        Args:
            layer_config (list): A nested list of dimensionality
                [num_layers, num_nodes, 2]. The contents are interpreted as
                pairs of nD Cartesian coordinates for each neuron, grouped by
                layer.

                Example:
                [[(0,0), (1,0)], [(1,0), (1,1)]] gives rise to a network of
                two layers, each containing two neurons at the given
                two-dimensional coordinates.

            source_config (list): A list of diffusion source nD Cartesian
                coordinates.

                Example:
                [(0,0), (1,1)] gives rise to two diffusion sources at
                coordinates 0,0 and 1,1.
        """
        self.layers = []
        self.sources = []
        for l_c in layer_config:
            if len(self.layers) == 0:
                layer = [InputNeuron(n_c) for n_c in l_c]
            else:
                layer = [Neuron(self.layers[-1], n_c) for n_c in l_c]
            self.layers.append(layer)
        for s_c in source_config:
            self.sources.append(DiffusionSource(s_c))

    def get_values(self, inputs, nm_inputs):
        for i in range(len(self.layers[0])):
            self.layers[0][i].set_value(inputs[i])
        for i in range(len(self.sources)):
            self.sources[i].set_value(nm_inputs[i])
        return [n.activation() for n in self.layers[-1]]

    def update_weights(self):
        for layer in self.layers:
            for n in layer:
                n.update_weights(self.sources)

    def mutate(self, p_toggle=0.20, p_reassign=0.15,
               p_biaschange=0.10, p_weightchange=-1):
        """Clones the network and mutates the copy.

        Args:
            p_toggle: The probability of toggling a connection.
                Defaults to 0.20 (20%).
            p_reassign: The probability of reassigning a connection's source
                or target from one neuron to another.
                Defaults to 0.15 (15%).
            p_biaschange: The prbability of changing the bias of a neuron.
                Defaults to 0.10 (10%).
            p_weightchange: The probability of changing the weight of a
                connection.
                If -1, sets the probability to 2/n, n being the number of
                connections in the whole network.
                Defaults to -1.

        Returns:
            Network: The mutated network.
        """
        n = sum([
                sum([len(neu.weights) for neu in lay])
                for lay in self.layers[1:]])
        if p_weightchange == -1:
            p_weightchange = 2/n
        # clone the network
        new_network = Network([], [])
        for layer in self.layers:
            new_layer = []
            for neuron in layer:
                new_neuron = Neuron(neuron.possible_input_neurons,
                                    neuron.coords)
                new_neuron.bias = neuron.bias
                new_neuron.input_neurons = neuron.input_neurons
                new_neuron.weights = neuron.weights
                new_layer.append(new_neuron)
            new_network.layers.append(new_layer)
        for source in self.sources:
            new_source = DiffusionSource(source.coords)
            new_source.set_value(source.value)
            new_network.sources.append(new_source)
        # mutate the clone
        for layer in new_network.layers[1:]:
            for neuron in layer:
                for inp in neuron.possible_input_neurons:
                    if np.random.rand() < p_toggle:
                        neuron.toggle_connection(inp)
                for i in range(len(neuron.input_neurons)):
                    if np.random.rand() < p_reassign:
                        candidates = np.random.shuffle(
                            neuron.possible_input_neurons.copy())
                        c = 0
                        swapped = False
                        while not swapped:
                            if candidates[c] not in neuron.input_neurons:
                                neuron.input_neurons[i] = candidates[c]
                                swapped = True
                            c += 1
                            if c == len(candidates):
                                swapped = True
                for i in range(len(neuron.weights)):
                    if np.random.rand() < p_weightchange:
                        neuron.weights[i] = polynomial_mutation(
                            neuron.weights[i], -1, 1, 20)
                if np.random.rand() < p_biaschange:
                    neuron.bias = polynomial_mutation(neuron.bias, -1, 1, 20)
        return new_network
