#!/usr/bin/env python3

import numpy as np


@np.vectorize
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


class Network():
    """
    TODO:
    - Document this
    - Refactor to use arrays instead of classes
        - null connection as weight=0
        - swap connections by swapping weights
        - weight array: [layers, nodes, weights]
        - bias array: [layers, nodes]
        - coordinate array: layer_config
        - source array: [sources]
        - source coord array: source_config
    - Add functionality for mutating coordinates
    """
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
        self.node_coords = layer_config
        self.source_coords = source_config
        self.weights = []
        self.biases = []
        self.activations = [np.zeros((len(layer),)) for layer in layer_config]
        self.source_inputs = np.zeros((len(source_config),))
        for layer in layer_config[1:]:
            units = self.activations[0] if len(self.weights) == 0 else self.weights[-1]
            self.weights.append(np.random.rand(len(layer), len(units))*2-1)
            self.biases.append(np.random.rand(len(layer))*2-1)

    def forward(self, inputs):
        if inputs is not np.ndarray:
            inputs = np.array(inputs)
        self.activations[0] = inputs
        for i in range(len(self.weights)):
            self.activations[i+1] = phi(
                self.weights[i] @ self.activations[i] + self.biases[i])
        return self.activations[-1]

    def update_weights(self, nm_inputs, eta=0.002):
        if nm_inputs is not np.ndarray:
            nm_inputs = np.array(nm_inputs)
        self.source_inputs = nm_inputs
        for i in range(len(self.weights)):
            M = np.zeros((len(self.weights[i]),))
            for j in range(len(self.weights[i])):
                distances = np.zeros((len(self.source_coords),))
                coords = np.array(self.node_coords[i+1][j])
                for k in range(len(self.source_coords)):
                    source_arr = np.array(self.source_coords[k])
                    distances[k] = g(np.linalg.norm(source_arr-coords))
                M[j] = phi(self.source_inputs @ distances)
            M = M.reshape((1, len(M)))
            self.weights[i] += np.outer(self.activations[i+1],
                                        self.activations[i]) * M.T * eta

    def reset_weights(self):
        for i in range(len(self.weights)):
            randoms = np.random.rand(*self.weights[i].shape)
            randoms[self.weights[i] == 0] = 0
            self.weights[i] = randoms

    def connection_cost(self):
        return 1 - sum([
            np.count_nonzero(w) for w in self.weights
        ])/sum([w.size for w in self.weights])

    def copy(self):
        clone = Network(self.node_coords, self.source_coords)
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = self.biases.copy()
        clone.activations = [a.copy() for a in self.activations]
        clone.source_inputs = self.source_inputs.copy()
        return clone

    def mutate(self, p_toggle=0.20, p_reassign=0.15,
               p_biaschange=0.10, p_weightchange=-1):
        """Mutates the network.

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
            None.
        """
        n = sum([np.count_nonzero(weights) for weights in self.weights])
        if p_weightchange == -1:
            p_weightchange = 2/n
        for layer in self.weights:
            i = np.random.choice(range(len(layer)), 1)
            j = np.random.choice(range(len(layer[i])), 1)
            if np.random.rand() < p_toggle:
                layer[i, j] = 0 if layer[i, j] != 0 else np.random.rand()*2-1
            i = np.random.choice(range(len(layer)), 1)
            j = np.random.choice(range(len(layer[i])), 1)
            if np.random.rand() < p_weightchange and layer[i, j] != 0:
                layer[i, j] = polynomial_mutation(layer[i, j], -1, 1, 20)
            i = np.random.choice(range(len(layer)), 1)
            if np.random.rand() < p_reassign:
                indices = np.random.choice(
                    range(len(layer[i].reshape(layer[i].shape[1]))), 2, replace=False)
                layer[i, [indices[0], indices[1]]] \
                    = layer[i, [indices[1], indices[0]]]
        for layer in self.biases:
            i = np.random.choice(range(len(layer)), 1)
            if np.random.rand() < p_biaschange:
                layer[i] = polynomial_mutation(layer[i], -1, 1, 20)
