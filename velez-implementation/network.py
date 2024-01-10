#!/usr/bin/env python3

import numpy as np
import numba
import math


@numba.vectorize(['f8(f8)', 'f4(f4)'])
def phi(x):
    return 2/(1 + np.exp(-32*x)) - 1


@numba.njit('f8(f8)', nogil=True, fastmath=True)
def g(x):
    sigma = 0.5
    if x > 1.5:
        return 0
    return math.exp(-2)/math.sqrt(
        2*sigma**2*math.pi)*math.exp(
            -x**2/(2*sigma**2))


@numba.njit('float64(float64, float64, float64, int64)', nogil=True)
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


@numba.njit(
    "f8[:,::1](f8[::1],f8[:,::1],f8[:,::1],f8[:,::1],f8[::1],f8[::1],f8)",
    nogil=True)
def _update_weights(nm_inputs, weights, next_node_coords, source_coords,
                    activations, next_activations, eta):
    M = np.zeros((1, len(weights)))
    distances = np.zeros((len(source_coords),))
    for j in range(len(weights)):
        for k in range(len(source_coords)):
            predist = source_coords[k]-next_node_coords[j]
            predistsum = 0
            for l in predist:
                predistsum += l**2
            distances[k] = g(math.sqrt(predistsum))
        M[0, j] = phi(nm_inputs @ distances)
    weights += np.outer(next_activations, activations) * M.T * eta
    weights.clip(-1, 1, out=weights)
    return weights


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
        self.source_coords = np.array(source_config).astype(np.float64)
        self.weights = []
        self.biases = []
        self.activations = [np.zeros((len(layer),)) for layer in layer_config]
        for layer in layer_config[1:]:
            if len(self.weights) == 0:
                size = len(self.activations[0])
            else:
                size = self.weights[-1].shape[0]
            self.weights.append(np.random.rand(len(layer), size)*2-1)
            self.biases.append(np.random.rand(len(layer))*2-1)

    def forward(self, inputs):
        self.activations[0] = inputs
        for i in range(len(self.weights)):
            self.activations[i+1] = phi(
                self.weights[i] @ self.activations[i] + self.biases[i])
        return self.activations[-1]

    def update_weights(self, nm_inputs, eta=0.002):
        for i in range(len(self.weights)):
            self.weights[i] = _update_weights(nm_inputs,
                                              self.weights[i],
                                              self.node_coords[i+1],
                                              self.source_coords,
                                              self.activations[i],
                                              self.activations[i+1],
                                              eta)

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
        return clone

    def mutate(self, p_toggle=0.20, p_reassign=0.15,
               p_biaschange=0.10, p_weightchange=-1,
               p_nudge=0.00):
        """Mutates the network.

        Args:
            p_toggle (float): The probability of toggling a connection.
                Defaults to 0.20 (20%).
            p_reassign (float): The probability of reassigning a connection's
                source or target from one neuron to another.
                Defaults to 0.15 (15%).
            p_biaschange (float): The prbability of changing the bias of a
                neuron.
                Defaults to 0.10 (10%).
            p_weightchange (float): The probability of changing the weight of a
                connection.
                If -1, sets the probability to 2/n, n being the number of
                connections in the whole network.
                Defaults to -1.
            p_nudge (float): The probablility of adjusting the position of a
                neuron.
                Defaults to 0.00 (0%).

        Returns:
            None.
        """
        n = sum([np.count_nonzero(weights) for weights in self.weights])
        if p_weightchange == -1:
            p_weightchange = 2/n
        layer = self.weights[
            np.random.choice(range(len(self.weights)))]
        i = np.random.choice(range(len(layer)), 1)
        j = np.random.choice(range(len(layer[i])), 1)
        if np.random.rand() < p_toggle:
            if layer[i, j] != 0:
                layer[i, j] = 0
            else:
                layer[i, j] = np.random.rand()*2-1
        for layer in self.weights:
            for i in range(len(layer)):
                for j in range(len(layer[i])):
                    if np.random.rand() < p_weightchange \
                       and layer[i, j] != 0:
                        layer[i, j] = polynomial_mutation(
                            layer[i, j], -1, 1, 20)
                    if np.random.rand() < p_reassign:
                        k = np.random.choice(range(len(layer[i])))
                        layer[i, [j, k]] = layer[i, [k, j]]
        for layer in self.node_coords:
            for i in range(len(layer)):
                if np.random.rand() < p_nudge:
                    layer[i] = [x + polynomial_mutation(0, -1, 1, 20)
                                for x in layer[i]]
        for layer in self.biases:
            for i in range(len(layer)):
                if np.random.rand() < p_biaschange:
                    layer[i] = polynomial_mutation(layer[i], -1, 1, 20)
