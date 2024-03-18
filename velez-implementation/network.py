#!/usr/bin/env python3

import math

import numba
import numpy as np

import utils


@numba.vectorize(['f8(f8)', 'f4(f4)'])
def phi(x):
    return 2/(1 + np.exp(-30*x)) - 1


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
    r = np.random.rand()
    if r < 0.5:
        dq = (2*r)**(1/(eta+1)) - 1
    else:
        dq = 1 - (2*(1-r))**(1/(eta+1))
    out = x + dq
    if out < lower:
        out = lower
    elif out > upper:
        out = upper
    return out


@numba.njit('f8[:,::1](f8[:,::1], f8, f8)', nogil=True)
def _mutate(layer, p_weightchange, p_reassign):
    """Internal function that performs weight alteration and reassignment.

    Separated for Numba acceleration purposes.

    Args:
        layer (np.array): The weight layer to mutate.
        p_weightchange (float): The probability of changing the weight of a
            connection.
        p_reassign (float): The probability of reassigning a connection's
            source or target from one neuron to another.

    Returns:
        np.array: The mutated layer.
    """
    for i in range(layer.shape[0]):
        for j in range(layer.shape[1]):
            if np.random.rand() < p_weightchange \
               and layer[i, j] != 0:
                layer[i, j] = polynomial_mutation(
                    layer[i, j], 1e-3, 1, 15)
    for i in range(layer.shape[0]):
        for j in range(layer.shape[1]):
            if np.random.rand() < p_reassign:
                if np.random.rand() < 0.5:
                    k = np.random.randint(0, layer.shape[1])
                    temp = layer[i, j]
                    layer[i, j] = 0
                    layer[i, k] = temp
                else:
                    k = np.random.randint(0, layer.shape[0])
                    temp = layer[i, j]
                    layer[i, j] = 0
                    layer[k, j] = temp
                return layer

    return layer


class Network():
    """
    Class representing a neural network as described in Velez's experiment.
    """

    def __init__(self, layer_config, source_config, _update_cm=True):
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

            _update_cm (bool): Internal parameter for optimizing copying.
                Defaults to True.

        """
        self.node_coords = layer_config
        self.source_coords = np.array(source_config).astype(np.float64)
        self.weights = []
        self.biases = []
        self.activations = [np.zeros((len(layer),)) for layer in layer_config]
        for layer in layer_config[1:]:
            if len(self.weights) == 0:
                size = self.activations[0].shape[0]
            else:
                size = self.weights[-1].shape[0]
            self.weights.append(np.random.rand(len(layer), size)*2-1)
            self.biases.append(np.random.rand(len(layer))*2-1)
        if _update_cm:
            self.update_coeff_map()

    def __eq__(self, other):
        """Magic method that enables identity comparison with other Network
            instances.

        Args:
            other (Network): The network to compare with.

        Returns:
            bool: If the network is identical to the other network.
        """
        for i in range(len(self.node_coords)-1):
            if np.any(self.weights[i] != other.weights[i]):
                return False
            if np.any(self.biases[i] != other.biases[i]):
                return False
        return True

    def update_weights(self, inputs, feedback, summer, eta=0.002, num_updates=5):
        """Updates the weights of the network, facilitating learning.

        Args:
            inputs (np.array): The inputs to the network, with feedback signals.
            feedback (int): The value of the feedback given for the current
                season.
                Assumes that only one seasonal feedback is given at a time.
            summer (bool): Whether or not the feedback value is tied to the
                summer season.
            eta (float): The learning rate.
                Defaults to 0.002.
            num_updates (int): How many times to present the input and perform
                the weight update calculations.
                Defaults to 5.

        Returns:
            None.
        """
        if feedback != 0:
            coeff_map = self.coeff_map_summer if summer else self.coeff_map_winter
            if feedback < 0:
                coeff_map = [-cm for cm in coeff_map]
            w, a = utils.present(inputs,
                                 coeff_map,
                                 self.weights,
                                 self.activations,
                                 self.biases,
                                 eta,
                                 num_updates)
            self.weights = w
            self.activations = a

    def forward(self, inputs):
        """Gives an output from the network based on an input.

        Args:
            inputs (np.array): The inputs to the network, with zeroed-out
                feedback signals.

        Returns:
            np.array: The output layer activations.
        """
        self.activations = utils.forward(inputs,
                                         self.weights,
                                         self.biases)
        return self.activations[-1]

    def update_coeff_map(self):
        """Motivation: Want to precalculate all g(d_im) terms
            Since phi(x) = -phi(-x) per properties of logistic fs,
            then phi(-x) = -phi(x).
            Since one of two nm_inputs is expected to be 0,
            and said nm_inputs are either -1, 0, or 1,
            one can thus simplify the calculation of m:
            m = phi(a_s g(d_is) + a_w g(d_iw))
              = phi(a_s g(d_is)) or phi(a_w g(d_iw))
              = a_s phi(g(d_is)) or a_w phi(g(d_iw))
            Thus, this method constructs a coefficient map
            consisting of all possible values of abs(m)

        Args:
            None.

        Returns:
            None.
        """
        coeff_map = [np.zeros((len(self.source_coords),
                               self.node_coords[i].shape[0]
                               ), dtype=np.float64)
                     for i in range(len(self.node_coords))]
        for i in range(len(self.node_coords)):
            for j in range(self.source_coords.shape[0]):
                for k in range(self.node_coords[i].shape[0]):
                    coeff_map[i][j, k] = phi(g(np.linalg.norm(
                        self.node_coords[i][k]-self.source_coords[j])))
        self.coeff_map_summer = [cm[0] for cm in coeff_map]
        self.coeff_map_winter = [cm[1] for cm in coeff_map]

    def convert_activations(self):
        """Retypes the activation arrays from Cython typed memoryviews to
            Numpy arrays.

        Necessary to allow pickling.

        Args:
            None.

        Returns:
            None.
        """
        for i in range(len(self.activations)):
            self.activations[i] = np.asarray(self.activations[i])

    def convert_weights(self):
        """Retypes the activation arrays from Cython typed memoryviews to
            Numpy arrays.

        Necessary to allow pickling, but made redundant by the handling of
            weights in the training phase.

        Args:
            None.

        Returns:
            None.
        """
        for i in range(len(self.weights)):
            self.weights[i] = np.asarray(self.weights[i])

    def store_original_weights(self):
        """Creates a backup copy of the weights prior to training.
            Necessary to avoid irregular behaviour.

        Args:
            None.

        Returns:
            None.
        """
        self.original_weights = [w.copy() for w in self.weights]

    def load_original_weights(self):
        """Restores the original weights from a backup copy after training.
            Necessary to avoid irregular behaviour.

        Args:
            None.

        Returns:
            None.
        """
        self.weights = [w.copy() for w in self.original_weights]

    def reset_activations(self):
        """Extra method that resets all activations to zero.

        Not used in the main experiments.

        Args:
            None.

        Returns:
            None.
        """
        for i in range(len(self.activations)):
            self.activations[i][:] = 0

    def randomize_weights(self):
        """Extra method that resets all weights to random values.

        Not used in the main experiments.

        Args:
            None.

        Returns:
            None.
        """
        for i in range(len(self.weights)):
            randoms = np.random.rand(*self.weights[i].shape)
            randoms[self.weights[i] == 0] = 0
            self.weights[i] = randoms

    def connection_cost(self):
        """Calculates the connection cost objective for PNGSA.

        Args:
            None.

        Returns:
            float: A value between 0 and 1, where 0 indicates a fully connected
                network, while 1 indicates a completely disconnected network.
                Values have been chosen to encourage sparse connections.
        """
        return 1 - sum([
            np.count_nonzero(w) for w in self.weights
        ])/sum([w.size for w in self.weights])

    def copy(self):
        """Clones the Network instance for e.g. mutation.

        Args:
            None.

        Returns:
            Network: A distinct Network instance with the same values as its
                parent.
        """
        clone = Network(self.node_coords, self.source_coords,
                        _update_cm=False)
        clone.coeff_map_summer = self.coeff_map_summer
        clone.coeff_map_winter = self.coeff_map_winter
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
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
            p_biaschange (float): The probability of changing the bias of a
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
        # add a connection (i.e. overwrite existing, add if 0)
        if np.random.rand() < p_toggle:
            layer = self.weights[np.random.randint(0, len(self.weights))]
            i = np.random.randint(0, layer.shape[0])
            j = np.random.randint(0, layer.shape[1])
            layer[i, j] = np.random.rand()*2-1
        # delete a connection
        if np.random.rand() < p_toggle:
            layer = self.weights[np.random.randint(0, len(self.weights))]
            i = np.random.randint(0, layer.shape[0])
            j = np.random.randint(0, layer.shape[1])
            layer[i, j] = 0
        # mutate weights and reassign weights
        n = sum([np.count_nonzero(weights) for weights in self.weights])
        if p_weightchange == -1:
            p_weightchange = 2/n
        for i in range(len(self.weights)):
            self.weights[i] = _mutate(
                self.weights[i], p_weightchange, p_reassign)
        # nudge positions of neurons
        for layer in self.node_coords:
            for i in range(len(layer)):
                if np.random.rand() < p_nudge:
                    layer[i] = [x + polynomial_mutation(0, 1e-3, 1, 15)
                                for x in layer[i]]
                    self.update_coeff_map()
        # mutate biases
        for layer in self.biases:
            for i in range(layer.shape[0]):
                if np.random.rand() < p_biaschange:
                    layer[i] = polynomial_mutation(layer[i], 1e-3, 1, 15)
