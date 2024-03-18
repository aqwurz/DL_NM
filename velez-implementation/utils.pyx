# cython: profile=False
# cython: np_pythran=True
import cython
import numpy as np

cimport numpy as cnp
from libc cimport math


cdef float sigma = 0.5
cdef float a = math.exp(-2)/math.sqrt(
    2*sigma*sigma*math.pi
)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline float phi(float x):
    return 2/(1 + math.exp(-30*x)) - 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline float g(float x):
    if x > 1.5:
        return 0
    return a*math.exp((-x**2)/(2*sigma*sigma))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef cnp.float64_t[::1] forward_partial(cnp.float64_t[:,::1] weights,
                                        cnp.float64_t[::1] activations,
                                        cnp.float64_t[::1] biases):
    """Performs forward calculations for one layer.

    Args:
        weights (np.array): The weights to use.
        activations (np.array): The inputs of the layer.
        biases (np.array): The biases to use.

    Returns:
        np.array: The activations of the layer.
    """
    cdef cnp.float64_t[::1] result = np.zeros(weights.shape[0])
    cdef Py_ssize_t i, j
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            result[i] += weights[i][j] * activations[j]
        result[i] = phi(result[i] + biases[i])
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def forward(cnp.float64_t[::1] inputs,
            list weights,
            list biases):
    """Gives an output from the network based on the input.

    Args:
        inputs (np.array): The inputs to the network, with zeroed-out
            feedback signals.
        weights (list): The weights of the network.
        biases (list): The biases of the network.

    Returns:
        list: The activations of the network.
    """
    cdef list activations = [0]*(len(weights)+1)
    activations[0] = inputs
    cdef Py_ssize_t i
    cdef int w_size = len(weights)
    for i in range(w_size):
        activations[i+1] = forward_partial(weights[i],
                                           activations[i],
                                           biases[i])
    return activations


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cnp.float64_t[:,::1] update_weights(cnp.float64_t[::1] m_arr,
                                         cnp.float64_t[:,::1] weights,
                                         cnp.float64_t[::1] activations,
                                         cnp.float64_t[::1] next_activations,
                                         float eta):
    """Performs weight updates for one layer.

    Args:
        m_arr (np.array): An array of precalculated m values.
        weights (np.array): The weights to update.
        activations (np.array): The activations of the previous layer,
            i.e. the inputs of this layer.
        next_activations (np.array): The activations of the current layer,
            i.e. the outputs of this layer.
        eta (float): The learning rate.

    Returns:
        np.array: The updated weights.
    """
    cdef float m
    cdef Py_ssize_t i, j
    for i in range(weights.shape[0]):
        m = m_arr[i]
        if m > 1e-3 or m < -1e-3:
            for j in range(weights.shape[1]):
                if weights[i][j] != 0:
                    weights[i][j] += eta * m * activations[j] * next_activations[i]
                    if weights[i][j] < -1:
                        weights[i][j] = -1
                    elif weights[i][j] > 1:
                        weights[i][j] = 1
    return weights


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def present(cnp.float64_t[::1] inputs,
            list ms,
            list weights,
            list activations,
            list biases,
            float eta,
            int num_updates):
    """Updates the weights of a network, facilitating learning.

    Args:
        inputs (np.array): The inputs to the network.
        ms (list): Arrays of precalculated m values.
        weights (list): The weights of the network.
        activations (list): The activations of the network.
        biases (list): The biases of the network.
        eta (float): The learning rate.
        num_updates (int): How many times to present the input and perform
            the weight update calculations.

    Returns:
        np.array: The updated weights of the network.
        np.array: The activations of the network calculated during weight
            updating.
    """
    cdef Py_ssize_t i, _
    activations[0] = inputs
    cdef int a_size = len(activations)
    cdef cnp.float64_t[::1] a
    for _ in range(num_updates):
        for i in range(1, a_size):
            m = ms[i]
            a = forward_partial(weights[i-1],
                                activations[i-1],
                                biases[i-1])
            weights[i-1] = update_weights(m,
                                          weights[i-1],
                                          activations[i-1],
                                          activations[i],
                                          eta)
            activations[i] = a
    return (weights, activations)
