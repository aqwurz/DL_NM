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
cpdef float polynomial_mutation(float x, float lower, float upper, int eta):
    """Adjusts the value of x in accordance with polynomial mutation.

    Args:
        x (float): The value to be mutated.
        lower (float): The minimum value of x.
        upper (float): The maximum value of x.
        eta (int): The distribution index.

    Returns:
        float: The mutated value.
    """
    """
    cdef float d1 = (x-lower)/(upper-lower)
    cdef float d2 = (upper-x)/(upper-lower)
    cdef float r = np.random.rand()
    cdef float dq
    if r <= 0.5:
        dq = (2*r + (1-2*r)*((1-d1)**(eta+1)))**(1/(eta+1))-1
    else:
        dq = 1 - (2*(1-r) + 2*(r-0.5)*((1-d2)**(eta+1)))**(1/(eta+1))
    return x + dq*(upper-lower)
    """
    cdef float r = np.random.rand()
    cdef float dq, out
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef cnp.float64_t[::1] calculate_m(cnp.float64_t[::1] nm_inputs,
                                     cnp.float64_t[:,::1] coeff_map):
    cdef cnp.float64_t[::1] m = np.zeros((coeff_map.shape[0],))
    cdef Py_ssize_t i
    for i in range(coeff_map.shape[0]):
        for j in range(coeff_map.shape[1]):
            m[i] += nm_inputs[j] * coeff_map[i][j]
        m[i] = phi(m[i])
    return m


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cnp.float64_t[:,::1] update_weights(cnp.float64_t[::1] m_arr,
                                         cnp.float64_t[:,::1] weights,
                                         cnp.float64_t[::1] activations,
                                         cnp.float64_t[::1] next_activations,
                                         float eta):
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
def update_weights_all(cnp.float64_t[::1] nm_inputs,
                       list weights,
                       list coeff_map,
                       list activations,
                       float eta):
    cdef Py_ssize_t i
    cdef int w_size = len(weights)
    for i in range(w_size):
        m = calculate_m(nm_inputs, coeff_map[i+1])
        weights[i] = update_weights(m,
                                    weights[i],
                                    activations[i],
                                    activations[i+1],
                                    eta)
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
