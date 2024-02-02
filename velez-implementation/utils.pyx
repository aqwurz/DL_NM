# cython: profile=False
# cython: np_pythran=True
import cython
import numpy as np

cimport numpy as cnp
from libc cimport math


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline float phi(float x):
    return 2/(1 + math.exp(-32*x)) - 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline float g(float x):
    cdef float sigma = 0.5
    if x > 1.5:
        return 0
    return math.exp(-2)/math.sqrt(
        2*sigma*sigma*math.pi)*math.exp(
            -x**2/(2*sigma*sigma))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef cnp.float64_t[::1] forward_partial(cnp.float64_t[:,::1] weights,
                                        cnp.float64_t[::1] activations,
                                        cnp.float64_t[::1] biases):
    cdef cnp.float64_t[::1] result = np.empty(weights.shape[0])
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
            list activations,
            list biases):
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
    cdef float d1 = (x-lower)/(upper-lower)
    cdef float d2 = (upper-x)/(upper-lower)
    cdef float r = np.random.rand()
    cdef float dq
    if r <= 0.5:
        dq = (2*r + (1-2*r)*(1-d1)**(eta+1))**(1/(eta+1))-1
    else:
        dq = 1 - (2*(1-r) + 2*(r-0.5)*(1-d2)**(eta+1))**(1/(eta+1))
    return x + dq*(upper-lower)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cnp.float64_t[:,::1] update_weights(cnp.float64_t[::1] nm_inputs,
                                         cnp.float64_t[:,::1] weights,
                                         cnp.float64_t[:,::1] next_node_coords,
                                         cnp.float64_t[:,::1] source_coords,
                                         cnp.float64_t[::1] activations,
                                         cnp.float64_t[::1] next_activations,
                                         float eta):
    cdef float predistsum, pre_phi, pre_sum
    cdef Py_ssize_t i, j, k
    for i in range(weights.shape[0]):
        pre_phi = 0
        for j in range(nm_inputs.shape[0]):
            predistsum = 0
            for k in range(source_coords.shape[1]):
                pre_sum = source_coords[j][k] - next_node_coords[i][k]
                predistsum += pre_sum * pre_sum
            pre_phi += nm_inputs[j] * g(math.sqrt(predistsum))
        m = phi(pre_phi)
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
                       list node_coords,
                       cnp.float64_t[:,::1] source_coords,
                       list activations,
                       float eta):
    cdef Py_ssize_t i
    cdef int w_size = len(weights)
    for i in range(w_size):
        weights[i] = update_weights(nm_inputs,
                                    weights[i],
                                    node_coords[i+1],
                                    source_coords,
                                    activations[i],
                                    activations[i+1],
                                    eta)
    return weights


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def mutate(cnp.float64_t[:,::1] layer,
                                float p_weightchange,
                                float p_reassign):
    cdef float temp
    cdef int k
    for i in range(layer.shape[0]):
        for j in range(layer.shape[1]):
            if np.random.rand() < p_weightchange \
               and layer[i][j] != 0:
                layer[i][j] = polynomial_mutation(
                    layer[i][j], -1, 1, 20)
            if np.random.rand() < p_reassign:
                k = np.random.randint(0, layer.shape[1])
                temp = layer[i][j]
                layer[i][j] = layer[i][k]
                layer[i][k] = temp
    return np.asarray(layer)
