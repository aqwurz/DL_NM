cimport numpy as cnp
from libc cimport math

cdef inline float phi(float x)
cdef inline float g(float x)
cdef cnp.float64_t[::1] forward_partial(cnp.float64_t[:,::1] weights, cnp.float64_t[::1] activations, cnp.float64_t[::1] biases)
cdef cnp.float64_t[:,::1] update_weights(cnp.float64_t[::1] m,
                                         cnp.float64_t[:,::1] weights,
                                         cnp.float64_t[::1] activations,
                                         cnp.float64_t[::1] next_activations,
                                         float eta)
