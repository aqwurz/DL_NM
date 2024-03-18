# cython: profile=False
# cython: np_pythran=True
import cython
import numpy as np
from cytest import cytest

cimport numpy as cnp
from libc cimport math

from utils cimport phi, g, forward_partial, update_weights


@cytest
def test_phi():
    cdef float r = np.random.rand()*2-1
    cdef float expected = 2/(1 + np.exp(-30*r)) - 1
    cdef float actual = phi(r)
    assert np.allclose(actual, expected), \
        f"phi gives wrong value: Expected {expected}, got {actual}"


@cytest
def test_g():
    cdef float r = np.random.rand()
    cdef float high = 2
    cdef float sigma = 0.5
    cdef float expected = (np.exp(-2)/np.sqrt((2*sigma**2)*np.pi))*np.exp((-r**2)/(2*sigma**2))
    cdef float actual_r = g(r)
    cdef float actual_high = g(high)
    assert np.allclose(actual_r, expected), \
        f"g gives wrong value: Expected {expected}, got {actual_r}"
    assert actual_high == 0, \
        f"g gives wrong value for x > 1.5: Expected 0, got {actual_high}"


@cytest
def test_forward_partial():
    cdef cnp.float64_t[:,::1] rw = np.random.rand(3,4)*2-1
    cdef cnp.float64_t[::1] ra = np.random.rand(4)*2-1
    cdef cnp.float64_t[::1] rb = np.random.rand(3)*2-1
    cdef cnp.float64_t[::1] expected = np.zeros((3,))
    cdef cnp.float64_t[::1] pre = np.asarray(rw) @ np.asarray(ra) + np.asarray(rb)
    for i in range(3):
        expected[i] = phi(pre[i])
    cdef cnp.float64_t[::1] actual = forward_partial(rw, ra, rb)
    assert np.allclose(actual, expected), \
        f"forward_partial gives wrong value: Expected {expected}, got {actual}"


@cytest
def test_update_weights():
    cdef cnp.float64_t[::1] rni = np.zeros((2,))
    cdef int season
    if np.random.rand() < 0.5:
        season = 0
    else:
        season = 1
    if np.random.rand() < 0.5:
        rni[season] = 1
    else:
        rni[season] = -1
    cdef cnp.float64_t[:,::1] rw = np.random.rand(8,12)*2-1
    cdef int rx, ry
    rx = np.random.randint(0,8)
    ry = np.random.randint(0,12)
    rw[rx][ry] = 0
    cdef cnp.float64_t[:,::1] expected = rw.copy()
    cdef cnp.float64_t[::1] ra = np.random.rand(12)*2-1
    cdef cnp.float64_t[::1] ran = np.random.rand(8)*2-1
    cdef cnp.float64_t[:,::1] nc = np.array(
        [(i-5.5, 1) for i in range(12)]
    )
    cdef cnp.float64_t[:,::1] sc = np.array(
        [(-3.5, 2), (3.5, 2)]
    )
    cdef cnp.float64_t[:,::1] distances = np.zeros((8,2))
    cdef float m
    cdef float eta = 0.002
    for i in range(8):
        distances[i,0] = g(np.linalg.norm(np.asarray(nc[i])-np.asarray(sc[0])))
        distances[i,1] = g(np.linalg.norm(np.asarray(nc[i])-np.asarray(sc[1])))
    for i in range(8):
        m = sum(np.asarray(rni) * np.asarray(distances[i]))
        m = phi(m)
        if np.abs(m) > 1e-3:
            for j in range(12):
                if expected[i][j] != 0:
                    expected[i][j] += eta * m * ran[i] * ra[j]
    expected = np.clip(expected, -1, 1)
    # TODO: Redo m calculation
    """
    cdef cnp.float64_t[:,::1] actual = update_weights(calculate_m(rni, distances), rw, ra, ran, eta)
    expected_print = np.array2string(np.asarray(expected), precision=8, suppress_small=True)
    actual_print = np.array2string(np.asarray(actual), precision=8, suppress_small=True)
    assert actual[rx][ry] == 0, \
        "update_weights erroneously reintroduces connections"
    assert np.allclose(actual, expected), \
        f"update_weights gives wrong value: Expected {expected_print}, got {actual_print}"
    """
