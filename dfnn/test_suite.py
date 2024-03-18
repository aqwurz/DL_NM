#!/usr/bin/env python3

import importlib
import sys

import numpy as np

from network import polynomial_mutation, Network, g
from pnsga import Environment, decode_food_id, dominates, crowded_compare, \
    initialize_pop, clone_individual, set_objective_value, mutate, \
    make_new_pop, fast_non_dominated_sort, non_dominated_sorting, \
    execute
from main import train

# the following code is from
# https://shwina.github.io/cython-testing/
cython_test_modules = ["test_utils"]

for mod in cython_test_modules:
    try:
        mod = importlib.import_module(mod)
        for name in dir(mod):
            item = getattr(mod, name)
            if callable(item) and name.startswith("test_"):
                setattr(sys.modules[__name__], name, item)
    except ImportError:
        pass

# end code snippet


layer_config_config = [5, 12, 8, 6, 2]
layer_config = [
    np.array([
        (j-layer_config_config[i]/2+0.5, float(i))
        for j in range(layer_config_config[i])])
    for i in range(len(layer_config_config))]
source_config = [(-3.0, 2.0), (3.0, 2.0)]

objectives_pa = {'performance': 1.00, 'behavioral_diversity': 1.00}
objectives_pcc = {'performance': 1.00, 'behavioral_diversity': 1.00,
                  'connection_cost_n': 0.75}


def create_random_food():
    return decode_food_id(np.random.randint(0, 8))


def prepare_individual():
    objectives = objectives_pa
    population = initialize_pop(layer_config, source_config, objectives, train)
    obj_indexing = list(objectives.keys())
    population[0]['mapping'] = {m: obj_indexing.index(m) for m in objectives.keys()}
    envs = [Environment() for _ in range(4)]
    population[0]['envs'] = envs
    return train(population[0])


def test_g_numba():
    r = np.random.rand()
    high = 2
    sigma = 0.5
    expected = (np.exp(-2)/np.sqrt((2*sigma**2)*np.pi))*np.exp((-r**2)/(2*sigma**2))
    actual_r = g(r)
    actual_high = g(high)
    assert np.allclose(actual_r, expected), \
        f"g gives wrong value: Expected {expected}, got {actual_r}"
    assert actual_high == 0, \
        f"g gives wrong value for x > 1.5: Expected 0, got {actual_high}"


def test_network_eq():
    a = Network(layer_config, source_config)
    b = Network(layer_config, source_config)
    assert a != b, "__eq__ does not distinguish between networks"
    c = a.copy()
    assert a == c, "__eq__ does not identify clones as equal"


def test_coeff_map():
    network = Network(layer_config, source_config)
    node_coords = network.node_coords
    expected = []
    for l in range(len(node_coords)):
        pre_gs = np.array([[
            np.linalg.norm(network.node_coords[l][i]-network.source_coords[j])
            for j in range(len(network.source_coords))]
            for i in range(len(network.node_coords[l]))
        ])
        gs = np.array([[g(pre_gs[i, j])
                        for j in range(pre_gs.shape[1])]
                       for i in range(pre_gs.shape[0])])
        expected.append(gs)
    actual = network.coeff_map
    for i in range(len(expected)):
        assert np.allclose(expected[i], actual[i]), \
            f"update_coeffs calculates wrong values for layer {i}:\n    Expected:\n{expected[i]},\n    got:\n{actual[i]}"


def test_forward():
    network = Network(layer_config, source_config)
    food = create_random_food()
    weights = network.weights
    biases = network.biases

    @np.vectorize
    def phi(x):
        return 2/(1+np.exp(-30*x))-1

    inputs = np.concatenate((food, np.zeros((2,))))
    expected_activations = [inputs]
    for i in range(len(weights)):
        expected_activations.append(
            phi(weights[i] @ expected_activations[i] + biases[i]))
    expected = expected_activations[-1]
    actual = network.forward(inputs)
    network.convert_activations()
    actual_activations = network.activations
    assert np.allclose(expected, actual), \
        f"forward gives wrong value: Expected {expected}, got {actual}"
    assert np.all([np.allclose(expected_activations[i], actual_activations[i],
                               atol=0.000001)
                   for i in range(len(expected_activations))]), \
        f"forward produces wrong activations: Expected {expected_activations}, got {actual_activations}"


def test_update_weights_none():
    network = Network(layer_config, source_config)
    food = create_random_food()
    expected = [w.copy() for w in network.weights]
    nm_inputs = np.zeros((2,))
    inputs = np.concatenate((food, nm_inputs))
    _ = network.forward(inputs)
    network.update_weights(nm_inputs)
    network.convert_weights()
    actual = network.weights
    for i in range(len(expected)):
        assert np.allclose(expected[i], actual[i]), \
            f"update_weights produces wrong weights in layer {i} when no update expected:\n    Expected:\n{expected[i]},\n    got:\n{actual[i]}"


def test_update_weights_all():
    network = Network(layer_config, source_config)
    food = create_random_food()
    rl = np.random.randint(1, len(layer_config)-1)
    rx = np.random.randint(0, network.weights[rl].shape[0])
    ry = np.random.randint(0, network.weights[rl].shape[1])
    network.weights[rl][rx, ry] = 0
    weights = [w.copy() for w in network.weights]
    nm_inputs = np.zeros((2,))
    inputs = np.concatenate((food, nm_inputs))
    _ = network.forward(inputs)
    activations = [a.copy() for a in network.activations]
    season = 0 if np.random.rand() < 0.5 else 1
    nm_inputs[season] = 1 if np.random.rand() < 0.5 else -1
    inputs = np.concatenate((food, nm_inputs))
    eta = 0.002

    @np.vectorize
    def phi(x):
        return 2/(1+np.exp(-30*x))-1

    expected = []
    activations[0] = inputs
    for l in range(len(weights)):
        w = weights[l]
        new_weights = w.copy()
        pre_gs = np.array([[
            np.linalg.norm(network.node_coords[l+1][i]-network.source_coords[j])
            for j in range(len(network.source_coords))]
            for i in range(len(network.node_coords[l+1]))
        ])
        gs = np.array([[g(pre_gs[i, j])
                        for j in range(pre_gs.shape[1])]
                       for i in range(pre_gs.shape[0])])
        a = phi(w @ activations[l] + network.biases[l])
        for i in range(w.shape[0]):
            m = phi(gs[i] @ nm_inputs)
            if np.abs(m) > 1e-3:
                for j in range(w.shape[1]):
                    if new_weights[i, j] != 0:
                        delta = eta * m * activations[l+1][i] * activations[l][j]
                        new_weights[i, j] += delta
                        assert w[i, j] + delta == new_weights[i, j]
        new_weights = np.clip(new_weights, -1, 1)
        activations[l+1] = a
        if l not in [0, 3]:
            assert np.any(new_weights != w), f"Error in test, layer {l+1}: No differences"
        expected.append(new_weights)

    network.present(inputs, nm_inputs)
    network.convert_weights()
    actual = network.weights
    assert actual[rl][rx, ry] == 0, \
        "update_weights reintroduces deleted connections"
    for i in range(len(expected)):
        print("---")
        print(weights[i])
        print("---")
        print(expected[i])
        print("---")
        print(actual[i])
        print("---")
        assert np.allclose(expected[i], actual[i]), '\n'.join([
            f"update_weights produces wrong weights in layer {i+1}:",
            "    Diff between expected and actual:",
            f"{actual[i] - expected[i]}",
            "    Diff between original and actual:",
            f"{actual[i] - weights[i]}"
        ])
        assert np.allclose(activations[i+1], network.activations[i+1]), \
            f"update_weights produces wrong activations in layer {i+1}: Expected {activations[i+1]}, got {network.activations[i+1]}"


def test_copy_network():
    network = Network(layer_config, source_config)
    clone = network.copy()
    clone.weights[0][0, 0] = 9
    assert network.weights[0][0, 0] != clone.weights[0][0, 0], \
        "copy fails to copy weights"
    clone.biases[0][0] = 9
    assert network.biases[0][0] != clone.biases[0][0], \
        "copy fails to copy biases"


def test_dominates():
    a = np.array([0.0, 0])
    b = np.array([1.0, 1])
    c = np.array([0.0, 1])
    d = np.array([1.0, 0])
    expected = np.array([
        [False, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ])
    actual = np.zeros((4,4), dtype=bool)
    for x in range(4):
        for y in range(4):
            actual[x, y] = dominates([a, b, c, d][x], [a, b, c, d][y])

    assert np.all(expected == actual), f"dominates gives wrong values: Expected {expected}, got {actual}"


def test_crowded_compare():
    a = {'rank': 0, 'distance': 0}
    b = {'rank': 1, 'distance': 1}
    c = {'rank': 0, 'distance': 1}
    d = {'rank': 1, 'distance': 0}
    expected = np.array([
        [False, True, False, True],
        [False, False, False, True],
        [True, True, False, True],
        [False, False, False, False]
    ])
    actual = np.zeros((4,4), dtype=bool)
    for x in range(4):
        for y in range(4):
            actual[x, y] = crowded_compare([a, b, c, d][x], [a, b, c, d][y])

    assert np.all(expected == actual), f"crowded_compare gives wrong values: Expected {expected}, got {actual}"


def test_clone_individual():
    individual = prepare_individual()
    clone = clone_individual(individual)
    clone['performance'] = 9
    assert individual['performance'] != clone['performance'], \
        "clone_invididual fails to clone values"
    clone['network'].weights[0][0, 0] = 9
    assert individual['network'].weights[0][0, 0] != clone['network'].weights[0][0, 0], \
        "clone_individual fails to copy weights"
    clone['network'].biases[0][0] = 9
    assert individual['network'].biases[0][0] != clone['network'].biases[0][0], \
        "clone_individual fails to copy biases"
    clone['eat_vector'] = np.zeros_like(individual['eat_vector'])
    assert np.sum(clone['eat_vector']) != np.sum(individual['eat_vector']), \
        "clone_individual fails to copy eat vectors"


def test_set_objective_value():
    individual = prepare_individual()
    m = "performance"
    set_objective_value(individual, m, 9)
    assert individual[m] == 9, "set_objective_value fails to set separate value"
    assert individual['objective_values'][individual['mapping'][m]] == 9, \
        "set_objective_value fails to set array value"
    assert individual['objective_values'][individual['mapping'][m]] == individual[m], \
        "set_objective_value fails to set separate and array values equally"


def test_mutate_function():
    as_expected_w = []
    as_expected_b = []
    individual = prepare_individual()
    for _ in range(100):
        mutated = mutate(individual, objectives_pa)
        for i in range(len(layer_config)-1):
            as_expected_w.append(np.any(
                individual['network'].weights[i] != mutated['network'].weights[i]))
        for i in range(len(layer_config)-1):
            as_expected_b.append(np.any(
                individual['network'].biases[i] != mutated['network'].biases[i]))
    assert np.any(as_expected_w), \
        "mutate fails to mutate weights (might be chance)"
    assert np.any(as_expected_b), \
        "mutate fails to mutate biases (might be chance)"


def test_initialize_pop():
    population = initialize_pop(layer_config, source_config, objectives_pa, train)
    duplicates = 0
    for ind_i in population:
        for ind_j in population:
            if ind_i['network'] == ind_j['network']:
                duplicates += 1
    assert duplicates-400 == 0, f"initialize_pop made {duplicates-400} duplicate individuals"


def test_make_new_pop():
    population = initialize_pop(layer_config, source_config, objectives_pa, train)
    mutated = make_new_pop(population, objectives_pa)
    identicals = 0
    for ind_i in population:
        for ind_j in mutated:
            if ind_i['network'] == ind_j['network']:
                identicals += 1
    print(f"Identicals: {identicals}")
    assert identicals < 400, "make_new_pop made an identical population from the source population"


def test_fast_nondominated_sort():
    values = np.random.rand(400, 2)
    expected = np.zeros((400,), dtype=np.int64)
    list_values = list(values)
    fr = 1
    while len(list_values) > 0:
        to_pop = []
        for i in range(len(list_values)):
            add = True
            for j in range(len(list_values)):
                if dominates(list_values[j], list_values[i]):
                    add = False
            if add:
                expected[np.where(values == list_values[i])[0][0]] = fr
                to_pop.append(i)
        fr += 1
        if len(to_pop) == 0:
            to_pop = range(len(list_values))
        to_pop.sort(reverse=True)
        for v in to_pop:
            list_values.pop(v)
    actual = fast_non_dominated_sort(values)
    assert np.all(expected == actual), \
        f"fast_non_dominated_sort gives wrong indices:\nAt indices {np.where(expected != actual)}, expected {expected[np.where(expected != actual)]}, got {actual[np.where(expected != actual)]}"
