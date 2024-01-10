import functools
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import cProfile
import math

from pnsga import fast_non_dominated_sort, \
    crowding_distance_assignment, crowded_compare, tournament_selection, \
    make_new_pop, execute, initialize_pop, mutate, \
    calculate_behavioral_diversity, pnsga
from network import phi, g
from main import train

objectives = {
    "performance": {
        "probability": 1.00,
        "min": -0.5,
        "max": 1.5
    },
    "behavioral_diversity": {
        "probability": 1.00,
        "min": 0,
        "max": 1
    }
}


def test_parent_creation():
    R = [{m: np.random.rand() for m in objectives.keys()} for _ in range(400)]
    F = fast_non_dominated_sort(R, objectives)
    P_new = []
    i = 0
    while i < len(F) and len(P_new) + len(F[i]) < 200:
        P_new += F[i]
        i += 1
    if i >= len(F):
        i = len(F) - 1
    crowding_distance_assignment(F[i], objectives)
    F[i].sort(key=functools.cmp_to_key(
        lambda i, j: (1 if crowded_compare(i, j) else -1)
    ), reverse=True)
    for ind in P_new:
        ind['chosen'] = True
    selected = tournament_selection(P_new)
    for ind in selected:
        ind['tc_chosen'] = True
    plt.scatter(x=[ind['performance'] for ind in R],
                y=[ind['behavioral_diversity'] for ind in R],
                c=[ind['rank'] for ind in R],
                s=[40 if 'chosen' in ind else 10 for ind in R],
                cmap='viridis_r')
    plt.scatter(x=[ind['performance'] for ind in R if 'tc_chosen' in ind],
                y=[ind['behavioral_diversity'] for ind in R if 'tc_chosen' in ind],
                marker='x',
                c='black')
    plt.gca().set_aspect('equal')
    plt.show()


def test_nds():
    P = [{m: np.random.rand() for m in objectives.keys()} for _ in range(400)]
    for ind in P:
        ind['objective_values'] = np.array([ind[m] for m in objectives.keys()])
    F = fast_non_dominated_sort(P, objectives)
    P_objv = np.array([ind['objective_values'] for ind in P])
    ranks = fast_non_dominated_sort_alt(P_objv)
    F_alt = [[]]*ranks.max()
    for i in range(ranks.max()):
        F_alt[i] = [P[j] for j in range(len(P)) if (ranks == i+1)[j]]
    print([[P.index(ind) for ind in Fi] for Fi in F])
    print([[P.index(ind) for ind in Fi] for Fi in F_alt])

def test_tournament():
    R = [{'performance': np.random.rand()} for _ in range(200)]
    out = tournament_selection(R)
    print(out)
    print(len(out))


def test_full_run():
    pop_size = 400
    num_cores = cpu_count()
    layer_config = [
        [(i-2, 0) for i in range(5)],
        [(i-5.5, 1) for i in range(12)],
        [(i-3.5, 2) for i in range(8)],
        [(i-2.5, 3) for i in range(6)],
        [(i-0.5, 4) for i in range(2)]
    ]
    source_config = [(-3,2), (3,2)]
    P = initialize_pop(layer_config, source_config,
                       objectives, pop_size=pop_size//2)
    P = execute(P, train, objectives, num_cores, None)
    Q = make_new_pop(P, pop_size//2, objectives)
    Q = execute(Q, train, objectives, num_cores, None)
    generation = 0
    while True:
        chosen_objectives = {}
        for obj in objectives.keys():
            p_obj = objectives[obj]['probability']
            if np.random.rand() <= p_obj:
                chosen_objectives[obj] = objectives[obj]
        N = pop_size//2
        F = fast_non_dominated_sort(P+Q, chosen_objectives)
        P_new = []
        i = 0
        while i < len(F) and len(P_new) + len(F[i]) < N:
            # crowding_distance_assignment(F[i], chosen_objectives)
            P_new += F[i]
            i += 1
        if i >= len(F):
            i = len(F) - 1
        crowding_distance_assignment(F[i], chosen_objectives)
        F[i].sort(key=functools.cmp_to_key(
            lambda i, j: (1 if crowded_compare(i, j) else -1)),
            reverse=True)
        for ind in P_new:
            ind['chosen'] = True
        P_new += F[i][0:(N-len(P_new))]
        selected = tournament_selection(P_new, objectives, num_selected=50)
        for ind in selected:
            ind['tc_chosen'] = True
        Q_new = [mutate(np.random.choice(selected))
                 for _ in range(pop_size//2)]
        for i in Q_new:
            for m in i.keys():
                if m not in ['network', 'connection_cost_n']:
                    i[m] = 0
        R = P+Q
        plt.scatter(x=[ind['performance'] for ind in R],
                    y=[ind['behavioral_diversity'] for ind in R],
                    c=[ind['rank'] for ind in R],
                    s=[40 if 'chosen' in ind else 10 for ind in R],
                    cmap='viridis_r')
        plt.scatter(x=[ind['performance'] for ind in R if 'tc_chosen' in ind],
                    y=[ind['behavioral_diversity'] for ind in R if 'tc_chosen' in ind],
                    marker='x',
                    c='black')
        plt.axvline(x=np.mean([ind['performance'] for ind in R]))
        plt.axvline(x=np.mean([ind['performance'] for ind in P]), color="red")
        P = P_new
        Q = execute(Q_new, train, objectives, num_cores, None)
        if 'behavioral_diversity' in objectives.keys():
            calculate_behavioral_diversity(P+Q)
        R = P+Q
        plt.scatter(x=[ind['performance'] for ind in Q],
                    y=[ind['behavioral_diversity'] for ind in Q],
                    c='blue',
                    s=20)
        y = lambda x: (-0.087462*x-0.84)/(-0.09*x-1.2)
        plt.axvline(y(generation), color="green")
        plt.show()
        for ind in P+Q:
            ind.pop('chosen', None)
            ind.pop('tc_chosen', None)
        generation += 1


def profile(single_core=False):
    objectives = {
        "performance": {
            "probability": 1.00,
            "min": -0.5,
            "max": 1.5
        },
        "behavioral_diversity": {
            "probability": 1.00,
            "min": 0,
            "max": 1
        },
        "connection_cost_n": {
            "probability": 0.75,
            "min": 0,
            "max": 1
        }
    }
    if single_core:
        pnsga(train, objectives,
              profile=True,
              num_generations=20,
              num_cores=1
              )
    else:
        pnsga(train, objectives,
              profile=True,
              num_generations=20,
              )


def profile_uw():
    nm_inputs = np.zeros((2,), dtype=np.float64)
    weights = np.random.rand(2,6)
    next_node_coords = np.array([[-1.5,4],[-0.5,4]])
    source_coords = np.array([[-3.0,2.0],[3.0,2.0]])
    activations = np.array([-1,-1,1,1,-1,-1]).astype(np.float64)
    next_activations = np.array([1,-0.54927423])
    eta = 0.002
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(10000):
        M = np.zeros((1, len(weights)))
        for j in range(len(weights)):
            distances = np.zeros((len(source_coords),))
            for k in range(len(source_coords)):
                predist = source_coords[k]-next_node_coords[j]
                predistsum = 0
                for l in predist:
                    predistsum += l**2
                distances[k] = g(math.sqrt(predistsum))
            M[0, j] = phi(nm_inputs @ distances)
        weights += np.outer(next_activations, activations) * M.T * eta
        weights.clip(-1, 1, out=weights)
    pr.disable()
    pr.print_stats(sort='tottime')


profile(single_core=True)
