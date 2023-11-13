import functools
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

from pnsga import fast_non_dominated_sort, \
    crowding_distance_assignment, crowded_compare, tournament_selection, \
    make_new_pop, execute, initialize_pop, mutate, \
    calculate_behavioral_diversity
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

test_full_run()
