#!/usr/bin/env python3

import numpy as np

from .network import Network


def dominates(i, j, objectives):
    """Tests if i dominates j.

    Args:
        i (dict): A solution.
        j (dict): A solution.
        objectives (list): What objectives to check.
    Returns:
        bool: individualsf i > j.
    """
    dom = False
    for m in objectives:
        if i[m] < j[m]:
            return False
        elif i[m] > j[m]:
            dom = True
    return dom


def crowded_compare(j, i):
    """Tests if i partially dominates j.

    Args:
        i (dict): A solution.
        j (dict): A solution.
    Returns:
        bool: individualsf i >n j.
    """
    return (i['rank'] < j['rank']) \
        or ((i['rank'] == j['rank']) and (i['distance'] > j['distance']))


def mutate(i, p_toggle=0.20, p_reassign=0.15,
           p_biaschange=0.10, p_weightchange=-1):
    """Creates a new individual by mutating its parent.

    Args:
        i (dict): The parent individual.
        p_toggle: The probability of toggling a connection.
            Defaults to 0.20 (20%).
        p_reassign: The probability of reassigning a connection's source
            or target from one neuron to another.
            Defaults to 0.15 (15%).
        p_biaschange: The prbability of changing the bias of a neuron.
            Defaults to 0.10 (10%).
        p_weightchange: The probability of changing the weight of a
            connection.
            If -1, sets the probability to 2/n, n being the number of
            connections in the whole network.
            Defaults to -1.

    Returns:
        dict: The mutated individual.
    """
    new_i = i.copy()
    new_i['network'] = new_i['network'].mutate(
        p_toggle=p_toggle,
        p_reassign=p_reassign,
        p_biaschange=p_biaschange,
        p_weightchange=p_weightchange)
    # TODO recalculate fitness
    return new_i


def initialize_pop(objectives, pop_size=400):
    """Creates an initial population.

    Args:
        objectives (list): The objectives that each individual is assessed for.
        pop_size (int): The amount of individuals to create.
    Returns:
        list: A list of individuals.
    """
    population = []
    for _ in range(pop_size):
        ind = {
            "network": Network(),
            "rank": 0,
            "distance": 0
        }
        for obj in objectives:
            ind[obj] = 0
        population.append(ind)
    return population


def make_new_pop(P):
    """Creates children from a parent population.

    Args:
        P (list): The parent population as dicts.
    Returns:
        list: The new child population as dicts.
    """
    return [mutate(i) for i in P]


def fast_non_dominated_sort(P, objectives):
    """Performs a non-dominated sort of P.

    Args:
        P (list): A list of solutions as dictionaries.
        objectives (list): What objectives to check.
    Returns:
        list: A list of all non-dominated fronts F_i.
    """
    F = []
    S = []
    n = []
    for p in P:
        Fi = []
        ip = P.index(p)
        S.append([])
        n.append(0)
        for q in P:
            if dominates(q, p, objectives):
                S[ip].append(q)
            elif dominates(p, q, objectives):
                n[ip] += 1
        if n[ip] == 0:
            p['rank'] = 1
            Fi.append(p)
        F.append(Fi)
    i = 0
    while len(F[i]) > 0:
        Q = []
        for p in F[i]:
            ip = P.index(p)
            for q in S[ip]:
                iq = S[ip].index(q)
                n[iq] -= 1
                if n[iq] == 0:
                    q['rank'] = i+1
                    Q.append(q)
        i += 1
        F[i] = Q
    return F


def crowding_distance_assignment(individuals, objectives):
    """Assigns distances to each solution in individuals.

    Args:
        individuals (list): A list of solutions as dictionaries.
        objectives (list): What objectives to check.
    Returns:
        None.
    """

    for i in individuals:
        i['distance'] = 0
    for m in objectives:
        I_sorted = sorted(individuals, key=lambda x: x[m])
        I_sorted[0]['distance'] = I_sorted[-1]['distance'] = np.infty
        for i in range(1, len(individuals)-1):
            I_sorted[i]['distance'] += (I_sorted[i+1][m]-I_sorted[i-1][m])/(...)


def generation(P, Q, objectives, num_parents=-1):
    """Performs an iteration of PNGSA.

    Args:
        P (list): A list of parent solutions.
        Q (list): A list of child solutions.
        objectives (dict): A dictionary of objectives and their probabilities.
        num_parents (int): The amount of new parents to select.
            If -1, num_parents is set to len(P).
            Defaults to -1.

    Returns:
        list: The new parents.
        list: The new children.
    """

    if num_parents == -1:
        num_parents = len(P)
    chosen_objectives = []
    for obj in objectives.keys():
        p_obj = objectives[obj]
        if np.random.rand() <= p_obj:
            chosen_objectives.append(obj)
    N = num_parents
    R = P + Q
    F = fast_non_dominated_sort(R, chosen_objectives)
    P_new = []
    i = 0
    while len(P_new) + len(F[i]) > N:
        crowding_distance_assignment(F[i], chosen_objectives)
        P_new += F[i]
        i += 1
    F[i].sort(cmp=lambda i, j: 1 if crowded_compare(i, j) else -1)
    P_new += F[i][0:(N-len(P_new))]
    Q_new = make_new_pop(P_new)

    return P_new, Q_new


def pngsa(pop_size=400, num_generations=20000):
    """Runs the PNGSA algorithm.

    Args:
        pop_size (int): The population size.
            Defaults to 400.
        num_generations (int): How many generations to run the algorithm for.
            Defaults to 20000.

    Returns:
        list: The final population.
    """
    objectives = {
        "performance": 1.00,
        "behavioral_diversity": 1.00,
        "connection_cost": 0.75
    }
    P = initialize_pop(list(objectives.keys()), pop_size=pop_size)
    Q = make_new_pop(P)
    for i in range(num_generations):
        P, Q = generation(P, Q, objectives)
    return P+Q
