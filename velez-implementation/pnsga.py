#!/usr/bin/env python3

import functools
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from time import time

from network import Network


def dominates(i, j, objectives):
    """Tests if i dominates j.

    Args:
        i (dict): A solution.
        j (dict): A solution.
        objectives (dict): What objectives to check.
    Returns:
        bool: if i > j.
    """
    dom = False
    for m in objectives.keys():
        if i[m] < j[m]:
            return False
        elif i[m] > j[m]:
            dom = True
    return dom


def crowded_compare(i, j):
    """Tests if i partially dominates j.

    Args:
        i (dict): A solution.
        j (dict): A solution.
    Returns:
        bool: if i >n j.
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
    new_i['network'] = new_i['network'].copy()
    new_i['network'].mutate(
        p_toggle=p_toggle,
        p_reassign=p_reassign,
        p_biaschange=p_biaschange,
        p_weightchange=p_weightchange)
    new_i['connection_cost_n'] = new_i['network'].connection_cost()
    return new_i


def initialize_pop(layer_config, source_config, objectives, pop_size=400):
    """Creates an initial population.

    Args:
        layer_config (list): A list of lists of coordinates for each neuron.
        source_config (list): A list of coordinates for each point source.
        objectives (dict): The objectives that each individual is assessed for.
        pop_size (int): The amount of individuals to create.
    Returns:
        list: A list of individuals.
    """
    population = []
    for _ in range(pop_size):
        ind = {
            "network": Network(layer_config, source_config),
            "rank": 0,
            "distance": 0
        }
        for obj in objectives.keys():
            ind[obj] = 0
        population.append(ind)
    return population


def make_new_pop(P, pop_size):
    """Creates children from a parent population.

    Args:
        P (list): The parent population as dicts.
        pop_size (int): The intended size of each population.
    Returns:
        list: The new child population as dicts.
    """
    output = [mutate(i) for i in P]
    if len(P) + len(output) < pop_size:
        output += [mutate(P[i%len(P)])
                   for i in range(pop_size-len(P)-len(output))]
    return output


def fast_non_dominated_sort(P, objectives):
    """Performs a non-dominated sort of P.

    Args:
        P (list): A list of solutions as dictionaries.
        objectives (dict): What objectives to check.
    Returns:
        list: A list of all non-dominated fronts F_i.
    """
    F = [[]]
    S = []
    n = []
    for p in P:
        ip = P.index(p)
        S.append([])
        n.append(0)
        for q in P:
            if dominates(p, q, objectives):
                S[ip].append(q)
            elif dominates(q, p, objectives):
                n[ip] += 1
        if n[ip] == 0:
            p['rank'] = 1
            F[0].append(p)
    i = 0
    while len(F[i]) > 0:
        Q = []
        for p in F[i]:
            ip = P.index(p)
            for q in S[ip]:
                iq = S[ip].index(q)
                n[iq] -= 1
                if n[iq] == 0:
                    q['rank'] = i+2
                    Q.append(q)
        i += 1
        if len(Q) == 0:
            break
        else:
            F.append(Q)
    return F


def crowding_distance_assignment(individuals, objectives):
    """Assigns distances to each solution in individuals.

    Args:
        individuals (list): A list of solutions as dictionaries.
        objectives (dict): What objectives to check.
    Returns:
        None.
    """

    for i in individuals:
        i['distance'] = 0
    for m in objectives.keys():
        I_sorted = sorted(individuals, key=lambda x: x[m])
        I_sorted[0]['distance'] = I_sorted[-1]['distance'] = np.infty
        for i in range(1, len(individuals)-1):
            I_sorted[i]['distance'] += (
                I_sorted[i+1][m]-I_sorted[i-1][m]
            )/(objectives[m]['max']-objectives[m]['min'])


def generation(R, objectives, pop_size, num_parents):
    """Performs an iteration of PNGSA.

    Args:
        P (list): A list of parent solutions.
        Q (list): A list of child solutions.
        objectives (dict): A dictionary of objectives and their parameters.
        pop_size (int): The intended size of each population.
        num_parents (int): The amount of new parents to select.

    Returns:
        list: The new parents.
        list: The new children.
    """

    chosen_objectives = {}
    for obj in objectives.keys():
        p_obj = objectives[obj]['probability']
        if np.random.rand() <= p_obj:
            chosen_objectives[obj] = objectives[obj]
    N = num_parents
    F = fast_non_dominated_sort(R, chosen_objectives)
    P_new = []
    i = 0
    while i < len(F) and len(P_new) + len(F[i]) < N:
        crowding_distance_assignment(F[i], chosen_objectives)
        P_new += F[i]
        i += 1
    if i >= len(F):
        i = len(F) - 1
    F[i].sort(key=functools.cmp_to_key(
        lambda i, j: 1 if crowded_compare(i, j) else -1))
    P_new += F[i][0:(N-len(P_new))]
    Q_new = make_new_pop(P_new, pop_size)
    [ind['network'].reset_weights() for ind in P_new+Q_new]

    return P_new, Q_new


def pnsga(trainer, objectives, pop_size=400, num_generations=20000,
          num_cores=cpu_count(), outfile=None):
    """Runs the PNGSA algorithm.

    Args:
        trainer (fn): The function that runs the experiment on the individual.
        objectives (dict): A dictionary describing each objective.
            Each entry consists of the following:
            - key: The name of the objective.
            - "probability": How likely the objective is to be selected in
                PNSGA.
            - "min": The lowest possible value of the objective.
            - "max": The highest possible value of the objective.
        pop_size (int): The population size.
            Defaults to 400.
        num_generations (int): How many generations to run the algorithm for.
            Defaults to 20000.
        num_cores (int): How many cores to use to use to train individuals.
            Defaults to the number of cores given by
            multiprocessing.cpu_count().
        outfile (str): A path to a file to write average fitnesses to.
            If None, does not write anything.
            If not None, blanks the file before writing to it.
            Defaults to None.

    Returns:
        list: The final population.
    """
    if outfile is not None:
        open(outfile, 'w').close()
    layer_config = [
        [(i-2, 0) for i in range(5)],
        [(i-5.5, 1) for i in range(12)],
        [(i-3.5, 2) for i in range(8)],
        [(i-2.5, 3) for i in range(6)],
        [(i-0.5, 4) for i in range(2)]
    ]
    source_config = [(-3,2), (3,2)]
    P = initialize_pop(layer_config, source_config,
                       objectives, pop_size=pop_size)
    Q = make_new_pop(P, pop_size)
    for i in tqdm(range(num_generations)):
        num_parents = len(P)
        with Pool(num_cores) as pool:
            R = pool.map(trainer, P+Q)
        for ind_i in R:
            ham_dist = 0
            for ind_j in R:
                ham_dist += np.count_nonzero(
                    ind_i['eat_vector'] != ind_j['eat_vector'])
            ind_i['behavioral_diversity'] = ham_dist/len(R)/len(
                ind_i['eat_vector'])
            if outfile is not None:
                with open(outfile, 'a') as f:
                    f.write(f"{ind_i['performance']},")
        if outfile is not None:
            with open(outfile, 'a') as f:
                f.write("\n")
        P, Q = generation(R, objectives, pop_size, num_parents)
    return P+Q
