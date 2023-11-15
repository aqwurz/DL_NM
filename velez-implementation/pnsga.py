#!/usr/bin/env python3

import functools
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pickle import dump

from network import Network


def dominates(i, j, objectives):
    """Tests if i dominates j.

    Args:
        i (dict): A solution.
        j (dict): A solution.
        objectives (dict): What objectives to check.
    Returns:
        bool: if i <dom j.
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
        bool: if i <ndom j.
    """
    return (i['rank'] < j['rank']) \
        or ((i['rank'] == j['rank']) and (i['distance'] > j['distance']))


def mutate(i, p_toggle=0.20, p_reassign=0.15,
           p_biaschange=0.10, p_weightchange=-1,
           p_nudge=0.00,
           paper_mutation=False):
    """Creates a new individual by mutating its parent.

    Args:
        i (dict): The parent individual.
        p_toggle (float): The probability of toggling a connection.
            Defaults to 0.20 (20%).
        p_reassign (float): The probability of reassigning a connection's
            source or target from one neuron to another.
            Defaults to 0.15 (15%).
        p_biaschange (float): The prbability of changing the bias of a neuron.
            Defaults to 0.10 (10%).
        p_weightchange (float): The probability of changing the weight of a
            connection.
            If -1, sets the probability to 2/n, n being the number of
            connections in the whole network.
            Defaults to -1.
        p_nudge (float): The probablility of adjusting the position of a
            neuron.
            Defaults to 0.00 (0%).
        paper_mutation (bool): Which mutation paradigm to use:
            If True, the mutation operations are applied thus, following
                the previous work in Ellefsen and Velez:
                - Toggling connections: Per network
                - Reassigning connections: Per connection
                - Mutating bias: Per neuron
                - Mutating weights: Per connection
            If False:
                - Toggling connections: Per layer
                - Reassigning connections: Per layer
                - Mutating bias: Per layer
                - Mutating weights: Per layer
            Defaults to False.

    Returns:
        dict: The mutated individual.
    """
    new_i = i.copy()
    new_i['network'] = new_i['network'].copy()
    new_i['network'].mutate(
        p_toggle=p_toggle,
        p_reassign=p_reassign,
        p_biaschange=p_biaschange,
        p_weightchange=p_weightchange,
        p_nudge=p_nudge,
        paper_mutation=paper_mutation)
    new_i['connection_cost_n'] = new_i['network'].connection_cost()
    return new_i


def initialize_pop(layer_config, source_config, objectives, pop_size=400):
    """Creates an initial population.

    Args:
        layer_config (list): A list of lists of coordinates for each neuron.
        source_config (list): A list of coordinates for each point source.
        objectives (dict): The objectives that each individual is assessed for.
        pop_size (int): The amount of individuals to create.
            Defaults to 400.
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


def tournament_selection(P, objectives, num_selected=50):
    """Performs tournament selection.

    Args:
        P (list): The parent population as dicts.
        num_selected (int): How many individuals to select.
            Defaults to 50.
    Returns:
        list: The selected individuals as dicts.
    """
    selected = P.copy()
    crowding_distance_assignment(selected, objectives)
    while len(selected) > num_selected:
        np.random.shuffle(selected)
        i = 0
        while i+1 < len(selected) and len(selected) > num_selected:
            # if selected[i]['performance'] > selected[i+1]['performance']:
            if crowded_compare(selected[i], selected[i+1]):
                selected.pop(i+1)
            else:  # necessary to guarantee termination
                selected.pop(i)
            i += 1
    return selected


def make_new_pop(P, num_children, objectives,
                 num_selected=50,
                 p_toggle=0.20, p_reassign=0.15,
                 p_biaschange=0.10, p_weightchange=-1,
                 p_nudge=0.00,
                 paper_mutation=False):
    """Creates children from a parent population.

    Args:
        P (list): The parent population as dicts.
        num_children (int): The intended amount of children to make.
        objectives (dict): The objectives that each individual is assessed for.
        num_selected (int): How many individuals to select in tournament
            selection.
            Defaults to 50.
        p_toggle (float): The probability of toggling a connection.
            Defaults to 0.20 (20%).
        p_reassign (float): The probability of reassigning a connection's
            source or target from one neuron to another.
            Defaults to 0.15 (15%).
        p_biaschange (float): The prbability of changing the bias of a neuron.
            Defaults to 0.10 (10%).
        p_weightchange (float): The probability of changing the weight of a
            connection.
            If -1, sets the probability to 2/n, n being the number of
            connections in the whole network.
            Defaults to -1.
        p_nudge (float): The probablility of adjusting the position of a
            neuron.
            Defaults to 0.00 (0%).
        paper_mutation (bool): Which mutation paradigm to use:
            If True, the mutation operations are applied thus, following
                the previous work in Ellefsen and Velez:
                - Toggling connections: Per network
                - Reassigning connections: Per connection
                - Mutating bias: Per neuron
                - Mutating weights: Per connection
            If False:
                - Toggling connections: Per layer
                - Reassigning connections: Per layer
                - Mutating bias: Per layer
                - Mutating weights: Per layer
            Defaults to False.

    Returns:
        list: The new child population as dicts.
    """
    selected = tournament_selection(P, objectives, num_selected=num_selected)
    output = [mutate(np.random.choice(selected),
                     p_toggle=p_toggle,
                     p_reassign=p_reassign,
                     p_biaschange=p_biaschange,
                     p_weightchange=p_weightchange,
                     p_nudge=p_nudge,
                     paper_mutation=paper_mutation)
              for _ in range(num_children)]
    for i in output:
        for m in i.keys():
            if m not in ['network', 'connection_cost_n']:
                i[m] = 0
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
                iq = P.index(q)
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


def generation(R, objectives, pop_size,
               num_selected=50,
               p_toggle=0.20, p_reassign=0.15,
               p_biaschange=0.10, p_weightchange=-1,
               p_nudge=0.00,
               paper_mutation=False):
    """Performs an iteration of PNGSA.

    Args:
        R (list): A list of parent and child solutions.
        objectives (dict): A dictionary of objectives and their parameters.
        pop_size (int): The intended size of each population.
        num_selected (int): How many individuals to select in tournament
            selection.
            Defaults to 50.
        p_toggle (float): The probability of toggling a connection.
            Defaults to 0.20 (20%).
        p_reassign (float): The probability of reassigning a connection's
            source or target from one neuron to another.
            Defaults to 0.15 (15%).
        p_biaschange (float): The prbability of changing the bias of a neuron.
            Defaults to 0.10 (10%).
        p_weightchange (float): The probability of changing the weight of a
            connection.
            If -1, sets the probability to 2/n, n being the number of
            connections in the whole network.
            Defaults to -1.
        p_nudge (float): The probablility of adjusting the position of a
            neuron.
            Defaults to 0.00 (0%).
        paper_mutation (bool): Which mutation paradigm to use:
            If True, the mutation operations are applied thus, following
                the previous work in Ellefsen and Velez:
                - Toggling connections: Per network
                - Reassigning connections: Per connection
                - Mutating bias: Per neuron
                - Mutating weights: Per connection
            If False:
                - Toggling connections: Per layer
                - Reassigning connections: Per layer
                - Mutating bias: Per layer
                - Mutating weights: Per layer
            Defaults to False.

    Returns:
        list: The new parents.
        list: The new children.
    """

    chosen_objectives = {}
    for obj in objectives.keys():
        p_obj = objectives[obj]['probability']
        if np.random.rand() <= p_obj:
            chosen_objectives[obj] = objectives[obj]
    N = pop_size//2
    F = fast_non_dominated_sort(R, chosen_objectives)
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
    P_new += F[i][0:(N-len(P_new))]
    Q_new = make_new_pop(P_new, pop_size//2, objectives,
                         num_selected=num_selected,
                         p_toggle=p_toggle,
                         p_reassign=p_reassign,
                         p_biaschange=p_biaschange,
                         p_weightchange=p_weightchange,
                         p_nudge=p_nudge,
                         paper_mutation=paper_mutation)

    return P_new, Q_new


def calculate_behavioral_diversity(R):
    """Calculates behavioral diversity based on Hamming distance.

    Args:
        R (list): A list of individuals as dictionaries.
    Returns:
        None.
    """
    for ind_i in R:
        ham_dist = 0
        for ind_j in R:
            ham_dist += np.count_nonzero(
                ind_i['eat_vector'] != ind_j['eat_vector'])
        ind_i['behavioral_diversity'] \
            = ham_dist/len(R)/len(ind_i['eat_vector'])


def write_to_file(R, fn, eol=True, only_max=False):
    """Writes fitnesses to a file.

    Args:
        R (list): A list of individuals as dictionaries.
        fn (str): The filename to write to.
        eol (bool): Whether to write a newline at the end.
            Defaults to True.
        only_max (bool): Whether to only write the highest fitness to file.
            Defaults to False.
    Returns:
        None.
    """
    max_ind = {'performance': -1}
    for ind_i in R:
        if max_ind['performance'] < ind_i['performance']:
            max_ind = ind_i
        if not only_max:
            with open(fn, 'a') as f:
                f.write(f"{ind_i['performance']},")
    if only_max:
        with open(fn, 'a') as f:
            f.write(f"{max_ind['performance']},")
    if eol:
        with open(fn, 'a') as f:
            f.write("\n")


def execute(R, trainer, num_cores, outfile, eol=True, only_max=False):
    """Trains individuals and writes fitnesses to a file.

    Args:
        R (list): A list of individuals as dictionaries.
        trainer (fn): The function that runs the experiment on the individual.
        objectives (dict): A dictionary describing each objective.
        num_cores (int): How many cores to use for training.
        outfile (str): The filename to write to.
        eol (bool): Whether to write a newline at the end.
            Defaults to True.
        only_max (bool): Whether to only write the highest fitness to file.
            Defaults to False.
    Returns:
        list: The processed individuals.
    """
    if num_cores > 1:
        with Pool(num_cores) as pool:
            R = pool.map(trainer, R)
    else:
        R = [trainer(ind) for ind in R]
    if outfile is not None:
        write_to_file(R, outfile, eol=eol, only_max=only_max)
    return R


def pnsga(trainer, objectives, pop_size=400, num_generations=20000,
          num_cores=cpu_count(),
          num_selected=50,
          p_toggle=0.20, p_reassign=0.15,
          p_biaschange=0.10, p_weightchange=-1,
          p_nudge=0.00,
          paper_mutation=False,
          outfile=None,
          only_max=False,
          position=0):
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
        num_selected (int): How many individuals to select in tournament
            selection.
            Defaults to 50.
        p_toggle (float): The probability of toggling a connection.
            Defaults to 0.20 (20%).
        p_reassign (float): The probability of reassigning a connection's
            source or target from one neuron to another.
            Defaults to 0.15 (15%).
        p_biaschange (float): The prbability of changing the bias of a neuron.
            Defaults to 0.10 (10%).
        p_weightchange (float): The probability of changing the weight of a
            connection.
            If -1, sets the probability to 2/n, n being the number of
            connections in the whole network.
            Defaults to -1.
        p_nudge (float): The probablility of adjusting the position of a
            neuron.
            Defaults to 0.00 (0%).
        paper_mutation (bool): Which mutation paradigm to use:
            If True, the mutation operations are applied thus, following
                the previous work in Ellefsen and Velez:
                - Toggling connections: Per network
                - Reassigning connections: Per connection
                - Mutating bias: Per neuron
                - Mutating weights: Per connection
            If False:
                - Toggling connections: Per layer
                - Reassigning connections: Per layer
                - Mutating bias: Per layer
                - Mutating weights: Per layer
            Defaults to False.
        outfile (str): A path to a file to write average fitnesses to.
            If None, does not write anything.
            If not None, blanks the file before writing to it.
            Defaults to None.
        only_max (bool): Whether to only write the highest fitness to file.
            Defaults to False.
        position (int): Argument for tqdm to function properly when executing
            multiple jobs in parallel.

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
                       objectives, pop_size=pop_size//2)
    P = execute(P, trainer, num_cores, None)
    Q = make_new_pop(P, pop_size//2, objectives,
                     num_selected=num_selected,
                     p_toggle=p_toggle,
                     p_reassign=p_reassign,
                     p_biaschange=p_biaschange,
                     p_weightchange=p_weightchange,
                     p_nudge=p_nudge,
                     paper_mutation=paper_mutation)
    Q = execute(Q, trainer, num_cores, None)
    if 'behavioral_diversity' in objectives.keys():
        calculate_behavioral_diversity(P+Q)
    for i in tqdm(range(num_generations), position=position):
        P, Q = generation(P+Q, objectives, pop_size,
                          num_selected=num_selected,
                          p_toggle=p_toggle,
                          p_reassign=p_reassign,
                          p_biaschange=p_biaschange,
                          p_weightchange=p_weightchange,
                          p_nudge=p_nudge,
                          paper_mutation=paper_mutation)
        if outfile is not None:
            write_to_file(P, outfile, eol=False, only_max=only_max)
        Q = execute(Q, trainer, num_cores, outfile, only_max=only_max)
        if 'behavioral_diversity' in objectives.keys():
            calculate_behavioral_diversity(P+Q)
    with open(f"{outfile[:-4]}.pickle", 'wb') as f:
        dump(P+Q, f)
    return P+Q
