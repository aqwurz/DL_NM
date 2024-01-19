#!/usr/bin/env python3

import functools
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pickle import dump
from time import time
import cProfile
import numba
from numba.typed import List as NBList

from network import Network


@numba.njit('b1(f8[:], f8[:])')
def dominates(i, j):
    return np.all(i > j)


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


def generate_environment_seed():
    """Creates a new seed for an environment.

    Args:
        None.
    Returns:
        int: The seed.
    """
    s_pos_ind = np.random.randint(0, 6)
    w_pos_ind = np.random.randint(0, 6)
    matrix = np.random.randint(0, 2**12)
    s_dec_ind = np.random.randint(0, 3)
    w_dec_ind = np.random.randint(0, 3)
    return int(f"{s_pos_ind:03b}{w_pos_ind:03b}{matrix:012b}{s_dec_ind:02b}{w_dec_ind:02b}", 2)


def mutate(i, objectives, p_toggle=0.20, p_reassign=0.15,
           p_biaschange=0.10, p_weightchange=-1,
           p_nudge=0.00):
    """Creates a new individual by mutating its parent.

    Args:
        i (dict): The parent individual.
        objectives (dict): The objectives that each individual is assessed for.
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
    )
    if 'connection_cost_n' in objectives.keys():
        cc = new_i['network'].connection_cost()
        new_i['connection_cost_n'] = cc
        new_i['objective_values'][new_i['mapping']['connection_cost_n']] = cc
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
            "distance": 0,
            "mapping": {m: objectives[m]['index'] for m in objectives.keys()},
            "objective_values": np.zeros((len(objectives)),)
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
                 p_nudge=0.00):
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

    Returns:
        list: The new child population as dicts.
    """
    selected = tournament_selection(P, objectives, num_selected=num_selected)
    output = [mutate(np.random.choice(selected),
                     objectives,
                     p_toggle=p_toggle,
                     p_reassign=p_reassign,
                     p_biaschange=p_biaschange,
                     p_weightchange=p_weightchange,
                     p_nudge=p_nudge,
                     )
              for _ in range(num_children)]
    for i in output:
        for m in objectives.keys():
            if m != 'connection_cost_n':
                i['objective_values'][i['mapping'][m]] = 0
                i[m] = 0
        i['rank'] = 0
        i['distance'] = 0
    return output


@numba.njit('i8[:](f8[:,:])', nogil=True, parallel=True)
def fast_non_dominated_sort(P):
    """Performs a non-dominated sort of P.

    Args:
        P (np.array): A matrix of all individuals' objective_values arrays.
    Returns:
        np.array: The ranks of each individual.
    """
    F = NBList()
    S = NBList()
    F0 = NBList()
    n = np.zeros((len(P),))
    R = np.zeros((len(P),), dtype=np.int64)
    for ip in range(P.shape[0]):
        Si = NBList()
        for iq in range(P.shape[0]):
            if dominates(P[ip], P[iq]):
                Si.append(iq)
            elif dominates(P[iq], P[ip]):
                n[ip] += 1
        if n[ip] == 0:
            R[ip] = 1
            F0.append(ip)
        S.append(Si)
    F.append(F0)
    i = 0
    while len(F[i]) > 0:
        Q = NBList()
        for ip in F[i]:
            for iq in S[ip]:
                n[iq] -= 1
                if n[iq] == 0:
                    R[iq] = i+2
                    Q.append(iq)
        i += 1
        if len(Q) == 0:
            break
        else:
            F.append(Q)
    return R


def apply_rank_to_individuals(P, R):
    """Gives each individual its rank after non-dominated sorting.

    Args:
        P (list): The individuals.
        R (np.array): The ranks.
    Returns:
        None.
    """


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
               p_nudge=0.00):
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
    ranks = fast_non_dominated_sort(
        np.array([ind['objective_values'] for ind in R]))
    F = [[]]*ranks.max()
    for i in range(len(R)):
        R[i]['rank'] = ranks[i]
    for i in range(ranks.max()):
        F[i] = [R[j] for j in range(len(R)) if (ranks == i+1)[j]]
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
                         )

    return P_new, Q_new


@numba.njit('f8[:](b1[:,:])', nogil=True, parallel=True)
def _cbd(R):
    return np.array([np.sum(R[i] ^ R)/R.size for i in range(R.shape[0])])


def calculate_behavioral_diversity(R):
    """Calculates behavioral diversity based on Hamming distance.

    Args:
        R (list): A list of individuals as dictionaries.
    Returns:
        None.
    """
    behavior_array = np.array([ind['eat_vector'] for ind in R])
    res_array = _cbd(behavior_array)
    for i in range(len(R)):
        R[i]['behavioral_diversity'] = res_array[i]
        R[i]['objective_values'][R[i]['mapping']['behavioral_diversity']] = res_array[i]


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
    for ind in R:
        ind['seeds'] = [generate_environment_seed() for _ in range(4)]
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
          outfile=None,
          only_max=False,
          position=0,
          profile=False):
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
        outfile (str): A path to a file to write average fitnesses to.
            If None, does not write anything.
            If not None, blanks the file before writing to it.
            Defaults to None.
        only_max (bool): Whether to only write the highest fitness to file.
            Defaults to False.
        position (int): Argument for tqdm to function properly when executing
            multiple jobs in parallel.
            Defaults to 0.
        profile (bool): Whether or not to do cProfile profiling.
            Defaults to False.

    Returns:
        list: The final population.
    """
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    if num_cores == -1:
        np.random.seed(int(f"{time():.0f}{position}"[4:]))
    if outfile is not None:
        open(outfile, 'w').close()
    layer_config_config = [5, 12, 8, 6, 2]
    layer_config = [
        np.array([
            (j-layer_config_config[i]/2-0.5, float(i))
            for j in range(layer_config_config[i])])
        for i in range(len(layer_config_config))]
    source_config = [(-3.0, 2.0), (3.0, 2.0)]
    count = 0
    for m in objectives.keys():
        objectives[m]['index'] = count
        count += 1
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
                     )
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
                          )
        if outfile is not None:
            write_to_file(P, outfile, eol=False, only_max=only_max)
        Q = execute(Q, trainer, num_cores, outfile, only_max=only_max)
        if 'behavioral_diversity' in objectives.keys():
            calculate_behavioral_diversity(P+Q)
    if outfile is not None:
        with open(f"{outfile[:-4]}.pickle", 'wb') as f:
            dump(P+Q, f)
    if profile:
        pr.disable()
        pr.print_stats(sort='tottime')
    return P+Q
