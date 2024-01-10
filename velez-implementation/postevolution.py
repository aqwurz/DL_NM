#!/usr/bin/env python3

from glob import glob
import sys
import pickle
import numpy as np
from tqdm import tqdm

from main import make_environment


def get_best_ind(sub_R):
    """Finds the best individual from a subpopulation.

    Args:
        sub_R: A list of individual dictionaries.
    Returns:
        dict: The best individual in sub_R.
    """
    best_ind = {'performance': -0.5}
    for ind in sub_R:
        if ind['performance'] > best_ind['performance']:
            best_ind = ind
    return ind


def post_evolution_analysis(dirname, num_envs=80, iterations=30,
                            season_length=5, num_foods=4):
    """Analyzes the performance of evolved individuals.

    Args:
        dirname (str): The name of a directory with pickled individuals.
        num_envs (int): How many new envs to evaluate for.
            Defaults to 80.
        iterations (int): How many days the environment lasts for.
            Defaults to 30.
        season_length (int): How many days a season lasts for.
            Defaults to 5.
    Returns:
        TODO
    """
    R = []
    for f in glob(dirname+"/*.pickle"):
        with open(f, 'rb') as bf:
            ind = pickle.load(bf)
            R.append(ind)
    best_R = [get_best_ind(sub_R) for sub_R in R]
    for _ in tqdm(range(num_envs)):
        foods, decision_bit_summer, decision_bit_winter = make_environment()
        for ind in best_R:
            cind = ind.copy()
            cind['network'] = cind['network'].copy()
            network = cind['network']
            # training phase
            eats = np.zeros((iterations,))
            summer = False
            winter = True
            prev_summer = 0
            prev_winter = 0
            score = 0
            eat_count = 0
            nm_inputs = np.zeros((2,), dtype=np.float64)
            for j in range(iterations):
                if j % season_length == 0:
                    summer, winter = winter, summer
                for k in range(num_foods):
                    food = foods[k]
                    inputs = np.zeros((5,))
                    inputs[:3] = food
                    inputs[3] = prev_summer
                    inputs[4] = prev_winter
                    outputs = network.forward(inputs)
                    ate_summer = summer and outputs[0] > 0
                    ate_winter = winter and outputs[1] > 0
                    eat_count += ate_summer + ate_winter
                    prev_summer = food[decision_bit_summer] if ate_summer else 0
                    prev_winter = food[decision_bit_winter] if ate_winter else 0
                    score += prev_summer + prev_winter
                    nm_inputs[0] = prev_summer
                    nm_inputs[1] = prev_winter
                    network.update_weights(nm_inputs)
                eats[j] = 0.5 + score/eat_count if eat_count != 0 else 0.5
            if eat_count != 0:
                score = 0.5 + score/eat_count
            else:
                score = 0.5
            ind['fitness_over_lifetime'] = eats
            ind['training_fitness'] = score
            # testing phase
            summer = False
            winter = True
            prev_summer = 0
            prev_winter = 0
            score = 0
            eat_count = 0
            for j in range(iterations):
                if j % season_length == 0:
                    summer, winter = winter, summer
                for k in range(num_foods):
                    food = foods[k]
                    inputs = np.zeros((5,))
                    inputs[:3] = food
                    inputs[3] = prev_summer
                    inputs[4] = prev_winter
                    outputs = network.forward(inputs)
                    ate_summer = summer and outputs[0] > 0
                    ate_winter = winter and outputs[1] > 0
                    eat_count += ate_summer + ate_winter
                    prev_summer = food[decision_bit_summer] if ate_summer else 0
                    prev_winter = food[decision_bit_winter] if ate_winter else 0
                    score += prev_summer + prev_winter
            if score != 0 and eat_count != 0:
                score = 0.5 + score/eat_count
            else:
                score = 0.5
            ind['testing_fitness'] = score
    for ind in best_R:
        print(f"Training fitness {ind['training_fitness']}")
        print(f"Testing fitness {ind['testing_fitness']}")


if __name__ == '__main__':
    post_evolution_analysis(sys.argv[1])
