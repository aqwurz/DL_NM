#!/usr/bin/env python3

from glob import glob
import sys
import pickle
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from pnsga import Environment, decode_food_id


def plot_population(dirname):
    R = []
    if "pickle" in dirname:
        with open(dirname, 'rb') as f:
            ind = pickle.load(f)
            R.append(ind)
    else:
        for f in glob(dirname+"/*.pickle"):
            with open(f, 'rb') as bf:
                ind = pickle.load(bf)
                R.append(ind)
    for Ri in R:
        x = [ind['performance'] for ind in Ri]
        y = [ind['behavioral_diversity'] for ind in Ri]
        c = [ind['rank'] for ind in Ri]
        plt.scatter(x=x, y=y, c=c, cmap='viridis_r')
        #for (xi, yi, rank) in zip(x, y, c):
        #    plt.text(xi, yi, rank, va='bottom', ha='center')
        plt.show()


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
                            season_length=5, num_foods=8):
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
    if "pickle" in dirname:
        with open(dirname, 'rb') as f:
            ind = pickle.load(f)
            R.append(ind)
    else:
        for f in glob(dirname+"/*.pickle"):
            with open(f, 'rb') as bf:
                ind = pickle.load(bf)
                R.append(ind)
    best_R = [get_best_ind(sub_R) for sub_R in R]
    for ind in best_R:
        ind['fitness_over_lifetime'] = 0
        ind['training_fitness'] = 0
        ind['testing_fitness'] = 0
        ind['perfect'] = 0
        ind['known'] = 0
        ind['retained'] = 0
    count_identical_decision_bits = 0
    for _ in tqdm(range(num_envs)):
        env = Environment()
        count_identical_decision_bits += np.all(env.foods_summer == env.foods_winter)
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
            nm_inputs = np.zeros((2,), dtype=np.float64)
            food_count = 0
            good_count = 0
            bad_count = 0
            known_prev_s = False
            known_prev_w = False
            for j in range(iterations):
                if j % season_length == 0:
                    summer, winter = winter, summer
                for k in range(num_foods):
                    food_count += 1
                    food = env.presentation_order[j][k]
                    inputs = np.zeros((5,))
                    inputs[:3] = decode_food_id(food)
                    outputs = network.forward(inputs)
                    ate_summer = summer and outputs[0] > 0
                    ate_winter = winter and outputs[1] > 0
                    prev_summer = 1 if food in env.foods_summer else -1
                    prev_winter = 1 if food in env.foods_winter else -1
                    feedback_summer = prev_summer if ate_summer else 0
                    feedback_winter = prev_winter if ate_winter else 0
                    if feedback_summer > 0:
                        good_count += 1
                    elif feedback_summer < 0:
                        bad_count += 1
                    elif feedback_winter > 0:
                        good_count += 1
                    elif feedback_winter < 0:
                        bad_count += 1
                    inputs[3] = feedback_summer
                    inputs[4] = feedback_winter
                    nm_inputs[0] = feedback_summer
                    nm_inputs[1] = feedback_winter
                    network.update_weights(
                        inputs,
                        feedback_summer if summer else feedback_winter,
                        summer
                    )
                eats[j] = 0.5 + (good_count-bad_count)/food_count
                if (j+1) % season_length == 0:
                    known_s = 0
                    known_w = 0
                    for k in range(num_foods):
                        food = env.presentation_order[j][k]
                        inputs = np.zeros((5,))
                        inputs[:3] = decode_food_id(food)
                        outputs = network.forward(inputs)
                        ate_summer = outputs[0] > 0
                        ate_winter = outputs[1] > 0
                        prev_summer = 1 if food in env.foods_summer else -1
                        prev_winter = 1 if food in env.foods_winter else -1
                        feedback_summer = prev_summer if ate_summer else 0
                        feedback_winter = prev_winter if ate_winter else 0
                        known_s += feedback_summer
                        known_w += feedback_winter
                    if known_s == 4:
                        ind['known'] += 1
                    if known_w == 4:
                        ind['known'] += 1
                    if known_s == 4 and known_w == 4:
                        ind['perfect'] += 1
                    if known_prev_s and known_s == 4:
                        ind['retained'] += 1
                    if known_prev_w and known_w == 4:
                        ind['retained'] += 1
                    known_prev_s = known_s == 4
                    known_prev_w = known_w == 4
            score = 0.5 + (good_count-bad_count)/food_count
            ind['fitness_over_lifetime'] += eats
            ind['training_fitness'] += score
            # testing phase
            summer = False
            winter = True
            prev_summer = 0
            prev_winter = 0
            food_count = 0
            good_count = 0
            bad_count = 0
            for j in range(iterations):
                if j % season_length == 0:
                    summer, winter = winter, summer
                for k in range(num_foods):
                    food_count += 1
                    food = env.presentation_order[j][k]
                    inputs = np.zeros((5,))
                    inputs[:3] = decode_food_id(food)
                    outputs = network.forward(inputs)
                    ate_summer = summer and outputs[0] > 0
                    ate_winter = winter and outputs[1] > 0
                    prev_summer = 1 if food in env.foods_summer else -1
                    prev_winter = 1 if food in env.foods_winter else -1
                    feedback_summer = prev_summer if ate_summer else 0
                    feedback_winter = prev_winter if ate_winter else 0
                    if feedback_summer > 0:
                        good_count += 1
                    elif feedback_summer < 0:
                        bad_count += 1
                    elif feedback_winter > 0:
                        good_count += 1
                    elif feedback_winter < 0:
                        bad_count += 1
            ind['testing_fitness'] += 0.5 + (good_count-bad_count)/food_count
    print(f"Identical decision bit envs: {count_identical_decision_bits}")
    for ind in best_R:
        ind['fitness_over_lifetime'] /= num_envs
        ind['training_fitness'] /= num_envs
        ind['testing_fitness'] /= num_envs
        print(f"Training fitness {ind['training_fitness']}")
        print(f"Testing fitness {ind['testing_fitness']}")
        print(f"Perfect {ind['perfect']}")
        print(f"Retained% {ind['retained']/ind['known']}")
        plt.plot(ind['fitness_over_lifetime'])
        plt.grid(axis='x', color='0.95')
        plt.show()
    print()
    print(f"Mean training fitness: {np.mean([ind['training_fitness'] for ind in best_R])}")
    print(f"Mean testing fitness: {np.mean([ind['testing_fitness'] for ind in best_R])}")
    print(f"Mean Perfect: {np.mean([ind['perfect'] for ind in best_R])}")
    print(f"Mean Retained%: {np.mean([ind['retained']/ind['known'] for ind in best_R])}")
    print()
    print(f"Max training fitness: {np.max([ind['training_fitness'] for ind in best_R])}")
    print(f"Max testing fitness: {np.max([ind['testing_fitness'] for ind in best_R])}")
    print(f"Max Perfect: {np.max([ind['perfect'] for ind in best_R])}")
    print(f"Max Retained%: {np.max([ind['retained']/ind['known'] for ind in best_R])}")


if __name__ == '__main__':
    # plot_population(sys.argv[1])
    post_evolution_analysis(sys.argv[1])
