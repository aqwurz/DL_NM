#!/usr/bin/env python3

import numpy as np
import os
import cProfile

from network import *
from pnsga import pnsga, Environment, decode_food_id
from multiprocessing import cpu_count, Pool


def train(individual, iterations=30, lifetimes=4, season_length=5,
          num_foods=8, profile=False):
    """Performs the experiment on an individual.

    Args:
        individual (dict): The dictionary representing the individual.
        iterations (int): How many days the environment lasts for.
            Defaults to 30.
        lifetimes (int): How many lifetimes to evaluate the individual for.
            Defaults to 8.
        season_length (int): How many days a season lasts for.
            Defaults to 5.
        num_foods (int): How many foods to include in each environment.
            Defaults to 4.
        profile (bool): Whether or not to do cProfile profiling.
            Defaults to False.

    Returns:
        dict: The trained individual.
    """
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    network = individual['network']
    scores = np.zeros((lifetimes,))
    eat_vector = np.zeros((lifetimes*iterations*num_foods,), dtype=bool)
    original_weights = [weights.copy() for weights in network.weights]
    food_count = 0
    good_count = 0
    bad_count = 0
    inputs = np.zeros((5,))
    nm_inputs = np.zeros((2,), dtype=np.float64)
    for i in range(lifetimes):
        summer = False
        winter = True
        prev_summer = 0
        prev_winter = 0
        env = individual['envs'][i]
        for j in range(iterations):
            if j % season_length == 0:
                summer, winter = winter, summer
            for k in range(num_foods):
                food_count += 1
                food = env.presentation_order[j][k]
                inputs[:3] = decode_food_id(food)
                inputs[3] = 0
                inputs[4] = 0
                outputs = network.forward(inputs)
                ate_summer = summer and outputs[0] > 0
                ate_winter = winter and outputs[1] > 0
                prev_summer = 0
                prev_winter = 0
                if ate_summer and food in env.foods_summer:
                    prev_summer = 1
                elif ate_summer:
                    prev_summer = -1
                if ate_winter and food in env.foods_winter:
                    prev_winter = 1
                elif ate_winter:
                    prev_winter = -1
                if ate_summer or ate_winter:
                    eat_vector[(i*iterations+j) * num_foods + k] = True
                    if prev_summer >= 1 or prev_winter >= 1:
                        good_count += 1
                    elif prev_summer <= -1 or prev_winter <= -1:
                        bad_count += 1
                inputs[3] = prev_summer
                inputs[4] = prev_winter
                network.forward(inputs)
                nm_inputs[0] = prev_summer
                nm_inputs[1] = prev_winter
                network.update_weights(nm_inputs)
        scores[i] = 0.5 + (good_count - bad_count)/food_count
        network.weights = original_weights
    individual['eat_vector'] = eat_vector
    m = np.mean(scores)
    individual['performance'] = m
    individual['objective_values'][individual['mapping']['performance']] = m
    individual['network'].convert_activations()
    individual['network'].convert_weights()
    if profile:
        pr.disable()
        pr.print_stats(sort='tottime')
    return individual


def _worker_wrapper(arg):
    """Wrapper function."""
    return pnsga(*arg)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from datetime import date
    objectives = {
        "performance": 1.00,
        "behavioral_diversity": 1.00,
        "connection_cost_n": 0.75
    }
    parser = ArgumentParser()
    parser.add_argument("--num-execs", "-e", type=int, default=1,
                        help="How many parallel executions to perform")
    parser.add_argument("--num-cores", "-c", type=int, default=cpu_count(),
                        help="The number of cores to use for training \
                        (unused if --num-execs is passed)")
    parser.add_argument("--pop-size", "-p", type=int, default=400,
                        help="The number of individuals per generation")
    parser.add_argument("--num-generations", "-g", type=int, default=20000,
                        help="The number of generations to run PNSGA for")
    parser.add_argument("--num-selected", "-s", type=int, default=50,
                        help="How many individuals to select in tournament \
                        selection")
    parser.add_argument("--p-toggle", "-pt", type=float, default=0.20,
                        help="Probability of toggling connection in mutation \
                        (defaults to 20%%)")
    parser.add_argument("--p-reassign", "-pr", type=float, default=0.15,
                        help="Probability of switching two weights in \
                        mutation (defaults to 15%%)")
    parser.add_argument("--p-biaschange", "-pb", type=float, default=0.10,
                        help="Probability of changing bias in mutation \
                        (defaults to 10%%)")
    parser.add_argument("--p-weightchange", "-pw", type=float, default=-1,
                        help="Probability of changing weight in mutation \
                        (defaults to 2/n)")
    parser.add_argument("--p-nudge", "-pn", type=float, default=0.00,
                        help="Probability of changing neuron position \
                        in mutation (defaults to 0%%)")
    parser.add_argument("--objectives", "-m", nargs="+",
                        default=["performance", "behavioral_diversity",
                                 "connection_cost_n"],
                        help="Which objectives to use")
    parser.add_argument("--outfile", "-o", type=str, required=True,
                        help="Output file name comment")
    parser.add_argument("--only-max", action="store_true",
                        help="Only write highest fitness to file")
    args = parser.parse_args()
    selected_objectives = {m: objectives[m] for m in args.objectives}
    current_date = str(date.today()).replace("-", "")
    if args.num_execs > 1:
        outfolder = f"logs/{current_date}_{args.outfile}"
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
        worker_args = [
            (train,
             selected_objectives,
             args.pop_size,
             args.num_generations,
             -1,
             args.num_selected,
             args.p_toggle,
             args.p_reassign,
             args.p_biaschange,
             args.p_weightchange,
             args.p_nudge,
             f"{outfolder}/{i+1}.csv",
             args.only_max,
             i)
            for i in range(args.num_execs)]
        with Pool(args.num_execs) as pool:
            _ = pool.map(_worker_wrapper, worker_args)
    else:
        pnsga(train,
              selected_objectives,
              pop_size=args.pop_size,
              num_generations=args.num_generations,
              num_selected=args.num_selected,
              num_cores=args.num_cores,
              p_nudge=args.p_nudge,
              outfile=f"logs/{current_date}_{args.outfile}.csv",
              only_max=args.only_max)
