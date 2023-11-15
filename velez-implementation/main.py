#!/usr/bin/env python3

import numpy as np
import os

from network import *
from pnsga import *
from multiprocessing import cpu_count, Pool


def train(individual, iterations=30, lifetimes=4, season_length=5):
    """Performs the experiment on an individual.

    Args:
        individual (dict): The dictionary representing the individual.
        iterations (int): How many days the environment lasts for.
            Defaults to 30.
        lifetimes (int): How many lifetimes to evaluate the individual for.
            Defaults to 4.
        season_length (int): How many days a season lasts for.
            Defaults to 5.

    Returns:
        dict: The trained individual.
    """
    network = individual['network']
    scores = np.zeros((lifetimes,))
    eat_vector = np.zeros((lifetimes*iterations,)).astype(bool)
    foods = np.array(
        [[-1 if bit == 0 else 1 for bit in ((i >> 2) % 2, (i >> 1) % 2, i % 2)]
         for i in range(8)])
    for i in range(lifetimes):
        score = 0
        summer = False
        winter = True
        prev_summer = 0
        prev_winter = 0
        decision_bit_summer = np.random.randint(0, 3)
        decision_bit_winter = np.random.randint(0, 3)
        for j in range(iterations):
            if j % season_length == 0:
                summer, winter = winter, summer
            food = foods[np.random.randint(0, len(foods))]
            inputs = np.zeros((5,))
            inputs[:3] = food
            inputs[3] = prev_summer
            inputs[4] = prev_winter
            outputs = network.forward(inputs)
            ate_summer = summer and outputs[0] > 0
            ate_winter = winter and outputs[1] > 0
            eat_vector[i*iterations+j] = ate_summer or ate_winter
            prev_summer = food[decision_bit_summer] if ate_summer else 0
            prev_winter = food[decision_bit_winter] if ate_winter else 0
            score += prev_summer + prev_winter
            network.update_weights([prev_summer, prev_winter])
        # network.reset_weights()
        scores[i] = 0.5 + score/iterations
    individual['eat_vector'] = eat_vector
    individual['performance'] = np.mean(scores)
    return individual


def _worker_wrapper(arg):
    """Wrapper function."""
    return pnsga(*arg)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from datetime import date
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
    parser.add_argument("--paper-mutation", action="store_true",
                        help="Use mutation distribution from paper")
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
             args.paper_mutation,
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
              paper_mutation=args.paper_mutation,
              outfile=f"logs/{current_date}_{args.outfile}.csv",
              only_max=args.only_max)
