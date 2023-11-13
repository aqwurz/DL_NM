#!/usr/bin/env python3

import numpy as np

from network import *
from pnsga import *
from multiprocessing import cpu_count


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
    parser.add_argument("--pop-size", "-p", type=int, default=400,
                        help="The number of individuals per generation")
    parser.add_argument("--num-generations", "-g", type=int, default=20000,
                        help="The number of generations to run PNSGA for")
    parser.add_argument("--num-selected", "-s", type=int, default=50,
                        help="How many individuals to select in tournament \
                        selection")
    parser.add_argument("--paper-mutation", action="store_true",
                        help="Which mutation distributions to use")
    parser.add_argument("--num-cores", "-c", type=int, default=cpu_count(),
                        help="The number of cores to use for training")
    parser.add_argument("--objectives", "-m", nargs="+",
                        default=["performance", "behavioral_diversity",
                                 "connection_cost_n"],
                        help="Which objectives to use")
    parser.add_argument("--outfile", "-o", type=str, required=True,
                        help="Output file name comment")
    args = parser.parse_args()
    selected_objectives = {m: objectives[m] for m in args.objectives}
    current_date = str(date.today()).replace("-", "")
    pnsga(train,
          selected_objectives,
          pop_size=args.pop_size,
          num_generations=args.num_generations,
          num_selected=args.num_selected,
          num_cores=args.num_cores,
          paper_mutation=args.paper_mutation,
          outfile=f"logs/{current_date}_{args.outfile}.csv")
