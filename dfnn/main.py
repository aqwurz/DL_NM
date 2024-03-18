#!/usr/bin/env python3

import cProfile
import os
from multiprocessing import Pool, cpu_count

import numpy as np

from environment import decode_food_id
from pnsga import clone_individual, pnsga, set_objective_value


def train(ind, iterations=30, lifetimes=4, season_length=5,
          num_foods=8, profile=False):
    """Performs the experiment on an individual.

    Args:
        ind (dict): The dictionary representing the individual.
        iterations (int): How many days the environment lasts for.
            Defaults to 30.
        lifetimes (int): How many lifetimes to evaluate the individual for.
            Defaults to 4.
        season_length (int): How many days a season lasts for.
            Defaults to 5.
        num_foods (int): How many foods to include in each environment.
            Defaults to 8.
        profile (bool): Whether or not to do cProfile profiling.
            Defaults to False.

    Returns:
        dict: The trained individual.
    """
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    individual = clone_individual(ind)
    eat_vector = np.zeros((lifetimes*iterations*num_foods,), dtype=bool)
    individual['network'].store_original_weights()
    food_count = 0
    good_count = 0
    bad_count = 0
    inputs = np.zeros((5,))
    for i in range(lifetimes):
        summer = False
        winter = True
        prev_summer = 0
        prev_winter = 0
        env = individual['envs'][i]
        for j in range(iterations):
            if j % season_length == 0:
                summer = not summer
                winter = not winter
            for k in range(num_foods):
                food_count += 1
                food = env.presentation_order[j][k]
                inputs[:3] = decode_food_id(food)
                inputs[3] = 0
                inputs[4] = 0
                outputs = individual['network'].forward(inputs)
                ate_summer = summer and outputs[0] > 0
                ate_winter = winter and outputs[1] > 0
                prev_summer = 1 if food in env.foods_summer else -1
                prev_winter = 1 if food in env.foods_winter else -1
                feedback_summer = prev_summer if ate_summer else 0
                feedback_winter = prev_winter if ate_winter else 0
                if ate_summer or ate_winter:
                    eat_vector[(i*iterations+j) * num_foods + k] = True
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
                individual['network'].update_weights(
                    inputs,
                    feedback_summer if summer else feedback_winter,
                    summer
                )
        individual['network'].load_original_weights()
    individual['eat_vector'] = eat_vector
    fitness = 0.5 + (good_count - bad_count)/food_count
    set_objective_value(individual, 'performance', fitness)
    individual['network'].convert_activations()
    if profile:
        pr.disable()
        pr.print_stats(sort='tottime')
    return individual


def _worker_wrapper(arg):
    """Wrapper function."""
    return pnsga(*arg)


if __name__ == '__main__':
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    from datetime import date
    objectives = {
        "performance": 1.00,
        "behavioral_diversity": 1.00,
        "connection_cost_n": 0.75
    }
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num-execs", "-e", type=int, default=1,
                        help="How many parallel executions to perform")
    parser.add_argument("--num-cores", "-c", type=int, default=cpu_count(),
                        help="The number of cores to use for training \
                        (unused if --num-execs is passed)")
    parser.add_argument("--pop-size", "-p", type=int, default=400,
                        help="The number of individuals per generation")
    parser.add_argument("--aleat", "-a", type=int, default=1,
                        help="Population multiplier for initialization")
    parser.add_argument("--num-generations", "-g", type=int, default=20000,
                        help="The number of generations to run PNSGA for")
    parser.add_argument("--p-toggle", "-pt", type=float, default=0.20,
                        help="Probability of toggling connection in mutation")
    parser.add_argument("--p-reassign", "-pr", type=float, default=0.15,
                        help="Probability of switching two weights in \
                        mutation")
    parser.add_argument("--p-biaschange", "-pb", type=float, default=0.10,
                        help="Probability of changing bias in mutation")
    parser.add_argument("--p-weightchange", "-pw", type=float, default=-1,
                        help="Probability of changing weight in mutation \
                        (a value of -1 is resolved to 2/n)")
    parser.add_argument("--p-nudge", "-pn", type=float, default=0.00,
                        help="Probability of changing neuron position \
                        in mutation")
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
    outfile = f"logs/{current_date}_{args.outfile}.csv" if "/" not in args.outfile else f"logs/{args.outfile}.csv"
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
             args.aleat,
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
              num_cores=args.num_cores,
              aleat=args.aleat,
              p_nudge=args.p_nudge,
              outfile=outfile,
              only_max=args.only_max)
