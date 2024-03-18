import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from pnsga import initialize_pop, pnsga, new_generation
from environment import Environment, decode_food_id
from main import train

objectives = {
    "performance": 1.00,
    "behavioral_diversity": 1.00,
}


def test_full_run_neo(num_selected=50):
    pop_size = 400
    num_generations = 1000
    num_cores = 1
    trainer = train
    p_toggle = 0.20
    p_reassign = 0.15
    p_biaschange = 0.10
    p_weightchange = -1
    p_nudge = 0.00
    position = 0
    layer_config_config = [5, 12, 8, 6, 2]
    layer_config = [
        np.array([
            (j-layer_config_config[i]/2-0.5, float(i))
            for j in range(layer_config_config[i])])
        for i in range(len(layer_config_config))]
    source_config = [(-3.0, 2.0), (3.0, 2.0)]
    obj_indexing = list(objectives.keys())
    P = initialize_pop(layer_config, source_config,
                       objectives, pop_size=pop_size)
    for ind in P:
        ind['mapping'] = {m: obj_indexing.index(m) for m in objectives.keys()}
    for i in tqdm(range(num_generations), position=position):
        P = new_generation(P, objectives, trainer,
                           num_cores=num_cores,
                           p_toggle=p_toggle,
                           p_reassign=p_reassign,
                           p_biaschange=p_biaschange,
                           p_weightchange=p_weightchange,
                           p_nudge=p_nudge,
                           )
        if True:
            parents = [ind for ind in P if 'parent' in ind and ind['parent']]
            selecteds = [ind for ind in P
                         if 'selected' in ind and ind['selected']]
            others = [ind for ind in P
                      if not ('parent' in ind and ind['parent'])
                      and not ('selected' in ind and ind['selected'])]
            plt.scatter(x=[ind['performance'] for ind in parents],
                        y=[ind['behavioral_diversity'] for ind in parents],
                        marker='+',
                        )
            plt.scatter(x=[ind['performance'] for ind in selecteds],
                        y=[ind['behavioral_diversity'] for ind in selecteds],
                        marker='P',
                        )
            plt.scatter(x=[ind['performance'] for ind in others],
                        y=[ind['behavioral_diversity'] for ind in others],
                        s=5,
                        )
            plt.axvline(x=np.mean([ind['performance'] for ind in P]))
            """
            ax = plt.gca()
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            """
            plt.show()
            perfs = [ind['performance'] for ind in P]
            behs = [ind['behavioral_diversity'] for ind in P]
            print(min(perfs), max(perfs))
            print(min(behs), max(behs))


def profile(single_core=False, num_generations=20,
            pop_size=400):
    objectives = {
        "performance": 1.00,
        "behavioral_diversity": 1.00,
        "connection_cost_n": 0.75
    }
    if single_core:
        pnsga(train, objectives,
              pop_size=pop_size,
              profile=True,
              num_generations=num_generations,
              num_cores=1
              )
    else:
        pnsga(train, objectives,
              pop_size=pop_size,
              profile=True,
              num_generations=num_generations,
              )


def profile_just_train():
    pop_size = 40
    layer_config_config = [5, 12, 8, 6, 2]
    layer_config = [
        np.array([
            (j-layer_config_config[i]/2-0.5, float(i))
            for j in range(layer_config_config[i])])
        for i in range(len(layer_config_config))]
    source_config = [(-3.0, 2.0), (3.0, 2.0)]
    obj_indexing = list(objectives.keys())
    P = initialize_pop(layer_config, source_config,
                       objectives, pop_size=pop_size)
    for ind in P:
        ind['mapping'] = {m: obj_indexing.index(m) for m in objectives.keys()}
    envs = [Environment() for _ in range(4)]
    for ind in P:
        ind['envs'] = envs
    train(P[0], profile=True)


def inspect_eating(filename, index):
    with open(filename, 'rb') as f:
        pop = pickle.load(f)
    for i in range(4):
        env = pop[index]['envs'][i]
        print("new env:", env.foods_summer, env.foods_winter)
        for j in range(30):
            pres = env.presentation_order[j]
            print(sorted(
                pres[pop[index]['eat_vector'][(i*30+j)*8:(i*30+j+1)*8]]
            ))
            if j % 5 == 0:
                stuff = np.array([
                    decode_food_id(food, env.switch) for food in pres
                ])[pop[index]['eat_vector'][(i*30+j)*8:(i*30+j+1)*8]]
                print(stuff)


def get_population(fn):
    with open(fn, 'rb') as f:
        pop = pickle.load(f)
    return pop


if __name__ == '__main__':
    profile(single_core=True, num_generations=10)
    # profile_just_train()
    # test_full_run_neo(num_selected=100)
    # test_nds()
    # inspect_eating(argv[1], int(argv[2]))
