#!/usr/bin/env python3

from multiprocessing import Pool, cpu_count
from pickle import load
from sys import argv

import imageio
import numpy as np
import pandas as pd
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm

from network import Network
from plot_final import dir_to_name
from postevolution import get_best_ind
from tst_environment import TstEnvironment


def examine_agent(model_dir,
                  num_episodes=1000,
                  dfnn_path='logs/good_population.pickle',
                  output_dir='logs/rl_evaluation',
                  position=0):
    if 'nodfnn' in model_dir:
        dfnn = None
        pretrain = False
        update_all_seen = False
        annotate = False
    else:
        with open(dfnn_path, 'rb') as f:
            dfnn = get_best_ind(load(f))['network']
        pretrain = 'pretrain' in model_dir
        update_all_seen = 'update_all' in model_dir
        annotate = 'annotate' in model_dir
    double_channel = 'double' in model_dir

    l_mean_reward = []
    l_std_reward = []
    l_non_timeout = []
    l_goal = []
    l_step_success = []
    l_variable_tile = []
    for i in tqdm(range(50), position=position):
        if dfnn is not None:
            dfnn.store_original_weights()
        env = TstEnvironment(render_mode=None, size=10,
                             dfnn=dfnn,
                             pretrain=pretrain,
                             update_all_seen=update_all_seen,
                             double_channel=double_channel,
                             no_color=not double_channel,
                             annotate=annotate,
                             )
        env = ImgObsWrapper(env)
        fn = f"{model_dir}/{i:03}"
        if 'dqn' in model_dir:
            model = DQN.load(fn, env=env)
        else:
            model = PPO.load(fn, env=env)

        mean_reward, std_reward = evaluate_policy(model, model.get_env(),
                                                  n_eval_episodes=num_episodes)
        l_mean_reward.append(mean_reward)
        l_std_reward.append(std_reward)
        l_non_timeout.append(env.unwrapped.non_timeout_count/num_episodes)
        l_goal.append(env.unwrapped.goal_count/num_episodes)
        try:
            l_step_success.append(env.unwrapped.step_success_count/env.unwrapped.goal_count)
        except ZeroDivisionError:
            l_step_success.append(0)
        l_variable_tile.append(env.unwrapped.variable_tile_count/num_episodes)
    df = pd.DataFrame(data={
        'mean_reward': l_mean_reward,
        'std_reward': l_std_reward,
        'non_timeout_share': l_non_timeout,
        'goal_share': l_goal,
        'steps_per_successful_epsiode': l_step_success,
        'variable_tiles_per_episode': l_variable_tile
    })
    df['name'] = dir_to_name(model_dir)
    df['num_episodes'] = num_episodes
    df.to_csv(f'{output_dir}/{model_dir.split("/")[-1]}.csv',
              index=False)

if __name__ == '__main__':
    model_paths = [
        'fox_ppo_pretrain_annotate_double_1712401463/004',
        'fox_ppo_pretrain_annotate_single_1712401078/030',
        'fox_ppo_pretrain_double_1712464516/041',
        'fox_ppo_pretrain_single_1712401541/041',
        'fox_ppo_update_all_annotate_double_1712581267/020',
        'fox_ppo_update_all_annotate_single_1712580285/023',
        'fox_ppo_update_all_double_1712580146/014',
        'fox_ppo_update_all_single_1712580285/006',
        'fox_ppo_update_one_annotate_double_1712401471/030',
        'fox_ppo_update_one_annotate_single_1712401351/014',
        'fox_ppo_update_one_double_1712465135/045',
        'fox_ppo_update_one_single_1712401564/030',
        'fox_nodfnn_ppo_update_one_single_1712218071/033',
        'fox_nodfnn_ppo_update_one_double_1712129555/025',
        'fox_dqn_pretrain_annotate_double_1712401449/020',
        'fox_dqn_pretrain_annotate_single_1712305600/002',
        'fox_dqn_pretrain_double_1712439339/018',
        'fox_dqn_pretrain_single_1712401532/031',
        'fox_dqn_update_all_annotate_double_1712581267/021',
        'fox_dqn_update_all_annotate_single_1712581267/045',
        'fox_dqn_update_all_double_1712580146/037',
        'fox_dqn_update_all_single_1712580146/023',
        'fox_dqn_update_one_annotate_double_1712401458/042',
        'fox_dqn_update_one_annotate_single_1712242693/046',
        'fox_dqn_update_one_double_1712439499/002',
        'fox_dqn_update_one_single_1712401535/011',
        'fox_nodfnn_dqn_update_one_double_1712065333/008',
        'fox_nodfnn_dqn_update_one_single_1712046029/029'
    ]
    
    dfnn_path = 'logs/good_population.pickle'
    for path in model_paths:
        if 'nodfnn' in path:
            dfnn = None
            pretrain = False
            update_all_seen = False
            annotate = False
        else:
            with open(dfnn_path, 'rb') as f:
                dfnn = get_best_ind(load(f))['network']
            pretrain = 'pretrain' in path
            update_all_seen = 'update_all' in path
            annotate = 'annotate' in path
        double_channel = 'double' in path
        if dfnn is not None:
            dfnn.store_original_weights()
        env = TstEnvironment(render_mode="rgb_array", size=10,
                             dfnn=dfnn,
                             pretrain=pretrain,
                             update_all_seen=update_all_seen,
                             double_channel=double_channel,
                             no_color=not double_channel,
                             annotate=annotate,
                             )
        env = ImgObsWrapper(env)
        fpath = f"logs/models/{path}"
        if 'dqn' in path:
            model = DQN.load(fpath, env=env)
        else:
            model = PPO.load(fpath, env=env)

        images = []
        obs = model.env.reset()
        img = model.env.render(mode="rgb_array")
        for i in range(500):
            images.append(img)
            action, _ = model.predict(obs)
            obs, _, _, _ = model.env.step(action)
            img = model.env.render(mode="rgb_array")

        imageio.mimsave(f"logs/gifs/{path[:-4]}_{path[-3:]}.gif",
                        [np.array(img) for img in images],
                        fps=5)


    """
    def _worker_wrapper(arg):
        return examine_agent(*arg)

    dirs = [
        'logs/models/fox_ppo_update_one_double_1712465135',
        'logs/models/fox_dqn_update_all_double_1712580146',
        'logs/models/fox_dqn_update_one_double_1712439499',
    ]
    worker_args = [
        (dirs[i],
         1000,
         'logs/good_population.pickle',
         'logs/rl_evaluation',
         i) for i in range(len(dirs))
    ]
    #with Pool(2) as pool:
    #    _ = pool.map(_worker_wrapper, worker_args[:2])
    #with Pool(2) as pool:
    #    _ = pool.map(_worker_wrapper, worker_args[2:])
    [examine_agent(dir) for dir in dirs]
    """

"""
the_env = model.get_env()
obs = the_env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = the_env.step(action)
    if dones[0]:
        print(rewards[0])
    the_env.render("human")
"""
