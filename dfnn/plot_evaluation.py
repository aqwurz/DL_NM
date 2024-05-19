#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_final import dir_to_name

sns.set_theme()
sns.set_palette(['#1b9e77', '#d95f02', '#7570b3', '#000000'])

dqn = False

if dqn:
    dirs = [
        "logs/rl_evaluation/fox_nodfnn_dqn_update_one_double_1712065333.csv",
        "logs/rl_evaluation/fox_nodfnn_dqn_update_one_single_1712046029.csv",
        "logs/rl_evaluation/fox_dqn_pretrain_annotate_double_1712401449.csv",
        "logs/rl_evaluation/fox_dqn_pretrain_annotate_single_1712305600.csv",
        "logs/rl_evaluation/fox_dqn_pretrain_double_1712439339.csv",
        "logs/rl_evaluation/fox_dqn_pretrain_single_1712401532.csv",
        "logs/rl_evaluation/fox_dqn_update_all_annotate_double_1712581267.csv",
        "logs/rl_evaluation/fox_dqn_update_all_annotate_single_1712581267.csv",
        "logs/rl_evaluation/fox_dqn_update_all_double_1712580146.csv",
        "logs/rl_evaluation/fox_dqn_update_all_single_1712580146.csv",
        "logs/rl_evaluation/fox_dqn_update_one_annotate_double_1712401458.csv",
        "logs/rl_evaluation/fox_dqn_update_one_annotate_single_1712242693.csv",
        "logs/rl_evaluation/fox_ppo_update_one_double_1712465135.csv",
        "logs/rl_evaluation/fox_dqn_update_one_single_1712401535.csv",
    ]
else:
    dirs = [
        "logs/rl_evaluation/fox_nodfnn_ppo_update_one_double_1712129555.csv",
        "logs/rl_evaluation/fox_nodfnn_ppo_update_one_single_1712218071.csv",
        "logs/rl_evaluation/fox_ppo_pretrain_annotate_double_1712401463.csv",
        "logs/rl_evaluation/fox_ppo_pretrain_annotate_single_1712401078.csv",
        "logs/rl_evaluation/fox_ppo_pretrain_double_1712464516.csv",
        "logs/rl_evaluation/fox_ppo_pretrain_single_1712401541.csv",
        "logs/rl_evaluation/fox_ppo_update_all_annotate_double_1712581267.csv",
        "logs/rl_evaluation/fox_ppo_update_all_annotate_single_1712580285.csv",
        "logs/rl_evaluation/fox_ppo_update_all_double_1712580146.csv",
        "logs/rl_evaluation/fox_ppo_update_all_single_1712580285.csv",
        "logs/rl_evaluation/fox_ppo_update_one_annotate_double_1712401471.csv",
        "logs/rl_evaluation/fox_ppo_update_one_annotate_single_1712401351.csv",
        "logs/rl_evaluation/fox_ppo_update_one_double_1712465135.csv",
        "logs/rl_evaluation/fox_ppo_update_one_single_1712401564.csv",
    ]


def colourer(dir):
    if 'nodfnn' in dir:
        return '#808080'
    elif 'pretrain' in dir:
        return '#1b9e77'
    elif 'update_all' in dir:
        return '#d95f02'
    return '#7570b3'


palette = {dir_to_name(dir): colourer(dir) for dir in dirs}

df_list = []
for dir in dirs:
    print(dir_to_name(dir))
for dir in dirs:
    df = pd.read_csv(dir)
    print(df.iloc[0]['name'])
    print(df.mean(numeric_only=True))
    print(df.std(numeric_only=True))
    print()
    df_list.append(df)

big_df = pd.concat(df_list, ignore_index=True)

for x in ['mean_reward',
          'non_timeout_share',
          'goal_share',
          'steps_per_successful_epsiode',
          'variable_tiles_per_episode']:
    fig, ax = plt.subplots(figsize=(10, 4))
    g = sns.boxplot(data=big_df,
                    y='name',
                    x=x,
                    hue='name',
                    palette=palette,
                    dodge=False,
                    ax=ax)
    g.get_legend().remove()
    plt.xlabel(x[0].upper()+x[1:].replace('_', ' '))
    plt.tight_layout()
    plt.show()
