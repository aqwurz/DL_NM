#!/usr/bin/env python3


from sys import argv

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu

from tbparse import SummaryReader


def dir_to_name(dir):
    name = ""
    if "nodfnn" in dir:
        name = "Control, "
        if "dqn" in dir:
            name += "DQN, "
        elif "ppo" in dir:
            name += "PPO, "
        if "single" in dir:
            name += "only tile type"
        elif "double" in dir:
            name += "full information"
    else:
        if "dqn" in dir:
            name += "DQN, "
        elif "ppo" in dir:
            name += "PPO, "
        if "pretrain" in dir:
            name += "pretrained, "
        elif "update_all" in dir:
            name += "trained on FOV, "
        elif "update_one" in dir:
            name += "trained on next tile, "
        if "single" in dir:
            name += "single channel, "
        elif "double" in dir:
            name += "double channel, "
        if "annotate" in dir:
            name += "annotated"
        else:
            name += "raw"
    return name


def training_comparer():
    sns.set_theme()
    sns.set_palette(['#1b9e77', '#d95f02', '#7570b3', '#000000'])
    dirs = [
        'fox_ppo_pretrain_annotate_double_1712401463',
        'fox_ppo_pretrain_annotate_single_1712401078',
        'fox_ppo_pretrain_double_1712464516',
        'fox_ppo_pretrain_single_1712401541',
        'fox_ppo_update_all_annotate_double_1712581267',
        'fox_ppo_update_all_annotate_single_1712580285',
        'fox_ppo_update_all_double_1712580146',
        'fox_ppo_update_all_single_1712580285',
        'fox_ppo_update_one_annotate_double_1712401471',
        'fox_ppo_update_one_annotate_single_1712401351',
        'fox_ppo_update_one_double_1712465135',
        'fox_ppo_update_one_single_1712401564',
    ]
    data = {
        'Annotated double': {'Pretrained': None, 'Trained on FOV': None, 'Trained on next tile': None},
        'Annotated single': {'Pretrained': None, 'Trained on FOV': None, 'Trained on next tile': None},
        'Raw double': {'Pretrained': None, 'Trained on FOV': None, 'Trained on next tile': None},
        'Raw single': {'Pretrained': None, 'Trained on FOV': None, 'Trained on next tile': None},
    }
    for dir in dirs:
        ds = []
        final_values = []
        for i in range(50):
            r = SummaryReader(f"logs/tensorboard/{dir}/{i:03}_1", pivot=True)
            raw_ds = r.scalars[['step', 'rollout/ep_rew_mean']]
            final_values.append(raw_ds.iloc[-1]['rollout/ep_rew_mean'])
            ds.append(raw_ds)
        df = pd.DataFrame(pd.concat(ds))
        df['name'] = dir_to_name(dir)
        if 'annotate_double' in dir:
            selected_data = data['Annotated double']
            df['Channel mode'] = 'Annotated double'
        elif 'annotate_single' in dir:
            selected_data = data['Annotated single']
            df['Channel mode'] = 'Annotated single'
        elif 'double' in dir:
            selected_data = data['Raw double']
            df['Channel mode'] = 'Raw double'
        elif 'single' in dir:
            selected_data = data['Raw single']
            df['Channel mode'] = 'Raw single'
        if 'pretrain' in dir:
            df['Training mode'] = 'Pretrained'
            selected_data['Pretrained'] = df
        elif 'update_all' in dir:
            df['Training mode'] = 'Trained on FOV'
            selected_data['Trained on FOV'] = df
        elif 'update_one' in dir:
            df['Training mode'] = 'Trained on next tile'
            selected_data['Trained on next tile'] = df

    df_sig_combined_list = []
    for channel_mode, value in data.items():
        for pair in [
            ['Pretrained', 'Trained on FOV'],
            ['Pretrained', 'Trained on next tile'],
            ['Trained on FOV', 'Trained on next tile']
        ]:
            df1 = value[pair[0]]
            df2 = value[pair[1]]
            significant_steps = []
            for i in range(2048, 500000, 2048):
                _, p = mannwhitneyu(
                    df1.loc[df1['step'] == i]['rollout/ep_rew_mean'],
                    df2.loc[df2['step'] == i]['rollout/ep_rew_mean'],
                    alternative='greater'
                )
                pt = 0.001
                if p < pt:
                    significant_steps.append(i)
            df_sig = pd.DataFrame(data={'step': significant_steps})
            df_sig['Channel mode'] = channel_mode
            df_sig['Comparison'] = f"{pair[0]} -- {pair[1]}"
            df_sig_combined_list.append(df_sig)
    df_sig_combined = pd.concat(df_sig_combined_list, ignore_index=True)
    print(df_sig_combined)
    fig, ax = plt.subplots(figsize=(12, 5))
    g = sns.stripplot(data=df_sig_combined,
                      x='step',
                      y='Channel mode',
                      hue="Comparison",
                      marker="s",
                      edgecolor=None,
                      dodge=True,
                      jitter=False,
                      ax=ax)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    )
    plt.tight_layout()
    plt.show()


def main_plotter(dqn=False,
                 flat=False,
                 many_plots=True,
                 stat_analysis=False):
    sns.set_theme()
    # sns.set_palette(['#b2df8a', '#1f78b4', '#a6cee3', '#000000'])
    sns.set_palette(['#1b9e77', '#d95f02', '#7570b3', '#000000'])

    dqn = True
    flat = False
    many_plots = False
    stat_analysis = False

    if not dqn:
        dirs = [
            'fox_ppo_pretrain_annotate_double_1712401463',
            'fox_ppo_pretrain_annotate_single_1712401078',
            'fox_ppo_pretrain_double_1712464516',
            'fox_ppo_pretrain_single_1712401541',
            'fox_ppo_update_all_annotate_double_1712581267',
            'fox_ppo_update_all_annotate_single_1712580285',
            'fox_ppo_update_all_double_1712580146',
            'fox_ppo_update_all_single_1712580285',
            'fox_ppo_update_one_annotate_double_1712401471',
            'fox_ppo_update_one_annotate_single_1712401351',
            'fox_ppo_update_one_double_1712465135',
            'fox_ppo_update_one_single_1712401564',
            'fox_nodfnn_ppo_update_one_single_1712218071',
            'fox_nodfnn_ppo_update_one_double_1712129555'
        ]
    else:
        dirs = [
            'fox_dqn_pretrain_annotate_double_1712401449',
            'fox_dqn_pretrain_annotate_single_1712305600',
            'fox_dqn_pretrain_double_1712439339',
            'fox_dqn_pretrain_single_1712401532',
            'fox_dqn_update_all_annotate_double_1712581267',
            'fox_dqn_update_all_annotate_single_1712581267',
            'fox_dqn_update_all_double_1712580146',
            'fox_dqn_update_all_single_1712580146',
            'fox_dqn_update_one_annotate_double_1712401458',
            'fox_dqn_update_one_annotate_single_1712242693',
            'fox_dqn_update_one_double_1712439499',
            'fox_dqn_update_one_single_1712401535',
            'fox_nodfnn_dqn_update_one_double_1712065333',
            'fox_nodfnn_dqn_update_one_single_1712046029'
        ]
    if many_plots:
        channel_modes = ["Annotated double-channel",
                         "Annotated single-channel",
                         "Raw double-channel",
                         "Raw single-channel"]
        indices = [
            [0, 4, 8, 12, 13],
            [1, 5, 9, 12, 13],
            [2, 6, 10, 12, 13],
            [3, 7, 11, 12, 13]
        ]
    else:
        channel_modes = ["All channel modes"]
        indices = [range(14)]
    c = 0
    for i_l in indices:
        selected_dirs = [dirs[i] for i in i_l]
        dfs = []
        for dir in selected_dirs:
            ds = []
            final_values = []
            for i in range(50):
                r = SummaryReader(f"logs/tensorboard/{dir}/{i:03}_1", pivot=True)
                raw_ds = r.scalars[['step', 'rollout/ep_rew_mean']]
                #ds[-1]['rolling'] = ds[-1]['rollout/ep_rew_mean'].rolling(50).mean()
                final_values.append(raw_ds.iloc[-1]['rollout/ep_rew_mean'])
                if dqn:
                    closest = [raw_ds.loc[(raw_ds['step'] - i).abs().idxmin()]['rollout/ep_rew_mean'] for i in range(2048, 500000, 2048)]
                    new_ds = pd.DataFrame(zip(range(2048, 500000, 2048), closest), columns=['step', 'closest_rew_mean'])
                    ds.append(new_ds)
                else:
                    ds.append(raw_ds)
            print(dir)
            print(final_values.index(max(final_values)), max(final_values))
            #print(np.mean(final_values))
            #print(np.std(final_values, ddof=1))
            df = pd.DataFrame(pd.concat(ds))
            df['name'] = dir
            if "pretrain" in dir:
                df['Training mode'] = "Pretrained"
            elif "update_all" in dir:
                df['Training mode'] = "Trained on FOV"
            elif "nodfnn" not in dir:
                df['Training mode'] = "Trained on next tile"
            else:
                df['Training mode'] = "Control"
            if "nodfnn" in dir:
                df['Channel mode'] = "Control, full" if "double" in dir else "Control, only type"
            elif many_plots:
                df['Channel mode'] = "(see title)"
            elif "annotate_double" in dir:
                df['Channel mode'] = "Annotated double"
            elif "annotate_single" in dir:
                df['Channel mode'] = "Annotated single"
            elif "double" in dir:
                df['Channel mode'] = "Raw double"
            elif "single" in dir:
                df['Channel mode'] = "Raw single"
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        if stat_analysis:
            nodfnn_single = combined[combined["name"] == 'fox_nodfnn_dqn_update_one_single_1712046029']
            nodfnn_double = combined[combined["name"] == 'fox_nodfnn_dqn_update_one_double_1712065333']
            df_sig_combined_list = []
            for dir in dirs:
                per_dir = combined.loc[combined['name'] == dir]
                if "nodfnn" not in dir:
                    significant_steps1 = []
                    significant_steps2 = []
                    for i in range(2048, 500000, 2048):
                        _, p1 = mannwhitneyu(
                                        per_dir.loc[per_dir['step'] == i]['closest_rew_mean'],
                                        nodfnn_double.loc[nodfnn_double['step'] == i]['closest_rew_mean'],
                                        alternative='greater'
                                        )
                        _, p2 = mannwhitneyu(
                                        per_dir.loc[per_dir['step'] == i]['closest_rew_mean'],
                                        nodfnn_single.loc[nodfnn_single['step'] == i]['closest_rew_mean'],
                                        alternative='greater'
                                        )
                        pt = 0.001
                        if p1 < pt:
                            significant_steps1.append(i)
                        if p2 < pt:
                            significant_steps2.append(i)
                    if len(significant_steps1) == 0:
                        significant_steps1.append(None)
                    if len(significant_steps2) == 0:
                        significant_steps2.append(None)
                    df_sig1 = pd.DataFrame(data={'step': significant_steps1})
                    df_sig1['name'] = dir_to_name(dir)
                    df_sig1['Compared to'] = 'Control, full information'
                    df_sig2 = pd.DataFrame(data={'step': significant_steps2})
                    df_sig2['name'] = dir_to_name(dir)
                    df_sig2['Compared to'] = 'Control, only tile type'
                    df_sig_combined_list.append(df_sig1)
                    df_sig_combined_list.append(df_sig2)
                #last_step = per_dir[per_dir['step'] == per_dir['step'].max()]
                #print(last_step['rollout/ep_rew_mean'].mean())
                #print(last_step['rollout/ep_rew_mean'].std())
            df_sig_combined = pd.concat(df_sig_combined_list)
            print(df_sig_combined)
            g = sns.relplot(data=df_sig_combined,
                            x='step',
                            y='name',
                            col="Compared to",
                            marker="s",
                            edgecolor=None)
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=(10, 3) if flat else (10, 7))
            l = sns.lineplot(data=combined,
                             x='step',
                             y='closest_rew_mean' if dqn else 'rollout/ep_rew_mean',
                             hue='Training mode',
                             style='Channel mode',
                             ax=ax,
                             )
            plt.title(channel_modes[c])
            #ax = plt.gca()
            if dqn:
                ax.set_ylim([0.03, 0.27])
            else:
                ax.set_ylim([0.05, 0.65])
            if flat:
                sns.move_legend(l, "upper left", bbox_to_anchor=(1, 1))
            """
            fig = plt.gcf()
            #fig.set_size_inches(10, 7, forward=True)
            fig.set_size_inches(10, 3, forward=True)
            """
            if flat:
                plt.tight_layout()
            plt.show()
            c += 1


if __name__ == '__main__':
    #main_plotter()
    training_comparer()
