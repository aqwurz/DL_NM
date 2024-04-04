#!/usr/bin/env python3

from pickle import load
from time import time

import gym
import numpy as np
import torch
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.wrappers import ImgObsWrapper, NoDeath
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from network import Network
from postevolution import get_best_ind
from tst_environment import TstEnvironment


# from https://minigrid.farama.org/content/training/
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space,
                 features_dim: int = 512,
                 normalized_image: bool = False,
                 ) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(
                observation_space.sample()[None]
            ).float()).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def main(filename,
         use_ppo=False,
         update_mode="update_one",
         channel_mode="double",
         name="test",
         num_runs=50,
         num_steps=int(5e5),
         render=False,
         ):
    run_name = name

    if filename != "":
        with open(filename, 'rb') as f:
            dfnn = get_best_ind(load(f))['network']
    else:
        dfnn = None
        run_name += "_nodfnn"

    if use_ppo:
        model_class = PPO
        run_name += "_ppo"
    else:
        model_class = DQN
        run_name += "_dqn"

    run_name += f"_{update_mode}"
    if update_mode == "pretrain":
        pretrain = True
        update_all_seen = False
    elif update_mode == "update_all":
        pretrain = False
        update_all_seen = True
    elif update_mode == "update_one":
        pretrain = False
        update_all_seen = False
    else:
        raise ValueError("wrong update mode provided")

    run_name += f"_{channel_mode}"
    if channel_mode == "single":
        double_channel = False
        annotate = False
    elif channel_mode == "double":
        double_channel = True
        annotate = False
    elif channel_mode == "annotate_single":
        double_channel = False
        annotate = True
    elif channel_mode == "annotate_double":
        double_channel = True
        annotate = True
    else:
        raise ValueError("wrong channel mode provided")

    env = TstEnvironment(render_mode=("human" if render else None), size=10,
                         dfnn=dfnn,
                         pretrain=pretrain,
                         update_all_seen=update_all_seen,
                         double_channel=double_channel,
                         no_color=not double_channel,
                         annotate=annotate,
                         )
    # env = NoDeath(env, no_death_types=("lava", "mud", "water", "wall", "slope", "hill"))
    if dfnn is not None:
        dfnn.store_original_weights()
    # from https://minigrid.farama.org/content/training/
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
        )
    )
    t = int(time())
    for i in range(num_runs):
        model = model_class("MlpPolicy", ImgObsWrapper(env),
                            policy_kwargs=policy_kwargs,
                            verbose=0,
                            tensorboard_log=f'./logs/tensorboard/{run_name}_{t}/',
                            )
        model.learn(total_timesteps=num_steps,
                    tb_log_name=f"{i:03}",
                    progress_bar=True,
                    )
        model.save(f"./logs/models/{run_name}_{t}/{i:03}")


if __name__ == '__main__':
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    from datetime import date

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dfnn", "-d", type=str,
                        help="The pickle file of the DFNN to use with the run.")
    parser.add_argument("--dqn", "-D", action="store_true",
                        help="Selects DQN as the navigator network.")
    parser.add_argument("--ppo", "-P", action="store_true",
                        help="Selects PPO as the navigator network.")
    parser.add_argument("--update-mode", "-u", default="update_one",
                        help="""What weight update mode to use.
Available modes:
  - pretrain:   The DFNN is fully trained on the environment before any steps
                are taken.
  - update_all: The DFNN starts untrained, and weights update for every seen
                tile.
  - update_one: The DFNN starts untrained, and weights update for only the next
                tile.
                        """)
    parser.add_argument("--channel-mode", "-c", default="double",
                        help="""What inter-network connection mode to use.
Available modes:
  - single:   Only the current season's safety values are used.
              If this mode is used without a DFNN, colour information is
              blanked.
  - double:   Both seasons' safety values are used.
  - annotate: The current season's values are provided alongside tile types.
                        """)
    parser.add_argument("--name", "-N", default="test",
                        help="Gives a name to the run (added to descriptor).")
    args = parser.parse_args()
    main(args.dfnn,
         use_ppo=args.ppo and not args.dqn,
         update_mode=args.update_mode,
         channel_mode=args.channel_mode,
         name=args.name,
         render=False,
         )
