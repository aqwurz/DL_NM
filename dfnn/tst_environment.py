#!/usr/bin/env python3

import contextlib
from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import MiniGridEnv
from minigrid.core.world_object import (Goal, Grass, Hill, Lava, Mud, Slope,
                                        Water)

t_water = 0.4
t_grass = 0.6
h_mud = 0.7
t_slope = 0.7


class TstEnvironment(MiniGridEnv):
    def __init__(self, size=8,
                 agent_start_pos=(1, 1),
                 agent_start_dir=0,
                 max_steps: int | None = None,
                 dfnn=None,
                 no_color=False,
                 pretrain=False,
                 update_all_seen=False,
                 double_channel=False,
                 annotate=False,
                 eval_name=None,
                 **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=256,
            **kwargs
        )
        self.switch_time = 5
        self.switch_count = 0
        self.mapping = None
        self.summer = True
        self.dfnn = dfnn
        if self.dfnn is not None:
            self.dfnn.store_original_weights()
        self.no_color = no_color
        self.pretrain = pretrain
        self.update_all_seen = update_all_seen
        self.double_channel = double_channel
        self.annotate = annotate
        self.eval_name = eval_name
        self.non_timeout_count = 0
        self.goal_count = 0
        self.step_success_count = 0
        self.variable_tile_count = 0

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # create world
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        # place terrain
        terrain_map = random_interpolation(width - 2, height - 2)
        humidity_map = random_interpolation(width - 2, height - 2)
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if terrain_map[i-1, j-1] < t_water:
                    self.grid.set(i, j, Water())
                elif terrain_map[i-1, j-1] < t_grass:
                    if humidity_map[i-1, j-1] > h_mud:
                        self.grid.set(i, j, Mud())
                    else:
                        self.grid.set(i, j, Grass())
                elif terrain_map[i-1, j-1] < t_slope:
                    self.grid.set(i, j, Slope())
                else:
                    self.grid.set(i, j, Hill())
        for _ in range(np.random.randint(0, 3)):
            ril = np.random.randint(1, width - 1)
            rjl = np.random.randint(1, height - 1)
            self.grid.set(ril, rjl, Lava())
        # place goal
        rig = np.random.randint(1, width - 1)
        rjg = np.random.randint(1, height - 1)
        attempts = 50
        while check_for_grass(rig, rjg, terrain_map, humidity_map) \
              and attempts != 0:
            rig = np.random.randint(1, width - 1)
            rjg = np.random.randint(1, height - 1)
            attempts -= 1
        self.put_obj(Goal(), rig, rjg)
        # place agent
        if self.agent_start_pos is not None:
            ri = np.random.randint(1, width - 1)
            rj = np.random.randint(1, height - 1)
            attempts = 50
            while (check_for_grass(ri, rj, terrain_map, humidity_map) \
                  or (ri == rig and rj == rjg)) \
                  and attempts != 0:
                ri = np.random.randint(1, width - 1)
                rj = np.random.randint(1, height - 1)
                attempts -= 1
            self.agent_pos = (ri, rj)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        self.mission = "grand mission"

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
                self.goal_count += 1
                self.non_timeout_count += 1
                self.step_success_count += self.step_count
            if fwd_cell is not None and fwd_cell.is_hazardous \
               and fwd_cell.type != "wall":
                terminated = True
                self.non_timeout_count += 1
            if fwd_cell.type in ["water", "mud", "slope", "hill"] \
               and not terminated:
                self.variable_tile_count += 1

        # Pick up an object
        elif action == self.actions.pickup:
            pass

        # Drop an object
        elif action == self.actions.drop:
            pass

        # Toggle/activate an object
        elif action == self.actions.toggle:
            pass

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        if self.dfnn is None:
            new_obs = obs['image']
            if self.no_color:
                new_obs[:, :, 1] = 0
        else:
            new_obs = np.zeros_like(obs['image'])
            safe_value = OBJECT_TO_IDX['empty']
            danger_value = OBJECT_TO_IDX['wall']
            goal_value = OBJECT_TO_IDX['goal']
            cells_seen = obs['image'][:, :, 0]
            raw_cells_seen, vis_mask = self.gen_obs_grid()
            for i in range(cells_seen.shape[0]):
                for j in range(cells_seen.shape[1]):
                    inp = self.mapping[cells_seen[i, j]].copy()
                    output = np.asarray(self.dfnn.forward(inp)) > 0
                    if cells_seen[i, j] == goal_value:
                        new_obs[i, j, 0] = goal_value
                        if self.double_channel:
                            new_obs[i, j, 1] = goal_value
                            if self.annotate:
                                new_obs[i, j, 2] = goal_value
                    elif self.double_channel:
                        if self.annotate:
                            new_obs[i, j, 0] = obs['image'][i, j, 0]
                            new_obs[i, j, 1] = safe_value if output[0] else danger_value
                            new_obs[i, j, 2] = safe_value if output[1] else danger_value
                        else:
                            new_obs[i, j, 0] = safe_value if output[0] else danger_value
                            new_obs[i, j, 1] = safe_value if output[1] else danger_value
                    else:
                        if self.summer:
                            result = safe_value if output[0] else danger_value
                        else:
                            result = safe_value if output[1] else danger_value
                        if self.annotate:
                            new_obs[i, j, 0] = obs['image'][i, j, 0]
                            new_obs[i, j, 1] = result
                        else:
                            new_obs[i, j, 0] = result
                    # print({v: k for k, v in OBJECT_TO_IDX.items()}[cells_seen[i, j]], output, self.summer)
                    if not self.pretrain and self.update_all_seen:
                        marked_as_safe = (self.summer and output[0]) \
                            or (not self.summer and output[1])
                        if not marked_as_safe:
                            feedback = 0
                        elif (vis_mask[i, j] and raw_cells_seen.get(i, j) is not None) \
                             and raw_cells_seen.get(i, j).is_hazardous:
                            feedback = -1
                        else:
                            feedback = 1
                        if self.summer:
                            inp[3] = feedback
                        else:
                            inp[4] = feedback
                        self.dfnn.update_weights(inp, feedback, self.summer)
            if not self.pretrain and not self.update_all_seen:
                fwd_type = OBJECT_TO_IDX[fwd_cell.type]
                inp = self.mapping[fwd_type]
                output = np.asarray(self.dfnn.forward(inp)) > 0
                marked_as_safe = (self.summer and output[0]) \
                    or (not self.summer and output[1])
                if not marked_as_safe:
                    feedback = 0
                elif fwd_cell.is_hazardous:
                    feedback = -1
                else:
                    feedback = 1
                if self.summer:
                    inp[3] = feedback
                else:
                    inp[4] = feedback
                self.dfnn.update_weights(inp, feedback, self.summer)

        self.switch_count += 1
        if self.switch_count % self.switch_time == 0:
            self.summer = not self.summer
            self.switch_count = 0
            for i in range(1, self.width-1):
                for j in range(1, self.height-1):
                    self.grid.get(i, j).switch_season()

        new_obs_dict = {
            'image': new_obs,
            'direction': obs['direction'],
            'mission': obs['mission']
        }
        # print("--------------------------------")
        # print(obs['image'][:, :, 0])
        # print(new_obs[:, :, 0])
        return new_obs_dict, reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        # reset season
        self.summer = True
        self.switch_count = 0
        for i in range(1, self.width-1):
            for j in range(1, self.height-1):
                if self.grid.get(i, j) is not None:
                    self.grid.get(i, j).switch_season()
        # create new mapping
        tiles = ["unseen", "grass", "slope", "hill", "mud", "water", "lava", "wall"]
        self.mapping = {}
        pairs = [
            ['unseen', 'grass'],
            ['mud', 'water'],
            ['slope', 'hill'],
            ['lava', 'wall']
        ]
        duds = []
        ctx = temp_seed(seed) if seed is not None else contextlib.suppress()
        with ctx:
            summer_idx, winter_idx, dud_idx = np.random.choice(3, 3, replace=False)
            [duds.append(pair[np.random.randint(0, 2)]) for pair in pairs]
        safe_summer = ["unseen", "grass", "slope", "hill"]
        safe_winter = ["unseen", "grass", "mud", "water"]
        for tile in tiles:
            self.mapping[OBJECT_TO_IDX[tile]] = np.array([-1., -1., -1., 0., 0.])
        for tile in safe_summer:
            self.mapping[OBJECT_TO_IDX[tile]][summer_idx] = 1
        for tile in safe_winter:
            self.mapping[OBJECT_TO_IDX[tile]][winter_idx] = 1
        for tile in duds:
            self.mapping[OBJECT_TO_IDX[tile]][dud_idx] = 1
        self.mapping[OBJECT_TO_IDX["empty"]] = self.mapping[OBJECT_TO_IDX["unseen"]]
        self.mapping[OBJECT_TO_IDX["goal"]] = self.mapping[OBJECT_TO_IDX["unseen"]]
        if self.dfnn is not None:
            self.dfnn.load_original_weights()
            if self.pretrain:
                safe_summer_idx = [OBJECT_TO_IDX[tile] for tile in safe_summer]
                safe_winter_idx = [OBJECT_TO_IDX[tile] for tile in safe_winter]
                pretrain_summer = False
                iterations = 30
                season_length = 5
                presentation_order = np.array([
                    [OBJECT_TO_IDX[tile] for tile in tiles]
                ]*iterations)
                [np.random.shuffle(day_order) for day_order in presentation_order]
                for i in range(iterations):
                    if i % season_length == 0:
                        pretrain_summer = not pretrain_summer
                    for j in range(8):
                        food = presentation_order[i][j]
                        inputs = self.mapping[food].copy()
                        inputs[3] = 0
                        inputs[4] = 0
                        outputs = self.dfnn.forward(inputs)
                        ate_summer = pretrain_summer and outputs[0] > 0
                        ate_winter = not pretrain_summer and outputs[1] > 0
                        prev_summer = 1 if food in safe_summer_idx else -1
                        prev_winter = 1 if food in safe_winter_idx else -1
                        feedback_summer = prev_summer if ate_summer else 0
                        feedback_winter = prev_winter if ate_winter else 0
                        inputs[3] = feedback_summer
                        inputs[4] = feedback_winter
                        self.dfnn.update_weights(
                            inputs,
                            feedback_summer if pretrain_summer else feedback_winter,
                            pretrain_summer
                        )
        return super().reset(seed=seed, options=options)


def check_for_grass(x, y, terrain_map, humidity_map):
    return terrain_map[x-1, y-1] < t_water \
        or terrain_map[x-1, y-1] >= t_grass \
        or humidity_map[x-1, y-1] > h_mud


@contextlib.contextmanager
def temp_seed(seed):
    """Provides a temporary deterministic seed, to not interfere with global
        randomness.

    Taken from https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed

    Args:
        seed (int): The seed for the local random state.

    Returns:
        None.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def random_interpolation(w, h, seed=None):
    out = np.zeros((w, h))
    ctx = temp_seed(seed) if seed is not None else contextlib.suppress()
    with ctx:
        R = np.random.rand(3, 3)
    for i in range(w):
        for j in range(h):
            ri = int(not i < w/2)
            rj = int(not j < h/2)
            r11 = R[ri, rj]
            r21 = R[ri+1, rj]
            r12 = R[ri, rj+1]
            r22 = R[ri+1, rj+1]
            x = (i if i < w/2 else i - w/2)/w * 2
            y = (j if j < h/2 else j - h/2)/h * 2
            i1 = (r21 - r11) * (x - np.floor(x)) + r11
            i2 = (r22 - r12) * (x - np.floor(x)) + r12
            out[i, j] = (i2 - i1) * (y - np.floor(y)) + i1
    return out


if __name__ == '__main__':
    from minigrid.manual_control import ManualControl

    env = TstEnvironment(render_mode="human", size=10)
    manual_control = ManualControl(env, seed=2024)
    manual_control.start()
