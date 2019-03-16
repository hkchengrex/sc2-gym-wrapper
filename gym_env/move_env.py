import logging

import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features
from pysc2.env import sc2_env

import gym
from gym import spaces

from gym_env.base_env import SC2BaseEnv

FUNCTIONS = actions.FUNCTIONS
PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale

# With reference from https://github.com/islamelnabarawy/sc2gym/blob/master/sc2gym/envs/movement_minigame.py

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SimpleMovementEnv(SC2BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._action_space = None
        self._observation_space = None

    def reset(self):
        super().reset()
        obs, _, _, _ = super().step([FUNCTIONS.select_army.id, [0]])
        return self._process_obs(obs)

    def step(self, action):
        action = self._process_action(action)
        obs, reward, done, info = super().step(action)
        if obs is None:
            return None, 0, True, {}
        obs = self._process_obs(obs)
        return obs, reward, done, info

    def _process_obs(self, obs):
        obs = obs.observation["feature_screen"][PLAYER_RELATIVE]
        obs = np.array(obs.reshape(self.observation_space.shape))
        return obs

    def _process_action(self, action):
        return [FUNCTIONS.Move_screen.id, [0], action]

    @property
    def observation_space(self):
        if self._observation_space is None:
            self._observation_space = self._get_observation_space()
        return self._observation_space

    def _get_observation_space(self):
        screen_shape = self.observation_spec[0]["feature_screen"][1:] + (1,)
        space = spaces.Box(low=0, high=PLAYER_RELATIVE_SCALE, shape=screen_shape, dtype=np.int32)
        return space

    @property
    def action_space(self):
        if self._action_space is None:
            self._action_space = self._get_action_space()
        return self._action_space

    def _get_action_space(self):
        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        return spaces.MultiDiscrete([s-1 for s in screen_shape])


class CollectMineralShardsEnv(SimpleMovementEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name='CollectMineralShards', **kwargs)