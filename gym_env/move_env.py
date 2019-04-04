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

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
#f
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
        # obs = self._process_obs(None)
        # reward = 0
        # done = False
        # info = {}

        return obs, reward, done, info


    def _process_obs(self, obs):
        # obs = np.zeros(self.observation_space.shape)

        self.n_feature_screen = 1
        self.n_info = 8

        feature_screen = obs.observation['feature_screen']
        screen_shape = self.observation_spec[0]["feature_screen"][1:]

        screens = np.zeros((self.n_feature_screen,) + screen_shape, dtype=np.int32)
        feature_screen[0] = feature_screen['player_relative']

        return {"feature_screen": screens,
                "info_discrete": obs.observation["player"][1:6],
                }

    # def _process_action(self, action):
    #     return [FUNCTIONS.Move_screen.id, [0], action]

    def _process_action(self, action):
        if action < 0 or action > self.action_space.n:
            return [FUNCTIONS.no_op.id]
        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        target = list(np.unravel_index(action, screen_shape))
        return [FUNCTIONS.Move_screen.id, [0], target]

    @property
    def observation_space(self):
        if self._observation_space is None:
            self._observation_space = self._get_observation_space()
        return self._observation_space

    def _get_observation_space(self):
        screen_shape = (1,) + self.observation_spec[0]["feature_screen"][1:]

        high = np.array([200, 200, 200, 200, 200])
        low = np.array([0, 0, 0, 0, 0])

        space = spaces.Dict({
            "feature_screen": spaces.Box(low=0, high=PLAYER_RELATIVE_SCALE, shape=screen_shape, dtype=np.int32),
            "info_discrete": spaces.Box(low=low, high=high, dtype=np.int32),
        })
        return space

    '''
    def _define_observation_space_dict(self):
        screen_shape = (1,) + self.observation_spec[0]["feature_screen"][1:]
 
        obv_space_dict = {
            "image": spaces.Dict({
                "   feature_screen": spaces.Box(low=0, high=PLAYER_RELATIVE_SCALE, shape=screen_shape, dtype=np.int32),
            }),
            "non-image": spaces.Dict({
                "minerals": spaces.Discrete(50000),
                'vespene': spaces.Discrete(50000),
                "food_used": spaces.Discrete(200),
                "food_cap": spaces.Discrete(200),
                "food used by army": spaces.Discrete(200),
                "food used by workers": spaces.Discrete(200),
                "idle_worker_count": spaces.Discrete(200),
                'army count': spaces.Discrete(200),
            })
        }


        return obv_space_dict
        '''

    @property
    def action_space(self):
        if self._action_space is None:
            self._action_space = self._get_action_space()
        return self._action_space

    # def _get_action_space(self):
    #     screen_shape = self.observation_spec[0]["feature_screen"][1:]
    #     return spaces.MultiDiscrete([s-1 for s in screen_shape])

    def _get_action_space(self):
        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        return spaces.Discrete(screen_shape[0] * screen_shape[1] - 1)


class CollectMineralShardsEnv(SimpleMovementEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name='CollectMineralShards', **kwargs)
