import numpy as np
from pysc2.lib import actions
import common.utils as U
from pysc2.lib import actions, features

PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale


class FeatureTransform:

    def __init__(self, ssize, msize):
        self.ssize = ssize
        self.msize = msize

        self.n_mini_map = 0
        self.n_feature_screen = 1
        self.n_info = 1

    def transform(self, obs):
        minimap = obs.observation['feature_minimap']
        feature_screen = obs.observation['feature_screen']

        screens = np.zeros([self.n_feature_screen, self.ssize, self.ssize], dtype=np.float32)
        infos = np.zeros([self.n_info], dtype=np.float32)

        ###################################################

        feature_screen[0] = minimap['player_relative']
        infos[0] = obs.observation["player"][1]

        ###################################################

        return screens, infos
