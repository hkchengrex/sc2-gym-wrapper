import gym
from gym_env.move_env import SimpleMovementEnv

from pysc2.env import sc2_env

# Make pysc2 happy
import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

env = gym.make('CollectMineralShards-v0')

env.settings['visualize'] = True
env.settings['step_mul'] = 8
env.settings['agent_interface_format'] = sc2_env.parse_agent_interface_format(
          feature_screen=64,
          feature_minimap=64,
          rgb_screen=None,
          rgb_minimap=None,
          action_space='FEATURES',
          use_feature_units=True,
          use_raw_units=True)

done = False

while True:
    env.reset()
    done = False
    while not done:
        obs, reward, done, _ = env.step(0)

env.close()
