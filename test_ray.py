import gym, ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env

from pysc2.env import sc2_env

# Make pysc2 happy
import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

from gym_env.move_env import CollectMineralShardsEnv

def env_creator(config):
    return CollectMineralShardsEnv(**config)  # return an env instance

register_env('CollectMineralShardsEnv', env_creator)

ray.init()

settings = {}
settings['visualize'] = True
settings['step_mul'] = 8
settings['agent_interface_format'] = sc2_env.parse_agent_interface_format(
          feature_screen=84,
          feature_minimap=84,
          rgb_screen=None,
          rgb_minimap=None,
          action_space='FEATURES',
          use_feature_units=True,
          use_raw_units=True)

trainer = ppo.PPOAgent(env='CollectMineralShardsEnv', config={
    "env_config": settings,  # config to pass to env class
})

# trainer = ppo.PPOAgent(env='Acrobot-v1', config={
#     "env_config": settings,  # config to pass to env class
# })

while True:
    print(trainer.train())

