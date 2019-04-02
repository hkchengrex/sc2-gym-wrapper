import gym, ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.tune import run_experiments, grid_search

from pysc2.env import sc2_env

# Make pysc2 happy
import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

from gym_env.move_env import CollectMineralShardsEnv
from gym_env.test_env import TestBaseEnv

def env_creator(config):
    return CollectMineralShardsEnv(**config)  # return an env instance

def env_creator2(config):
    return TestBaseEnv(**config)  # return an env instance

register_env('CollectMineralShardsEnv', env_creator)
register_env('TestBaseEnv', env_creator2)

ray.init(num_gpus=0, object_store_memory=2*(10**9),redis_max_memory=4*(10**9))

settings = {}
settings['visualize'] = False
settings['step_mul'] = 8
settings['agent_interface_format'] = sc2_env.parse_agent_interface_format(
          feature_screen=84,
          feature_minimap=84,
          rgb_screen=None,
          rgb_minimap=None,
          action_space='FEATURES',
          use_feature_units=True,
          use_raw_units=True)

trainer = ppo.APPOAgent(env='CollectMineralShardsEnv', config={
    "env_config": settings,  # config to pass to env class
    "num_gpus":1, 
    "tf_session_args": {"device_count": { "GPU": 1 }},
    "num_workers":4,
    "num_envs_per_worker":2,
    "sample_batch_size": 10,
    "train_batch_size": 100,
})

# trainer = ppo.APPOAgent(env='TestBaseEnv', config={
#     "env_config": settings,  # config to pass to env class
#     "num_gpus":1, 
#     "tf_session_args": {"device_count": { "GPU": 1 }},
#     "num_workers":12,
#     "sample_batch_size": 10,
#     "train_batch_size": 100,
# })


# trainer = ppo.PPOAgent(env='Acrobot-v1', config={
#     "num_gpus":1, 
#     "tf_session_args": {"device_count": { "GPU": 1 }},
#     "num_workers":32,
#     "simple_optimizer":True,
# })

while True:
    print(trainer.train())

# run_experiments({
#     "demo": {
#         "run": "PPO",
#         "env": "CollectMineralShardsEnv",  
#         "stop": {
#             "timesteps_total": 1e4,
#         },
#         "resources_per_trial": {
#             "cpu": 1, 
#             "gpu": 1,
#         },
#         "config": {
#             "lr": grid_search([1e-4]),  # try different lrs
#             "num_workers": 1,  # parallelism
#             "env_config": settings,
#         },
#     },
# }, reuse_actors=True)

