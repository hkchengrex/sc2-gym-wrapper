# pytorch-sc2-ppo

## Get RL library and install it
`git clone git@github.com:hkchengrex/pytorch-a2c-ppo-acktr-gail.git && cd pytorch-a2c-ppo-acktr-gail.git`

`pip install -e .`

## Test PPO on CollectMineralShards
`python test_ppo.py --env-name "CollectMineralShards-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01`
