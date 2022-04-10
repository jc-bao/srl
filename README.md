# Sparse-RL

## TODO

### L1

- [x] Add SAC
- [x] Add PPO
- [x] Add RedQ
- [x] Add HER
  - [x] add info (env_step, traj_idx) to HER
  - [x] add trajectory extra info (traj_len, ag_pool) to HER
- [x] Toy Env
  - [x] reach
  - [x] pnp
  - [x] handover
- [x] Eval Func (in agent)
- [x] Normalizer
- [x] logger
- [x] check buffer function for multi done collect
- [x] env reset function 

### L2

- [ ] Add Attention
- [x] Add Isaac Env (fast auto reset)
  - [x] PNP
  - [ ] Handover
- [x] Isaac render + viewer
- [x] Hydra
- [x] Render function
- [ ] merge other buffer and state buffer
- [x]  `get_env_params` function

### L3

- [ ] add MEGA
- [ ] add HGG
- [ ] add curiosity
- [ ] add optuna (wandb)
- [ ] update according to collected steps
- [x] merge buffer into agent
- [ ] fix relabel for to generate to left index
- [x] add vec transitions at a time


## check

- [x] use mask to change final state obs to solve boundary issue (if correct )
- [ ] use of target network

## Use Cases

train

```
python run.py -k '{"env_name":"PandaPNP-v0"}'
``` 

show viewer when train:

```
python run.py -k '{"env_name":"PandaPNP-v0","wandb":false,"render":false,"steps_per_rollout":400}' -e '{"num_envs":4,"num_cameras":0,"headless":false}'
```

change episode length to debug:

```
python run.py -k '{"env_name":"PandaPNP-v0","wandb":false,"render":false,"steps_per_rollout":20,"batch_size":8}' -e '{"num_envs":1,"num_cameras":0,"headless":true,"base_steps":20}'
```