# Sparse-RL

## Use Cases

#### Basics

``` bash
python run.py
```

`-f`, `--file`: config file to load in ./config

`-k`, `--kwargs`: key-value pairs to override config, e.g. `python run.py -k '{"lr":1e-3}'`

`-e`, `--envargs` : key-value pairs to override env config, e.g. `python run.py -e '{"num_goals":2}'`

Example: train pick and place:

``` bash
python run.py -k td3
```

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

#### Curriculum Learning

add these line to config file to run automatically curriculum:

```yaml
curri:
  table_gap: # env params to change
    now: 0
    end: 0.2
    step: 0.05
    bar: -0.2 # reward bar to 
```

## Issues

- [ ] HER relabel lead to fluctuations in success rate
  - [x] check relabel boundary case
  - [x] check relabel reward, mask
  - [ ] check trajectory buffer
  - [ ] check 


## Features
  
* Fast: Fully GPU-Based Pipeline  (isaac support, fast buffer indexing, fast batch relabeling)
* Designed for Sparse Reward (HER, Red-Q)

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
- [ ] Add Hydra

### L2

- [x] Add Attention Dense Net
- [x] Resume run from cloud
- [x] Add Isaac Env (fast auto reset)
  - [x] PNP
  - [x] Handover
  - [x] Multi Robot Environment
- [x] Isaac render + viewer
- [x] Render function
- [x] merge all buffer together (fast indexing)
- [x]  `get_env_params`, `obs_parser`, `info_parser` function
- [x]  curriculum learning

### L3

- [x] add ray tune (wandb)
- [x] update according to collected steps
- [x] merge buffer into agent
- [x] fix relabel for to generate to left index
- [x] add vec transitions at a time

## Misc

- [x] resume from remote
- [x] update env info dim automatically
- [x] classified log info


## Requirements

'''
isaacgym
torch
attrdict
gym
pytorch_kinematics
'''