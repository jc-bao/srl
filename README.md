# Sparse-RL

## TODO

### L1

- [x] Add SAC
- [x] Add HER
  - [x] add info (env_step, traj_idx) to HER
  - [-] add trajectory extra info (traj_len, ag_pool) to HER
- [ ] Toy Env
  - [x] reach
  - [x] pnp
  - [x] handover
- [x] Eval Func (in agent)
- [ ] Normalizer
- [x] logger
- [x] check buffer function for multi done collect
- [x] env reset function 

### L2

- [ ] Add Attention
- [ ] Add Isaac Env
- [ ] Hydra
- [ ] Render function
- [ ] merge other buffer and state buffer
- [ ]  `get_env_params` function

### L3

- [ ] add MEGA
- [ ] add HGG
- [ ] add curiosity
- [ ] add optuna (wandb)
- [ ] update according to collected steps
- [ ] merge buffer into agent
- [ ] fix relabel for to generate to left index
- [ ] add vec transitions at a time


## check

- [x] use mask to change final state obs to solve boundary issue (if correct )
- [ ] use of target network