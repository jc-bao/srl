import os
import torch
import numpy as np
from copy import deepcopy
from pprint import pprint

def build_env(env=None, env_func=None, env_args=None):
  if env is not None:
    env = deepcopy(env)
  elif env_func.__module__ == 'gym.envs.registration':
    import gym
    gym.logger.set_level(40)  # Block warning
    env = env_func(id=env_args['env_name'])
  else:
    env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))

  for attr_str in ('state_dim', 'action_dim', 'max_step', 'if_discrete', 'target_return'):
    if (not hasattr(env, attr_str)) and (attr_str in env_args):
      setattr(env, attr_str, env_args[attr_str])
  return env


def kwargs_filter(func, kwargs: dict):
  import inspect

  sign = inspect.signature(func).parameters.values()
  sign = set([val.name for val in sign])

  common_args = sign.intersection(kwargs.keys())
  filtered_kwargs = {key: kwargs[key] for key in common_args}
  return filtered_kwargs


class Arguments:
  def __init__(self, agent, env=None, env_func=None, env_args=None):
    self.env = env  # the environment for training
    self.env_func = env_func  # env = env_func(*env_args)
    self.env_args = env_args  # env = env_func(*env_args)

    # HER
    self.her_rate = 0.8

    # env params
    # env_num = 1. In vector env, env_num > 1.
    self.env_num = self.update_attr('env_num')
    self.max_step = self.update_attr('max_step')  # the max step of an episode
    # the env name. Be used to set 'cwd'.
    self.env_name = self.update_attr('env_name')
    self.max_env_step = self.update_attr('max_step')
    # vector dimension (feature number) of state
    self.state_dim = self.update_attr('state_dim')
    self.other_dims = self.update_attr('other_dims')
    self.goal_dim = self.update_attr('goal_dim')
    self.info_dim = self.update_attr('info_dim')
    # vector dimension (feature number) of action
    self.action_dim = self.update_attr('action_dim')
    # discrete or continuous action space
    self.if_discrete = self.update_attr('if_discrete')
    self.target_return = self.update_attr(
      'target_return')  # target average episode return

    self.agent = agent  # DRL algorithm
    self.net_type = 'deepset' # 'deepset' or 'mlp'
    self.net_dim = 2 ** 7  # the network width
    # self.net_dim = 2 ** 9  # the network width
    # layer number of MLP (Multi-layer perception, `assert layer_num>=2`)
    self.layer_num = 2
    # self.layer_num = 3
    self.if_off_policy = self.get_if_off_policy()  # agent is on-policy or off-policy
    # save old data to splice and get a complete trajectory (for vector env)
    self.if_use_old_traj = False
    if self.if_off_policy:  # off-policy
      self.num_rollout_per_update = 1
      self.reuse = 1
      self.max_memo = 2 ** 20  # capacity of replay buffer
      # self.target_steps_per_env = 2 ** 10  # repeatedly update network to keep critic's loss small
      # self.target_steps = self.env_num * self.max_step * self.num_rollout_per_update
      # self.target_steps = 1000
      self.target_steps = 102400 
      # num of transitions sampled from replay buffer.
      # self.batch_size = self.net_dim
      self.batch_size = 2**13 # 8k 
      # self.batch_size = 2**10 # 1k 
      # self.repeat_times = 2 ** 0  # collect target_steps_per_env, then update network
      self.repeat_times = int(self.target_steps / self.batch_size * self.reuse)
      # use PER (Prioritized Experience Replay) for sparse reward
      self.if_use_per = False
    else:  # on-policy
      self.max_memo = 2 ** 12  # capacity of replay buffer
      # repeatedly update network to keep critic's loss small
      self.target_steps = 100*self.env_num 
      # num of transitions sampled from replay buffer.
      self.repeat_times = 2 ** 5  # collect target_steps_per_env, then update network
      self.batch_size = self.target_steps // self.repeat_times
      # use PER: GAE (Generalized Advantage Estimation) for sparse reward
      self.if_use_gae = False

    '''Arguments for training'''
    self.gamma = 0.99  # discount factor of future rewards
    self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
    self.learning_rate = 2 ** -12  # 2 ** -15 ~= 3e-5
    self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

    '''Arguments for device'''
    self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
    # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
    self.thread_num = 8
    self.random_seed = 0  # initialize random seed in self.init_before_training()
    self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

    '''Arguments for evaluate'''
    self.cwd = 'results'  # current working directory to save model. None means set automatically
    self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
    self.break_step = +np.inf  # break training if 'total_step > break_step'
    # over write the best policy network (actor.pth)
    self.if_over_write = False
    # allow break training when reach goal (early termination)
    self.if_allow_break = True

    '''Arguments for evaluate'''
    eval_cycle_gap = 1
    eval_ep_per_env = 10
    self.eval_gap = self.target_steps * eval_cycle_gap  # evaluate the agent per eval_gap steps
    self.eval_times = 2 ** 4  # number of times that get episode return
    self.eval_steps = eval_ep_per_env * self.env_num * self.max_step

  def init_before_training(self):
    np.random.seed(self.random_seed)
    torch.manual_seed(self.random_seed)
    torch.set_num_threads(self.thread_num)
    torch.set_default_dtype(torch.float32)

    '''auto set'''
    if self.cwd is None:
      self.cwd = f'./{self.env_name}_{self.agent.__name__[5:]}_{self.learner_gpus}'

    '''remove history'''
    if self.if_remove is None:
      self.if_remove = bool(
        input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
    elif self.if_remove:
      import shutil
      shutil.rmtree(self.cwd, ignore_errors=True)
      print(f"| Arguments Remove cwd: {self.cwd}")
    else:
      print(f"| Arguments Keep cwd: {self.cwd}")
    os.makedirs(self.cwd, exist_ok=True)

  def update_attr(self, attr: str):
    return getattr(self.env, attr) if self.env_args is None else self.env_args[attr]

  def get_if_off_policy(self):
    name = self.agent.__name__
    # if_off_policy
    return all((name.find('PPO') == -1, name.find('A2C') == -1))

  def print(self):
    # prints out args in a neat, readable format
    pprint(vars(self))
