import gym
import itertools
from gym import spaces
import numpy as np
import torch
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt
from attrdict import AttrDict


class ReachToyEnv(gym.Env):
  def __init__(self, dim: int = 2, env_num: int = 2, gpu_id=0, max_step=20, auto_reset=True, err=0.05, vel=0.2):
    self.device = torch.device(f"cuda:{gpu_id}" if (
      torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    self.dim = dim
    self.env_num = env_num
    self._max_episode_steps = max_step
    self.err = err
    self.vel = vel
    self.auto_reset = auto_reset
    self.num_step = torch.empty(self.env_num, dtype=torch.int, device=self.device)
    self.reach_step = torch.empty(self.env_num, dtype=torch.int, device=self.device)
    self.goal = torch.empty((self.env_num, self.dim), dtype=torch.float32, device=self.device)
    self.pos = torch.empty((self.env_num, self.dim), dtype=torch.float32, device=self.device)

    # gym space
    self.space = spaces.Box(low=-np.ones(self.dim), high=np.ones(self.dim))
    self.action_space = spaces.Box(
      low=-np.ones(self.dim), high=np.ones(self.dim))
    self.obs_space = spaces.Box(
      low=-np.ones(self.dim*2), high=np.ones(self.dim*2)),
    self.goal_space = spaces.Box(
      low=-np.ones(self.dim), high=np.ones(self.dim))

    # torch space for vec env
    self.torch_space = Uniform(
      low=-torch.ones(self.dim, device=self.device), high=torch.ones(self.dim, device=self.device))
    self.torch_action_space = self.torch_space
    self.torch_obs_space = Uniform(
      low=-torch.ones(self.dim*2, device=self.device), high=torch.ones(self.dim*2, device=self.device)
    )
    self.torch_goal_space = self.torch_space
    
    self.reset()

  def step(self, action):
    self.num_step += 1
    action = torch.clamp(action, self.torch_action_space.low,
                        self.torch_action_space.high)
    self.pos = torch.clamp(self.pos + action*self.vel,
                          self.torch_space.low, self.torch_space.high)
    d = torch.norm(self.pos - self.goal, dim=-1)
    if_reach = (d < self.err)
    # if reached, not update reach step
    self.reach_step[~if_reach] = self.num_step[~if_reach]
    reward = self.compute_reward(self.pos, self.goal, None)
    info = torch.cat((
      if_reach.type(torch.float32).unsqueeze(-1),  # success
      self.num_step.type(torch.float32).unsqueeze(-1),  # step
      self.pos),  # achieved goal
      dim=-1)
    done = torch.logical_or(
      (self.num_step >= self._max_episode_steps), (self.num_step - self.reach_step) > 4).type(torch.float32)
    if self.auto_reset:
      env_idx = torch.where(done > 1-1e-3)[0] # to avoid nan
      self.reset(env_idx)
    return self.get_obs(), reward, done, info

  def reset(self, env_idx = None):
    if env_idx is None:
      env_idx = torch.arange(self.env_num)
    num_reset_env = env_idx.shape[0]
    self.num_step[env_idx] = 0
    self.reach_step[env_idx] = 0
    self.goal[env_idx] = self.torch_goal_space.sample((num_reset_env,))
    self.pos[env_idx] = self.torch_goal_space.sample((num_reset_env,))
    return self.get_obs()

  def render(self):
    if self.num_step[0] == 1:
      self.data = [self.pos]
    else:
      self.data.append(self.pos)
    if self.num_step[0] == self._max_episode_steps:
      fig, ax = plt.subplots(self.env_num)
      if self.dim == 2:
        for t, env_id in itertools.product(range(self._max_episode_steps), range(self.env_num)):
          x = self.data[t][env_id][0].detach().cpu()
          y = self.data[t][env_id][1].detach().cpu()
          ax[env_id].plot(x, y, 'o', color=[
                          0, 0, 1, t/self._max_episode_steps])
        for env_id in range(self.env_num):
          x = self.goal[env_id][0].detach().cpu()
          y = self.goal[env_id][1].detach().cpu()
          ax[env_id].plot(x, y, 'rx')
        plt.show()
      elif self.dim == 3:
        fig, ax = plt.subplots(1, self.env_num)
        for i in range(self.env_num):
          for i, d in enumerate(self.data[i]):
            ax[i].scatter(d[0], d[1], d[2], 'o', color=[0, 0, 1, i/50])
        plt.show()

  def get_obs(self):
    return torch.cat((self.pos, self.goal), dim=-1)

  def compute_reward(self, ag, dg, info):
    return -(torch.norm(ag-dg,dim=-1) > self.err).type(torch.float32)

  def ezpolicy(self, obs):
    delta = - obs[..., :self.dim] + obs[..., self.dim:]
    return (delta / torch.norm(delta, dim=-1, keepdim=True))

  def parse_info(self, info):
    is_success = info[..., 0:1]
    step = info[..., 1:2]
    achieved_goal = info[..., 2:self.dim+2]
    return is_success, step, achieved_goal


if __name__ == '__main__':
  env = ReachToyEnv(gpu_id=-1, err=0.2)
  obs = env.reset()
  for _ in range(env._max_episode_steps):
    act = env.ezpolicy(obs)
    obs, reward, done, info = env.step(act)
    # env.render()
    # print('[obs, reward, done]', obs, reward, done)
