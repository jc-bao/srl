import gym
import itertools
from gym import spaces
import numpy as np
import torch
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt
from attrdict import AttrDict


class ReachToyEnv(gym.Env):
  def __init__(self, dim: int = 2, env_num: int = 2, device='cuda:0'):
    self.dim = dim
    self.env_num = env_num
    self.device = device
    self._max_episode_steps = 20
    self.err = 0.05
    self.vel = 0.2

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
    reward = -(d > self.err).type(torch.float32)
    info = torch.cat(
      ((d < self.err).type(torch.float32).unsqueeze(-1),  # success
       self.pos),  # achieved goal
      dim=-1)
    done = torch.logical_or(
      (self.num_step >= self._max_episode_steps), (d < self.err)).type(torch.float32)
    return self.get_obs(), reward, done, info

  def reset(self):
    self.num_step = torch.zeros(self.env_num, device=self.device)
    self.goal = self.torch_goal_space.sample((self.env_num,))
    self.pos = self.torch_goal_space.sample((self.env_num,))
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

  def compute_reward(self, achieved_goal, desired_goal, info):
    d = np.linalg.norm(achieved_goal-desired_goal)
    return (d < self.err).astype(np.float32)

  def ezpolicy(self, obs):
    delta = - obs.observation + obs.desired_goal
    return delta / torch.norm(delta, dim=-1, keepdim=True)

  def parse_info(self, info):
    is_success = info[..., 0:1]
    achieved_goal = info[..., 1:self.dim+1]
    return is_success, achieved_goal


if __name__ == '__main__':
  env = ReachToyEnv()
  obs = env.reset()
  for _ in range(env._max_episode_steps):
    act = env.ezpolicy(obs)
    obs, reward, done, info = env.step(act)
    env.render()
    # print('[obs, reward, done]', obs, reward, done)
