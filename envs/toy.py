import gym
import itertools
from gym import spaces
import numpy as np
import torch
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt
from attrdict import AttrDict
import pathlib
import yaml

class ReachToyEnv(gym.Env):
	def __init__(self, cfg_file='configs/ReachToy.yaml', **kwargs):
		# get config and setup base class
		cfg_path = pathlib.Path(__file__).parent.resolve()/cfg_file
		with open(cfg_path) as config_file:
			try:
				cfg = AttrDict(yaml.load(config_file, Loader=yaml.SafeLoader))
			except yaml.YAMLError as exc:
				print(exc)
		cfg.update(**kwargs)  # overwrite params from args
		self.cfg = self.update_config(cfg)
		self.device = torch.device(f"cuda:{self.cfg.sim_device_id}" if (
			torch.cuda.is_available() and (self.cfg.sim_device_id>= 0)) else "cpu")
			
		self.num_step = torch.empty(self.cfg.num_envs, dtype=torch.int, device=self.device)
		self.reach_step = torch.empty(self.cfg.num_envs, dtype=torch.int, device=self.device)
		self.goal = torch.empty((self.cfg.num_envs, self.cfg.dim), dtype=torch.float32, device=self.device)
		self.pos = torch.empty((self.cfg.num_envs, self.cfg.dim), dtype=torch.float32, device=self.device)

		# gym space
		self.space = spaces.Box(low=-np.ones(self.cfg.dim), high=np.ones(self.cfg.dim))
		self.action_space = spaces.Box(
			low=-np.ones(self.cfg.dim), high=np.ones(self.cfg.dim))
		self.obs_space = spaces.Box(
			low=-np.ones(self.cfg.dim*2), high=np.ones(self.cfg.dim*2)),
		self.goal_space = spaces.Box(
			low=-np.ones(self.cfg.dim), high=np.ones(self.cfg.dim))

		# torch space for vec env
		self.torch_space = Uniform(
			low=-torch.ones(self.cfg.dim, device=self.device), high=torch.ones(self.cfg.dim, device=self.device))
		self.torch_action_space = self.torch_space
		self.torch_obs_space = Uniform(
			low=-torch.ones(self.cfg.dim*2, device=self.device), high=torch.ones(self.cfg.dim*2, device=self.device)
		)
		self.torch_goal_space = self.torch_space
		
		self.reset()

	def step(self, action):
		self.num_step += 1
		action = torch.clamp(action, self.torch_action_space.low,
												self.torch_action_space.high)
		self.pos = torch.clamp(self.pos + action*self.cfg.vel,
													self.torch_space.low, self.torch_space.high)
		d = torch.norm(self.pos - self.goal, dim=-1)
		if_reach = (d < self.cfg.err)
		# if reached, not update reach step
		self.reach_step[~if_reach] = self.num_step[~if_reach]
		reward = self.compute_reward(self.pos, self.goal, None)
		info = torch.cat((
			if_reach.type(torch.float32).unsqueeze(-1),  # success
			self.num_step.type(torch.float32).unsqueeze(-1),  # step
			torch.empty((self.cfg.num_envs, 3), device=self.device, dtype=torch.float),# traj_idx, traj_len, tleft 
			self.pos),  # achieved goal
			dim=-1)
		done = torch.logical_or(
			(self.num_step >= self.cfg.max_steps), (self.num_step - self.reach_step) > 4).type(torch.float32)
		if self.cfg.auto_reset:
			env_idx = torch.where(done > 1-1e-3)[0] # to avoid nan
			self.reset(env_idx)
		return self.get_obs(), reward, done, info

	def reset(self, env_idx = None):
		if env_idx is None:
			env_idx = torch.arange(self.cfg.num_envs)
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
		if self.num_step[0] == self.cfg.max_steps:
			fig, ax = plt.subplots(self.cfg.num_envs)
			if self.cfg.dim == 2:
				for t, env_id in itertools.product(range(self.cfg.max_steps), range(self.cfg.num_envs)):
					x = self.data[t][env_id][0].detach().cpu()
					y = self.data[t][env_id][1].detach().cpu()
					ax[env_id].plot(x, y, 'o', color=[
													0, 0, 1, t/self.cfg.max_steps])
				for env_id in range(self.cfg.num_envs):
					x = self.goal[env_id][0].detach().cpu()
					y = self.goal[env_id][1].detach().cpu()
					ax[env_id].plot(x, y, 'rx')
				plt.show()
			elif self.cfg.dim == 3:
				fig, ax = plt.subplots(1, self.cfg.num_envs)
				for i in range(self.cfg.num_envs):
					for i, d in enumerate(self.data[i]):
						ax[i].scatter(d[0], d[1], d[2], 'o', color=[0, 0, 1, i/50])
				plt.show()

	def get_obs(self):
		return torch.cat((self.pos, self.goal), dim=-1)

	def compute_reward(self, ag, dg, info):
		return -(torch.norm(ag-dg,dim=-1) > self.cfg.err).type(torch.float32)
	
	def update_config(self, cfg):
		cfg.update(
			action_dim = cfg.dim, 
			state_dim = cfg.dim * 2,
			shared_dim = 0, 
			seperate_dim = cfg.dim, 
			goal_dim = cfg.dim, 
			info_dim = 5 + cfg.dim, 
		)
		return cfg

	def ezpolicy(self, obs):
		delta = - obs[..., :self.cfg.dim] + obs[..., self.cfg.dim:]
		return (delta / torch.norm(delta, dim=-1, keepdim=True))

	def obs_parser(self, obs):
		assert obs.shape[-1] == self.cfg.state_dim
		return AttrDict(
			shared = obs[..., :self.cfg.shared_dim],
			seperate = obs[..., self.cfg.shared_dim:self.cfg.shared_dim+self.cfg.seperate_dim],
			ag = obs[..., self.cfg.shared_dim+self.cfg.seperate_dim-self.cfg.goal_dim:self.cfg.shared_dim+self.cfg.seperate_dim], 
			g = obs[..., self.cfg.shared_dim+self.cfg.seperate_dim:]
		)
	
	def obs_updater(self, old_obs, new_obs:AttrDict):
		if 'shared' in new_obs:
			old_obs[..., :self.cfg.shared_dim] = new_obs.shared	
		if 'seperate' in new_obs:
			old_obs[..., self.cfg.shared_dim:self.cfg.shared_dim+self.cfg.seperate_dim] = new_obs.seperate
		if 'ag' in new_obs:
			old_obs[..., self.cfg.shared_dim+self.cfg.seperate_dim-self.cfg.goal_dim:self.cfg.shared_dim+self.cfg.seperate_dim] = new_obs.ag
		if 'g' in new_obs:
			old_obs[..., self.cfg.shared_dim+self.cfg.seperate_dim:] = new_obs.g
		return old_obs

	def info_parser(self, info):
		assert info.shape[-1] == self.cfg.info_dim, f'info {self.cfg.info_dim} shape error: {info.shape}' 
		return AttrDict(
			success = info[..., 0], 
			step= info[...,1],
			traj_idx = info[..., 2],
			traj_len = info[..., 3],
			tleft = info[..., 4],
			ag = info[..., 5:5+self.cfg.goal_dim]
		)

	def info_updater(self, old_info, new_info:AttrDict):
		if 'success' in new_info:
			old_info[..., 0] = new_info.success
		if 'step' in new_info:
			old_info[..., 1] = new_info.step
		if 'traj_idx' in new_info:
			old_info[..., 2] = new_info.traj_idx
		if 'traj_len' in new_info:
			old_info[..., 3] = new_info.traj_len
		if 'tleft' in new_info:
			old_info[..., 4] = new_info.tleft
		if 'ag' in new_info:
			old_info[..., 5:5+self.cfg.goal_dim] = new_info.ag
		return old_info

	def close(self):
		self.gym.destroy_viewer(self.viewer)
		self.gym.destroy_sim(self.sim)

	def env_params(self):
		return AttrDict(
			# dims
			action_dim=self.cfg.action_dim,
			state_dim=self.cfg.state_dim,
			shared_dim=self.cfg.shared_dim,
			seperate_dim=self.cfg.seperate_dim,
			goal_dim=self.cfg.goal_dim,
			info_dim=self.cfg.info_dim,  # is_success, step, achieved_goal
			# numbers
			num_goals = 1, 
			num_envs=self.cfg.num_envs,
			max_env_step=self.cfg.max_steps,
			# functions
			compute_reward=self.compute_reward,
			info_parser=self.info_parser,
			info_updater=self.info_updater,
			obs_parser=self.obs_parser, 
			obs_updater=self.obs_updater,
		)


class PNPToyEnv(gym.Env):
	def __init__(self, dim: int = 2, num_envs: int = 2, gpu_id=0, max_steps=40, auto_reset=True, err=0.1, vel=0.2, reward_type='sparse', num_goals = 1):
		self.device = torch.device(f"cuda:{gpu_id}" if (
			torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
		self.cfg.dim = dim
		self.num_goals = num_goals
		self.reward_type = reward_type
		self.cfg.num_envs = num_envs
		self.cfg.max_steps = max_steps
		self.cfg.err = torch.tensor(err, device=self.device)
		self.cfg.vel = vel
		self.cfg.auto_reset = auto_reset
		self.num_step = torch.empty(self.cfg.num_envs, dtype=torch.int, device=self.device)
		self.reach_step = torch.empty(self.cfg.num_envs, dtype=torch.int, device=self.device)
		self.goal = torch.empty((self.cfg.num_envs, self.num_goals, self.cfg.dim), dtype=torch.float32, device=self.device)
		self.pos = torch.empty((self.cfg.num_envs, 1, self.cfg.dim), dtype=torch.float32, device=self.device)
		self.obj = torch.empty((self.cfg.num_envs, self.num_goals, self.cfg.dim), dtype=torch.float32, device=self.device)
		self.attached = torch.empty((self.cfg.num_envs, self.num_goals), dtype=torch.bool, device=self.device)
		self.reached = torch.empty((self.cfg.num_envs, self.num_goals), dtype=torch.bool, device=self.device)

		# gym space
		self.space = spaces.Box(low=-np.ones(self.cfg.dim), high=np.ones(self.cfg.dim))
		self.action_space = spaces.Box(
			low=-np.ones(self.cfg.dim), high=np.ones(self.cfg.dim))
		self.obs_space = spaces.Box(
			low=-np.ones(self.cfg.dim*2), high=np.ones(self.cfg.dim*2)),
		self.goal_space = spaces.Box(
			low=-np.ones(self.cfg.dim), high=np.ones(self.cfg.dim))

		# torch space for vec env
		self.torch_space = Uniform(
			low=-torch.ones(self.cfg.dim, device=self.device), high=torch.ones(self.cfg.dim, device=self.device))
		self.torch_action_space = self.torch_space
		self.torch_pos_space = Uniform(
			low=-torch.ones((1, self.cfg.dim), device=self.device), high=torch.ones((1, self.cfg.dim), device=self.device))
		self.torch_goal_space = Uniform(
			low=-torch.ones((self.num_goals, self.cfg.dim), device=self.device), high=torch.ones((self.num_goals, self.cfg.dim), device=self.device))
		
		self.reset()

	def step(self, action):
		# move agent
		self.num_step += 1
		action = torch.clamp(action, self.torch_action_space.low,
												self.torch_action_space.high)
		self.pos = torch.clamp(self.pos + action.unsqueeze
		(1)*self.cfg.vel,
													self.torch_pos_space.low, self.torch_pos_space.high)
		# move obj with agent
		self.obj[self.attached] = self.pos.repeat(1, self.num_goals, 1)[self.attached]
		# fix obj if reached
		self.obj[self.reached] = self.goal[self.reached]
		# if reached, not update reach step
		if_reach = self.reached.all(dim=-1)
		self.reach_step[~if_reach] = self.num_step[~if_reach]
		# check if obj is attached (make it when every thing is over)
		distance = torch.norm(self.obj - self.pos, dim=-1)
		min_dis = distance.min(dim=-1)[0]
		# only consider the most close point
		self.attached = (distance <= torch.min(self.cfg.err, min_dis).unsqueeze(-1))
		self.reached = torch.norm(self.goal - self.obj, dim=-1) < self.cfg.err
		reward = self.compute_reward(self.obj, self.goal, None)
		info = torch.cat((
			if_reach.type(torch.float32).unsqueeze(-1),  # success
			self.num_step.type(torch.float32).unsqueeze(-1),  # step
			self.obj.reshape(self.cfg.num_envs, -1)),  # achieved goal
			dim=-1)
		done = torch.logical_or(
			(self.num_step >= self.cfg.max_steps), (self.num_step - self.reach_step) > 4).type(torch.float32)
		if self.cfg.auto_reset:
			env_idx = torch.where(done > 1-1e-3)[0] # to avoid nan
			self.reset(env_idx)
		return self.get_obs(), reward, done, info

	def reset(self, env_idx = None):
		if env_idx is None:
			env_idx = torch.arange(self.cfg.num_envs)
		num_reset_env = env_idx.shape[0]
		self.num_step[env_idx] = 0
		self.reach_step[env_idx] = 0
		self.goal[env_idx] = self.torch_goal_space.sample((num_reset_env,))
		self.pos[env_idx] = self.torch_pos_space.sample((num_reset_env,))
		self.obj[env_idx] = self.torch_goal_space.sample((num_reset_env,))
		self.attached[env_idx] = False
		self.reached[env_idx] = False
		return self.get_obs()

	def render(self):
		if self.num_step[0] == 1:
			self.data = [[self.pos.clone(), self.obj.clone()]]
		else:
			self.data.append([self.pos.clone(), self.obj.clone()])
		if self.num_step[0] == self.cfg.max_steps:
			fig, ax = plt.subplots(1, self.cfg.num_envs)
			if self.cfg.dim == 2:
				for t, env_id in itertools.product(range(self.cfg.max_steps), range(self.cfg.num_envs)):
					for goal_id in range(self.num_goals):
						# object
						o_x = self.data[t][1][env_id,goal_id,0].detach().cpu()
						o_y = self.data[t][1][env_id,goal_id,1].detach().cpu()
						ax[env_id].plot(o_x, o_y, 'o', color=[
														0, 1, goal_id/self.num_goals, t/self.cfg.max_steps], markersize=10)
					# agent
					a_x = self.data[t][0][env_id,0,0].detach().cpu()
					a_y = self.data[t][0][env_id,0,1].detach().cpu()
					ax[env_id].plot(a_x, a_y, 'o', color=[
													0, 0, 1, t/self.cfg.max_steps])
				for env_id in range(self.cfg.num_envs):
					for goal_id in range(self.num_goals):
						x = self.goal[env_id,goal_id,0].detach().cpu()
						y = self.goal[env_id,goal_id,1].detach().cpu()
						ax[env_id].plot(x, y, 'x', color=[0.2,1,goal_id/self.num_goals,1], markersize=20)
				plt.show()
			elif self.cfg.dim == 3:
				fig, ax = plt.subplots(1, self.cfg.num_envs)
				for i in range(self.cfg.num_envs):
					for i, d in enumerate(self.data[i]):
						ax[i].scatter(d[0], d[1], d[2], 'o', color=[0, 0, 1, i/50])
				plt.show()

	def get_obs(self):
		return torch.cat((self.pos.reshape(self.cfg.num_envs, -1), self.obj.reshape(self.cfg.num_envs, -1), self.goal.reshape(self.cfg.num_envs, -1)), dim=-1)

	def compute_reward(self, ag, dg, info):
		if self.reward_type == 'sparse':
			return -torch.mean((torch.norm(ag.reshape(-1, self.num_goals, self.cfg.dim)-dg.reshape(-1, self.num_goals, self.cfg.dim),dim=-1) > self.cfg.err).type(torch.float32), dim=-1)
		elif self.reward_type == 'dense':
			return 1-torch.mean(torch.norm(ag.reshape(-1, self.num_goals, self.cfg.dim)-dg.reshape(-1, self.num_goals, self.cfg.dim),dim=-1), dim=-1)/self.num_goals

	def ezpolicy(self, obs):
		pos = obs[..., :self.cfg.dim]
		obj = obs[..., self.cfg.dim:(self.num_goals+1)*self.cfg.dim].reshape(self.cfg.num_envs, self.num_goals, self.cfg.dim)
		goal = obs[..., (self.num_goals+1)*self.cfg.dim:].reshape(self.cfg.num_envs, self.num_goals, self.cfg.dim)
		action = torch.zeros(self.cfg.num_envs, self.cfg.dim)
		for env_id in range(self.cfg.num_envs):
			pos_now = pos[env_id]
			for goal_id in range(self.num_goals):
				obj_now = obj[env_id, goal_id] 
				goal_now = goal[env_id, goal_id]
				reached = torch.norm(obj_now-goal_now) < self.cfg.err
				attached = torch.norm(pos_now - obj_now) < self.cfg.err
				if reached:
					pass
				elif attached:
					action[env_id] = goal_now - pos_now
				else:
					action[env_id] = obj_now - pos_now
		return action

	def parse_info(self, info):
		is_success = info[..., 0:1]
		step = info[..., 1:2]
		achieved_goal = info[..., 2:self.cfg.dim+2]
		return is_success, step, achieved_goal
	
	def env_params(self):
		return AttrDict(
			# dims
			action_dim = self.cfg.dim, 
			state_dim = self.cfg.dim*(2*self.num_goals+1), 
			shared_dim = self.cfg.dim, 
			seperate_dim = self.cfg.dim*self.num_goals,
			goal_dim = self.cfg.dim*self.num_goals, 
			info_dim = 2 + self.num_goals*self.cfg.dim, # is_success, step, achieved_goal
			# numbers
			num_goals = self.num_goals, 
			num_envs = self.cfg.num_envs,
			max_env_step = self.cfg.max_steps,
			# functions
			reward_fn = self.compute_reward,
		)

class HandoverToyEnv(gym.Env):
	"""handover topy env

	Args:
			gym (_type_): _description_

	NOTE:
		1. the obs is normalized, please parse it before use
	"""
	def __init__(self, dim: int = 2, num_envs: int = 2, gpu_id=0, max_steps=40, auto_reset=True, err=0.1, vel=0.2, use_gripper = False):
		self.device = torch.device(f"cuda:{gpu_id}" if (
			torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
		self.use_gripper = use_gripper
		self.cfg.dim = dim
		self.cfg.num_envs = num_envs
		self.cfg.max_steps = max_steps
		self.cfg.err = err
		self.cfg.vel = vel
		self.cfg.auto_reset = auto_reset
		self.num_step = torch.empty(self.cfg.num_envs, dtype=torch.int, device=self.device)
		self.reach_step = torch.empty(self.cfg.num_envs, dtype=torch.int, device=self.device)
		self.goal = torch.empty((self.cfg.num_envs, self.cfg.dim), dtype=torch.float32, device=self.device)
		self.pos0 = torch.empty((self.cfg.num_envs, self.cfg.dim), dtype=torch.float32, device=self.device)
		self.grip0 = torch.empty((self.cfg.num_envs,), dtype=torch.float32, device=self.device)
		self.pos1 = torch.empty((self.cfg.num_envs, self.cfg.dim), dtype=torch.float32, device=self.device)
		self.grip1 = torch.empty((self.cfg.num_envs,), dtype=torch.float32, device=self.device)
		self.obj = torch.empty((self.cfg.num_envs, self.cfg.dim), dtype=torch.float32, device=self.device)
		self.attached0 = torch.empty((self.cfg.num_envs, ), dtype=torch.bool, device=self.device)
		self.attached1 = torch.empty((self.cfg.num_envs, ), dtype=torch.bool, device=self.device)

		# torch space for vec env
		self.torch_space0 = Uniform(
			low=torch.tensor([-2.0,-1.0], device=self.device), high=torch.tensor([0.,1.], device=self.device))
		self.torch_space0_mean = (self.torch_space0.low + self.torch_space0.high)/2
		self.torch_space1 = Uniform(
			low=torch.tensor([0.,-1.], device=self.device), high=torch.tensor([2.,1.], device=self.device))
		self.torch_space1_mean = (self.torch_space1.low + self.torch_space1.high)/2
		self.torch_action_space = Uniform(
			low=-torch.ones(self.cfg.dim*2+2, device = self.device), high=torch.ones(self.cfg.dim*2+2, device = self.device))
		self.torch_goal_space = Uniform(
			low=torch.tensor([-2.0,-1.0], device=self.device), high=torch.tensor([2.0,1.0], device=self.device))
		
		self.reset()

	def step(self, action):
		# check if obj is attached
		self.attached0 = (torch.norm(self.obj - self.pos0, dim=-1) < self.cfg.err) \
			& (self.grip0 < 0)
		self.attached1 = (torch.norm(self.obj - self.pos1, dim=-1) < self.cfg.err) \
			& (self.grip1 < 0)   
		both_attached = self.attached0 & self.attached1   
		# move agent
		self.num_step += 1
		action = torch.clamp(action, self.torch_action_space.low,
												self.torch_action_space.high)
		# self.pos0[~both_attached] = torch.clamp(self.pos0 + action[..., :self.cfg.dim+1]*self.cfg.vel,
		#                       self.torch_space0.low, self.torch_space0.high)[~both_attached]
		self.grip0 = torch.clamp(action[..., self.cfg.dim],-1, 1) 
		self.pos0 = torch.clamp(self.pos0 + action[..., :self.cfg.dim]*self.cfg.vel,
													self.torch_space0.low, self.torch_space0.high)
		self.grip1 = torch.clamp(action[..., self.cfg.dim*2+1], -1, 1) 
		self.pos1 = torch.clamp(self.pos1 + action[..., self.cfg.dim+1:self.cfg.dim*2+1]*self.cfg.vel,
													self.torch_space1.low, self.torch_space1.high)
		# move obj with agent
		if self.use_gripper:
			old_pos = self.obj[both_attached]
			self.obj[self.attached0] = self.pos0[self.attached0]
			self.obj[self.attached1] = self.pos1[self.attached1]
			self.obj[both_attached] = old_pos
		else:
			self.obj[self.attached0 & ~both_attached] = self.pos0[self.attached0 & ~both_attached]
			self.obj[self.attached1 & ~both_attached] = self.pos1[self.attached1 & ~both_attached]
			goal_side = self.goal[..., 0] > 0
			need_handover0 = torch.logical_and((goal_side), self.pos0[..., 0] > -0.005)
			need_handover1 = torch.logical_and((~goal_side), self.pos1[..., 0] < 0.005)
			self.obj[both_attached & need_handover0] = self.pos1[both_attached & need_handover0]
			self.obj[both_attached & need_handover1] = self.pos0[both_attached & need_handover1]
		# compute states
		d = torch.norm(self.obj - self.goal, dim=-1)
		if_reach = (d < self.cfg.err)
		# if reached, not update reach step
		self.reach_step[~if_reach] = self.num_step[~if_reach]
		reward = self.compute_reward(self.obj, self.goal, None)
		info = torch.cat((
			if_reach.type(torch.float32).unsqueeze(-1),  # success
			self.num_step.type(torch.float32).unsqueeze(-1),  # step
			self.obj),  # achieved goal
			dim=-1)
		done = torch.logical_or(
			(self.num_step >= self.cfg.max_steps), (self.num_step - self.reach_step) > 4).type(torch.float32)
		if self.cfg.auto_reset:
			env_idx = torch.where(done > 1-1e-3)[0] # to avoid nan
			self.reset(env_idx)
		return self.get_obs(), reward, done, info

	def reset(self, env_idx = None):
		if env_idx is None:
			env_idx = torch.arange(self.cfg.num_envs)
		num_reset_env = env_idx.shape[0]
		self.num_step[env_idx] = 0
		self.reach_step[env_idx] = 0
		self.goal[env_idx] = self.torch_goal_space.sample((num_reset_env,))
		self.pos0[env_idx] = self.torch_space0.sample((num_reset_env,))
		self.pos1[env_idx] = self.torch_space1.sample((num_reset_env,))
		self.obj[env_idx] = self.torch_goal_space.sample((num_reset_env,))
		self.attached0[env_idx] = False
		self.attached1[env_idx] = False
		return self.get_obs()

	def render(self):
		if self.num_step[0] == 1:
			self.data = [[self.pos0.clone(), self.pos1.clone(), self.obj.clone()]]
		else:
			self.data.append([self.pos0.clone(), self.pos1.clone(), self.obj.clone()])
		if self.num_step[0] == self.cfg.max_steps:
			fig, ax = plt.subplots(1, self.cfg.num_envs)
			if self.cfg.dim == 2:
				for t, env_id in itertools.product(range(self.cfg.max_steps), range(self.cfg.num_envs)):
					# object
					o_x = self.data[t][2][env_id][0].detach().cpu()
					o_y = self.data[t][2][env_id][1].detach().cpu()
					ax[env_id].plot(o_x, o_y, 'o', color=[
													0, 1, 0, t/self.cfg.max_steps], markersize=10)
					# agent
					a_x = self.data[t][0][env_id][0].detach().cpu()
					a_y = self.data[t][0][env_id][1].detach().cpu()
					ax[env_id].plot(a_x, a_y, 'o', color=[
													0, 0, 1, t/self.cfg.max_steps])
					# agent
					a_x = self.data[t][1][env_id][0].detach().cpu()
					a_y = self.data[t][1][env_id][1].detach().cpu()
					ax[env_id].plot(a_x, a_y, 'o', color=[
													1, 0, 0, t/self.cfg.max_steps])
				for env_id in range(self.cfg.num_envs):
					x = self.goal[env_id][0].detach().cpu()
					y = self.goal[env_id][1].detach().cpu()
					ax[env_id].plot(x, y, 'rx')
					o_x = self.data[0][2][env_id][0].detach().cpu()
					o_y = self.data[0][2][env_id][1].detach().cpu()
					ax[env_id].plot(o_x, o_y, 'o', color=[
													1, 1, 0, 1], markersize=10)
				plt.show()
			elif self.cfg.dim == 3:
				fig, ax = plt.subplots(self.cfg.num_envs, 1)
				for i in range(self.cfg.num_envs):
					for i, d in enumerate(self.data[i]):
						ax[i].scatter(d[0], d[1], d[2], 'o', color=[0, 0, 1, i/50])
				plt.show()

	def get_obs(self):
		return torch.cat((
			# self.pos0, self.grip0.unsqueeze(-1), 
			# self.pos1, self.grip1.unsqueeze(-1), 
			self.pos0-self.torch_space0_mean, self.grip0.unsqueeze(-1), 
			self.pos1-self.torch_space1_mean, self.grip1.unsqueeze(-1), 
			self.obj, self.goal), dim=-1)

	def compute_reward(self, ag, dg, info):
		return -(torch.norm(ag-dg,dim=-1) > self.cfg.err).type(torch.float32)

	def ezpolicy(self, obs):
		obj_pos = obs[..., 2*self.cfg.dim+2:3*self.cfg.dim+2]
		goal = obs[..., 3*self.cfg.dim+2:] 
		goal_side = goal[..., 0] > 0
		pos0 = obs[..., :self.cfg.dim] + self.torch_space0_mean
		grip0 = obs[..., self.cfg.dim]
		pos1 = obs[..., self.cfg.dim+1:self.cfg.dim*2+1] + self.torch_space1_mean
		grip1 = obs[..., self.cfg.dim*2+1]
		attached0 = torch.logical_and((torch.norm(obj_pos - pos0, dim=-1) < self.cfg.err), 
		(grip0 < 0))
		attached1 = torch.logical_and((torch.norm(obj_pos - pos1, dim=-1) < self.cfg.err), 
		(grip1 < 0))
		# move 0
		delta0 = - pos0 + obj_pos
		delta0[attached0] =  (- pos0 + goal)[attached0]
		vel0 = -torch.ones((self.cfg.num_envs, self.cfg.dim+1), device = self.device)
		vel0[..., :self.cfg.dim] = (delta0 / torch.norm(delta0, dim=-1, keepdim=True))
		need_handover0 = torch.logical_and((goal_side), pos0[..., 0] > -0.005)
		vel0[need_handover0, -1] = 1 
		# move 1
		delta1 = - pos1 + obj_pos
		delta1[attached1] =  (- pos1 + goal)[attached1]
		vel1 = -torch.ones((self.cfg.num_envs, self.cfg.dim+1), device = self.device)
		vel1[..., :self.cfg.dim] = (delta1 / torch.norm(delta1, dim=-1, keepdim=True))
		need_handover1 = torch.logical_and((~goal_side), pos1[..., 0] < 0.005)
		vel1[need_handover1, -1] = 1 
		reached = (torch.norm(obj_pos-goal, dim=-1) < self.cfg.err).unsqueeze(-1)
		return torch.cat((vel0, vel1), dim=-1) * (~reached)

	def parse_info(self, info):
		is_success = info[..., 0:1]
		step = info[..., 1:2]
		achieved_goal = info[..., 2:self.cfg.dim+2]
		return is_success, step, achieved_goal

gym.register(id='ReachToy-v0', entry_point=ReachToyEnv)
gym.register(id='PNPToy-v0', entry_point=PNPToyEnv)
gym.register(id='HandoverToy-v0', entry_point=HandoverToyEnv)

if __name__ == '__main__':
	env = PNPToyEnv(gpu_id=-1, err=0.2, auto_reset=False, num_goals=2)
	obs = env.reset()
	for _ in range(env._max_episode_steps):
		act = env.ezpolicy(obs)
		obs, reward, done, info = env.step(act)
		env.render()