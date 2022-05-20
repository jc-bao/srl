from attrdict import AttrDict
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
	def __init__(self, cfg):
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		if cfg.net_type == 'deepset':
			self.net = nn.Sequential(
				ActorDeepsetBlock(cfg),
				*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()] *
				(self.cfg.net_layer-self.cfg.shared_net_layer-1),
				# *[nn.Linear(cfg.net_dim, cfg.net_dim),nn.ReLU()]*(self.cfg.net_layer-4),
				nn.Linear(cfg.net_dim, EP.action_dim))
		elif cfg.net_type == 'attn':
			self.net = nn.Sequential(
				ActorAttnBlock(cfg),
				*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()] *
				(self.cfg.net_layer-self.cfg.shared_net_layer-1),
				nn.Linear(cfg.net_dim, EP.action_dim))
		elif cfg.net_type == 'mlp':
			self.net = nn.Sequential(
				nn.Linear(EP.state_dim, cfg.net_dim), nn.ReLU(),
				# nn.Linear(cfg.net_dim, cfg.net_dim),nn.ReLU(),
				# nn.Linear(cfg.net_dim, cfg.net_dim),nn.ReLU(),
				*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()] * \
				(self.cfg.net_layer-2),
				nn.Linear(cfg.net_dim, EP.action_dim),
			)
		else:
			raise NotImplementedError(f'net_type {cfg.net_type} not implemented')
		# standard deviation of exploration action noise
		self.explore_noise = cfg.explore_noise

	def forward(self, state):
		return self.net(state).tanh()  # action.tanh()

	def get_action(self, state):  # for exploration
		action = self.net(state).tanh()
		noise = (torch.randn_like(action) *
						 self.explore_noise).clamp(-0.5, 0.5)
		return (action + noise).clamp(-1.0, 1.0)

	def get_action_noise(self, state, action_std):
		action = self.net(state).tanh()
		noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
		return (action + noise).clamp(-1.0, 1.0)


class ActorSAC(nn.Module):
	def __init__(self, cfg):
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		self.net_state = nn.Sequential(nn.Linear(EP.state_dim, cfg.net_dim), nn.ReLU(),
																	 nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU(), )
		self.net_a_avg = nn.Sequential(nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU(),
																	 nn.Linear(cfg.net_dim, EP.action_dim))  # the average of action
		self.net_a_std = nn.Sequential(nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU(),
																	 nn.Linear(cfg.net_dim, EP.action_dim))  # the log_std of action
		self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

	def forward(self, state):
		tmp = self.net_state(state)
		return self.net_a_avg(tmp).tanh()  # action

	def get_action(self, state):
		t_tmp = self.net_state(state)
		a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
		a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
		return torch.normal(a_avg, a_std).tanh()  # re-parameterize

	def get_action_logprob(self, state):
		t_tmp = self.net_state(state)
		a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
		a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
		a_std = a_std_log.exp()

		noise = torch.randn_like(a_avg, requires_grad=True)
		a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()

		logprob = a_std_log + self.log_sqrt_2pi + \
			noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
		# fix logprob using the derivative of action.tanh()
		logprob = logprob + (-a_tan.pow(2) + 1.000001).log()
		return a_tan, logprob.sum(1, keepdim=True)  # todo negative logprob


class ActorFixSAC(nn.Module):
	def __init__(self, cfg):
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		if cfg.net_type == 'deepset':
			self.net_state = nn.Sequential(
				ActorDeepsetBlock(cfg),
				*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()] *
				(self.cfg.net_layer-self.cfg.shared_net_layer-1),)
		elif cfg.net_type == 'attn':
			self.net_state = nn.Sequential(
				ActorAttnBlock(cfg),
				*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()] *
				(self.cfg.net_layer-self.cfg.shared_net_layer-1))
		elif cfg.net_type == 'mlp':
			self.net_state = nn.Sequential(
				nn.Linear(EP.state_dim, cfg.net_dim), nn.ReLU(),
				*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()] * \
				(self.cfg.net_layer-2),
			)
		else:
			raise NotImplementedError(f'net_type {cfg.net_type} not implemented')
		self.net_a_avg = nn.Linear(cfg.net_dim, EP.action_dim)  # the average of action
		self.net_a_std = nn.Linear(cfg.net_dim, EP.action_dim)  # the log_std of action
		self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
		self.soft_plus = nn.Softplus()

	def forward(self, state):
		tmp = self.net_state(state)
		return self.net_a_avg(tmp).tanh()  # action

	def get_action(self, state):
		t_tmp = self.net_state(state)
		a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
		a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
		return torch.normal(a_avg, a_std).tanh()  # re-parameterize

	def get_a_log_std(self, state):
		t_tmp = self.net_state(state)
		return self.net_a_std(t_tmp).clamp(-20, 2).exp()

	def get_logprob(self, state, action):
		t_tmp = self.net_state(state)
		a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
		a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
		a_std = a_std_log.exp()

		'''add noise to a_noise in stochastic policy'''
		a_noise = a_avg + a_std * torch.randn_like(a_avg, requires_grad=True)
		noise = a_noise - action  # todo

		log_prob = a_std_log + self.log_sqrt_2pi + \
			noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
		log_prob += (np.log(2.) - a_noise - self.soft_plus(-2. *
																											 a_noise)) * 2.  # better than below
		return log_prob

	def get_action_logprob(self, state):
		t_tmp = self.net_state(state)
		a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
		a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
		a_std = a_std_log.exp()

		'''add noise to a_noise in stochastic policy'''
		noise = torch.randn_like(a_avg, requires_grad=True)
		a_noise = a_avg + a_std * noise
		# Can only use above code instead of below, because the tensor need gradients here.
		# a_noise = torch.normal(a_avg, a_std, requires_grad=True)

		'''compute log_prob according to mean and std of a_noise (stochastic policy)'''
		# self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
		log_prob = a_std_log + self.log_sqrt_2pi + \
			noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
		"""same as below:
				from torch.distributions.normal import Normal
				log_prob = Normal(a_avg, a_std).log_prob(a_noise)
				# same as below:
				a_delta = (a_avg - a_noise).pow(2) /(2*a_std.pow(2))
				log_prob = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
				"""

		'''fix log_prob of action.tanh'''
		log_prob += (np.log(2.) - a_noise - self.soft_plus(-2. *
																											 a_noise)) * 2.  # better than below
		"""same as below:
				epsilon = 1e-6
				a_noise_tanh = a_noise.tanh()
				log_prob = log_prob - (1 - a_noise_tanh.pow(2) + epsilon).log()

				Thanks for:
				https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/actor.py#L37
				↑ MIT License， Thanks for https://www.zhihu.com/people/Z_WXCY 2ez4U
				They use action formula that is more numerically stable, see details in the following link
				https://pytorch.org/docs/stable/_modules/torch/distributions/transforms.html#TanhTransform
				https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f
				"""
		return a_noise.tanh(), log_prob.sum(1, keepdim=True)


class ActorPPO(nn.Module):
	def __init__(self, cfg):
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		self.net = nn.Sequential(nn.Linear(EP.state_dim, cfg.net_dim), nn.ReLU(),
														 nn.Linear(
			cfg.net_dim, cfg.net_dim), nn.ReLU(),
			nn.Linear(cfg.net_dim, EP.action_dim))

		# the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
		self.a_std_log = nn.Parameter(torch.zeros(
			(1, EP.action_dim)) - 0., requires_grad=True)
		self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

	def forward(self, state):
		return self.net(state).tanh()  # action.tanh()

	def get_action(self, state):
		a_avg = self.net(state)
		a_std = self.a_std_log.exp()

		noise = torch.randn_like(a_avg)
		action = a_avg + noise * a_std
		return action, noise

	def get_logprob(self, state, action):
		a_avg = self.net(state)
		a_std = self.a_std_log.exp()

		delta = ((a_avg - action) / a_std).pow(2) * 0.5
		return -(self.a_std_log + self.sqrt_2pi_log + delta)

	def get_logprob_entropy(self, state, action):
		a_avg = self.net(state)
		a_std = self.a_std_log.exp()

		delta = ((a_avg - action) / a_std).pow(2) * 0.5
		logprob = -(self.a_std_log + self.sqrt_2pi_log +
								delta).sum(-1)  # new_logprob

		dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
		return logprob, dist_entropy

	def get_old_logprob(self, _action, noise):  # noise = action - a_noise
		delta = noise.pow(2) * 0.5
		# old_logprob
		return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(-1)

	@staticmethod
	def get_a_to_e(action):
		return action.tanh()


class Critic(nn.Module):
	def __init__(self, cfg):
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		self.net = nn.Sequential(nn.Linear(EP.state_dim + EP.action_dim, cfg.net_dim), nn.ReLU(),
														 nn.Linear(
			cfg.net_dim, cfg.net_dim), nn.ReLU(),
			nn.Linear(
			cfg.net_dim, cfg.net_dim), nn.ReLU(),
			nn.Linear(cfg.net_dim, 1))

	def forward(self, state, action):
		return self.net(torch.cat((state, action), dim=1))  # q value


class CriticTwin(nn.Module):  # shared parameter
	def __init__(self, cfg):
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		if self.cfg.net_type == 'deepset':
			self.net_sa = CriticDeepsetBlock(cfg)
		elif self.cfg.net_type == 'attn':
				self.net_sa = CriticAttnBlock(cfg)
		elif self.cfg.net_type == 'mlp':
			self.net_sa = nn.Sequential(nn.Linear(EP.state_dim + EP.action_dim, cfg.net_dim), nn.ReLU(),
																	*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()]*(self.cfg.shared_net_layer-1))  # concat(state, action)
		else:
			raise NotImplementedError
		self.net_q1 = nn.Sequential(*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()]*(self.cfg.net_layer-1-self.cfg.shared_net_layer),
																nn.Linear(cfg.net_dim, 1))  # q1 value
		self.net_q2 = nn.Sequential(*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()]*(self.cfg.net_layer-1-self.cfg.shared_net_layer),
																nn.Linear(cfg.net_dim, 1))  # q2 value

	def forward(self, state, action):
		return torch.mean(self.get_q_all(state, action))

	def get_q_min(self, state, action):
		# min Q value
		return torch.min(self.get_q_all(state, action), dim=-1, keepdim=True)[0]

	def get_q_all(self, state, action):
		tmp = self.net_sa(torch.cat((state, action), dim=1))
		# two Q values
		return torch.cat((self.net_q1(tmp), self.net_q2(tmp)), dim=-1)


class CriticRed(nn.Module):  # shared parameter
	def __init__(self, cfg):
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		if self.cfg.net_type == 'deepset':
			self.net_sa = nn.Sequential(
				CriticDeepsetBlock(cfg),
				*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()]*(self.cfg.net_layer-1-self.cfg.shared_net_layer)
			)
		elif self.cfg.net_type == 'attn':
			self.net_sa = CriticAttnBlock(cfg)
		elif self.cfg.net_type == 'mlp':
			self.net_sa = nn.Sequential(
				nn.Linear(EP.state_dim + EP.action_dim, cfg.net_dim), nn.ReLU(),
				*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()]*(self.cfg.net_layer-3))  # concat(state, action)
		else:
			raise NotImplementedError
		self.all_idx = torch.arange(self.cfg.q_num, device=self.cfg.device)
		self.net_q = nn.ModuleList()
		for _ in range(self.cfg.q_num):
			self.net_q.append(nn.Sequential(
				nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU(),
				nn.Linear(cfg.net_dim, 1)))  # q values

	def forward(self, state, action):
		return torch.mean(self.get_q_all(state, action))  # mean Q value

	def get_q_min(self, state, action):
		rand_idx = np.random.choice(self.cfg.q_num, size=(
			self.cfg.random_q_num,), replace=False)
		# min Q value
		return torch.min(self.get_q_all(state, action, idx=rand_idx), dim=-1, keepdim=True)[0]

	def get_q_all(self, state, action, idx=None):
		tmp = self.net_sa(torch.cat((state, action), dim=1))
		if idx is None:
			return torch.cat([self.net_q[i](tmp) for i in range(self.cfg.q_num)], dim=-1)
		else:
			# all Q values
			return torch.cat([self.net_q[i](tmp) for i in idx], dim=-1)


class CriticREDq(nn.Module):  # modified REDQ (Randomized Ensemble Double Q-learning)
	def __init__(self, cfg):
		super().__init__()
		self.critic_num = 8
		self.critic_list = list()
		for critic_id in range(self.critic_num):
			if cfg.net_type == 'deepset':
				child_cri_net = CriticDeepset(cfg)
			elif cfg.net_type == 'mlp':
				child_cri_net = Critic(cfg).net
			else:
				raise NotImplementedError
			setattr(self, f'critic{critic_id:02}', child_cri_net)
			self.critic_list.append(child_cri_net)

	def forward(self, state, action):
		# mean Q value
		return self.get_q_values(state, action).mean(dim=1, keepdim=True)

	def get_q_min(self, state, action):
		tensor_qs = self.get_q_values(state, action)
		q_min = torch.min(tensor_qs, dim=1, keepdim=True)[0]  # min Q value
		q_sum = tensor_qs.sum(dim=1, keepdim=True)  # mean Q value
		# better than min
		return (q_min * (self.critic_num * 0.5) + q_sum) / (self.critic_num * 1.5)

	def get_q_values(self, state, action):
		tensor_sa = torch.cat((state, action), dim=1)
		tensor_qs = [cri_net(tensor_sa) for cri_net in self.critic_list]
		tensor_qs = torch.cat(tensor_qs, dim=1)
		return tensor_qs  # multiple Q values


class CriticPPO(nn.Module):
	def __init__(self, cfg):
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		self.net = nn.Sequential(nn.Linear(EP.state_dim, cfg.net_dim), nn.ReLU(),
														 nn.Linear(
			cfg.net_dim, cfg.net_dim), nn.ReLU(),
			nn.Linear(cfg.net_dim, 1))

	def forward(self, state):
		return self.net(state)  # advantage value


# TODO make it sequential
class ActorDeepsetBlock(nn.Module):
	def __init__(self, cfg):
		# state_dim=[shared_dim, seperate_dim, goal_dim, num_goals]
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		self.shared_dim = EP.shared_dim
		self.seperate_dim = EP.seperate_dim
		self.goal_dim = EP.goal_dim
		self.num_goals = EP.num_goals
		assert self.goal_dim % self.num_goals == 0, f'goal dim {self.goal_dim} should be divisible by num goals {self.num_goals}'
		self.single_goal_dim = self.goal_dim // self.num_goals
		assert self.seperate_dim % self.num_goals == 0, f'seperate dim {self.seperate_dim} should be divisible by num goals {self.num_goals}'
		self.single_seperate_dim = self.seperate_dim // self.num_goals
		self.fc = nn.Sequential(
			nn.Linear(self.shared_dim + self.single_seperate_dim +
								self.single_goal_dim, cfg.net_dim), nn.ReLU(),
			*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()] *
			(self.cfg.shared_net_layer-1),
		)

	def forward(self, state):
		grip = state[..., :self.shared_dim]
		obj = state[..., self.shared_dim:self.shared_dim +
								self.seperate_dim].reshape(-1, self.num_goals, self.single_seperate_dim)
		g = state[..., self.shared_dim+self.seperate_dim:self.shared_dim +
							self.seperate_dim+self.goal_dim].reshape(-1, self.num_goals, self.single_goal_dim)
		grip = grip.unsqueeze(1).repeat(1, self.num_goals, 1)
		x = torch.cat((grip, obj, g), -1)
		x = self.fc(x)
		return x.mean(dim=1)

class ActorAttnBlock(nn.Module):
	def __init__(self, cfg):
		# state_dim=[shared_dim, seperate_dim, goal_dim, num_goals]
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		self.shared_dim = EP.shared_dim
		self.seperate_dim = EP.seperate_dim
		self.goal_dim = EP.goal_dim
		self.num_goals = EP.num_goals
		assert self.goal_dim % self.num_goals == 0, f'goal dim {self.goal_dim} should be divisible by num goals {self.num_goals}'
		self.single_goal_dim = self.goal_dim // self.num_goals
		assert self.seperate_dim % self.num_goals == 0, f'seperate dim {self.seperate_dim} should be divisible by num goals {self.num_goals}'
		self.single_seperate_dim = self.seperate_dim // self.num_goals
		if self.cfg.actor_pool_type == 'cross':
			self.query_embed = nn.Sequential(
				nn.Linear(self.shared_dim, cfg.net_dim), nn.ReLU())
			self.embed = nn.Sequential(
				nn.Linear(self.single_seperate_dim +
									self.single_goal_dim, cfg.net_dim), nn.ReLU())
			self.cross_attn = nn.MultiheadAttention(self.cfg.net_dim, self.cfg.n_head, dropout=0.0)
		else:
			self.embed = nn.Sequential(
				nn.Linear(self.shared_dim + self.single_seperate_dim +
									self.single_goal_dim, cfg.net_dim), nn.ReLU())
		self.enc = nn.Sequential(*[EncoderLayer(self.cfg.net_dim, n_head=self.cfg.n_head, dim_ff=self.cfg.net_dim,
														 pre_lnorm=True, dropout=0.0) for _ in range(self.cfg.shared_net_layer-1)])
		if self.cfg.actor_pool_type == 'berd':
			self.berd_query = nn.parameter.Parameter(torch.randn(self.cfg.net_dim))
			self.berd_attn = nn.MultiheadAttention(self.cfg.net_dim, self.cfg.n_head, dropout=0.0)

	def forward(self, state):
		grip = state[..., :self.shared_dim]
		obj = state[..., self.shared_dim:self.shared_dim +
								self.seperate_dim].reshape(-1, self.num_goals, self.single_seperate_dim)
		g = state[..., self.shared_dim+self.seperate_dim:self.shared_dim +
							self.seperate_dim+self.goal_dim].reshape(-1, self.num_goals, self.single_goal_dim)
		if self.cfg.actor_pool_type == 'cross':
			query = self.query_embed(grip).unsqueeze(0)
			x = torch.cat((obj, g), -1)
		else:
			grip = grip.unsqueeze(1).repeat(1, self.num_goals, 1)
			x = torch.cat((grip, obj, g), -1)
		x = self.embed(x).transpose(0, 1) # Tensor(num_goals, num_envs, net_dim)
		token = self.enc(x)
		if self.cfg.actor_pool_type == 'mean':
			x = token.mean(dim=0)
		elif self.cfg.actor_pool_type == 'max':
			return token.max(dim=0)[0]
		elif self.cfg.actor_pool_type == 'berd':
			return self.berd_attn(self.berd_query.tile(1, token.shape[1], 1), token, token)[0].squeeze(0)
		elif self.cfg.actor_pool_type == 'cross':
			return self.cross_attn(query, token, token)[0].squeeze(0)
		else:
			raise NotImplementedError

class CriticDeepsetBlock(nn.Module):
	def __init__(self, cfg):
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		self.shared_dim = EP.shared_dim
		self.seperate_dim = EP.seperate_dim
		self.goal_dim = EP.goal_dim
		self.num_goals = EP.num_goals
		self.action_dim = EP.action_dim
		assert self.goal_dim % self.num_goals == 0, f'goal dim {self.goal_dim} should be divisible by num goals {self.num_goals}'
		self.single_goal_dim = self.goal_dim // self.num_goals
		assert self.seperate_dim % self.num_goals == 0, f'seperate dim {self.seperate_dim} should be divisible by num goals {self.num_goals}'
		self.single_seperate_dim = self.seperate_dim // self.num_goals
		self.net_in = nn.Sequential(
			nn.Linear(self.shared_dim+self.single_seperate_dim +
								self.single_goal_dim+EP.action_dim, cfg.net_dim), nn.ReLU(),
			*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()]*(self.cfg.shared_net_layer-1),)

	def forward(self, state, action=None):
		if action is None:
			action = state[..., -self.action_dim:]
		obj = state[..., self.shared_dim:self.shared_dim +
								self.seperate_dim].reshape(-1, self.num_goals, self.single_seperate_dim)
		g = state[..., self.shared_dim+self.seperate_dim:self.shared_dim +
							self.seperate_dim+self.goal_dim].reshape(-1, self.num_goals, self.single_goal_dim)
		grip = state[..., :self.shared_dim].unsqueeze(
			1).repeat(1, self.num_goals, 1)
		action = action.unsqueeze(1).repeat(1, self.num_goals, 1)
		x = torch.cat((grip, obj, g, action), -1)  # batch, obj, feature
		return self.net_in(x).mean(dim=1)


class CriticAttnBlock(nn.Module):
	def __init__(self, cfg):
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		self.shared_dim = EP.shared_dim
		self.seperate_dim = EP.seperate_dim
		self.goal_dim = EP.goal_dim
		self.num_goals = EP.num_goals
		self.action_dim = EP.action_dim
		assert self.goal_dim % self.num_goals == 0, f'goal dim {self.goal_dim} should be divisible by num goals {self.num_goals}'
		self.single_goal_dim = self.goal_dim // self.num_goals
		assert self.seperate_dim % self.num_goals == 0, f'seperate dim {self.seperate_dim} should be divisible by num goals {self.num_goals}'
		self.single_seperate_dim = self.seperate_dim // self.num_goals
		if self.cfg.actor_pool_type == 'cross':
			self.query_embed = nn.Sequential(
				nn.Linear(self.shared_dim+EP.action_dim, cfg.net_dim), nn.ReLU())
			self.embed = nn.Sequential(
				nn.Linear(self.single_seperate_dim +
									self.single_goal_dim, cfg.net_dim), nn.ReLU())
			self.cross_attn = nn.MultiheadAttention(self.cfg.net_dim, self.cfg.n_head, dropout=0.0)
		else:
			self.embed = nn.Sequential(
				nn.Linear(self.shared_dim + self.single_seperate_dim +
									self.single_goal_dim+EP.action_dim, cfg.net_dim), nn.ReLU())
		self.enc = nn.Sequential(*[EncoderLayer(self.cfg.net_dim, n_head=self.cfg.n_head, dim_ff=self.cfg.net_dim,
														 pre_lnorm=True, dropout=0.0) for _ in range(self.cfg.shared_net_layer-1)])
		if self.cfg.critic_pool_type == 'berd':
			self.berd_query = nn.parameter.Parameter(torch.randn(self.cfg.net_dim))
			self.berd_attn = nn.MultiheadAttention(self.cfg.net_dim, self.cfg.n_head, dropout=0.0)

	def forward(self, state, action=None):
		if action is None:
			action = state[..., -self.action_dim:]
		obj = state[..., self.shared_dim:self.shared_dim +
								self.seperate_dim].reshape(-1, self.num_goals, self.single_seperate_dim)
		g = state[..., self.shared_dim+self.seperate_dim:self.shared_dim +
							self.seperate_dim+self.goal_dim].reshape(-1, self.num_goals, self.single_goal_dim)
		if self.cfg.actor_pool_type == 'cross':
			grip = state[..., :self.shared_dim]
			query = self.query_embed(torch.cat([grip,action], dim=-1)).unsqueeze(0)
			x = torch.cat((obj, g), -1)
		else:
			grip = state[..., :self.shared_dim].unsqueeze(
				1).repeat(1, self.num_goals, 1)
			action = action.unsqueeze(1).repeat(1, self.num_goals, 1)
			x = torch.cat((grip, obj, g, action), -1)  # batch, obj, feature
		x = self.embed(x).transpose(0, 1)
		token = self.enc(x)
		if self.cfg.critic_pool_type == 'mean':
			return token.mean(dim=0)
		elif self.cfg.critic_pool_type == 'max':
			return token.max(dim=0)[0]
		elif self.cfg.critic_pool_type == 'berd':
			return self.berd_attn(self.berd_query.tile(1, token.shape[1], 1), token, token)[0].squeeze(0)
		elif self.cfg.critic_pool_type == 'cross':
			return self.cross_attn(query, token, token)[0].squeeze(0)
		else: 
			raise NotImplementedError


class EncoderLayer(nn.Module):
	"""Adapted from: https://github.com/jwang0306/transformer-pytorch."""

	def __init__(self, hidden_size, n_head, dim_ff, pre_lnorm, dropout=0.0):
		super(EncoderLayer, self).__init__()
		# self-attention part
		self.self_attn = nn.MultiheadAttention(
			hidden_size, n_head, dropout=dropout)
		self.dropout = nn.Dropout(dropout)
		self.self_attn_norm = nn.LayerNorm(hidden_size)

		# feed forward network part
		self.pff = nn.Sequential(
			nn.Linear(hidden_size, dim_ff),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(dim_ff, hidden_size),
			nn.Dropout(dropout)
		)
		self.pff_norm = nn.LayerNorm(hidden_size)
		self.pre_lnorm = pre_lnorm

	def forward(self, src, src_mask=None):
		if self.pre_lnorm:
			pre = self.self_attn_norm(src)
			# residual connection
			src = src + self.dropout(self.self_attn(pre, pre, pre, src_mask)[0])
			pre = self.pff_norm(src)
			src = src + self.pff(pre)  # residual connection
		else:
			# residual connection + layerNorm
			src2 = self.dropout(self.self_attn(src, src, src, src_mask)[0])
			src = self.self_attn_norm(src + src2)
			# residual connection + layerNorm
			src = self.pff_norm(src + self.pff(src))
		return src


class CriticDeepset(nn.Module):
	def __init__(self, cfg):
		self.cfg, EP = filter_cfg(cfg)
		super().__init__()
		self.shared_dim = EP.shared_dim
		self.seperate_dim = EP.seperate_dim
		self.goal_dim = EP.goal_dim
		self.num_goals = EP.num_goals
		self.action_dim = EP.action_dim
		assert self.goal_dim % self.num_goals == 0, f'goal dim {self.goal_dim} should be divisible by num goals {self.num_goals}'
		self.single_goal_dim = self.goal_dim // self.num_goals
		assert self.seperate_dim % self.num_goals == 0, f'seperate dim {self.seperate_dim} should be divisible by num goals {self.num_goals}'
		self.single_seperate_dim = self.seperate_dim // self.num_goals
		self.net_in = nn.Sequential(
			nn.Linear(self.shared_dim+self.single_seperate_dim +
								self.single_goal_dim+EP.action_dim, cfg.net_dim), nn.ReLU(),
			nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU(),)
		self.net_out = nn.Sequential(
			nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU(),
			nn.Linear(cfg.net_dim, 1)
		)

	def forward(self, state, action=None):
		if action is None:
			action = state[..., -self.action_dim:]
		obj = state[..., self.shared_dim:self.shared_dim +
								self.seperate_dim].reshape(-1, self.num_goals, self.single_seperate_dim)
		g = state[..., self.shared_dim+self.seperate_dim:self.shared_dim +
							self.seperate_dim+self.goal_dim].reshape(-1, self.num_goals, self.single_goal_dim)
		grip = state[..., :self.shared_dim].unsqueeze(
			1).repeat(1, self.num_goals, 1)
		action = action.unsqueeze(1).repeat(1, self.num_goals, 1)
		x = torch.cat((grip, obj, g, action), -1)  # batch, obj, feature
		x = self.net_in(x).mean(dim=1)
		return self.net_out(x)


def filter_cfg(config):
	cfg, EP = AttrDict(), AttrDict()
	# filter out isaac object to make function pickleable
	for k, v in config.env_params.items():
		if not hasattr(v, '__call__'):
			EP[k] = v
	for k, v in config.items():
		if k != 'env_params':
			cfg[k] = config[k]
	return cfg, EP
