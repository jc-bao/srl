import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ActorSAC(nn.Module):
  def __init__(self, mid_dim, state_dim, action_dim, net_type='mlp', other_dims=None):
    super().__init__()
    self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
    self.net_a_avg = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, action_dim))  # the average of action
    self.net_a_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, action_dim))  # the log_std of action
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
  def __init__(self, mid_dim, state_dim, action_dim, net_type='mlp', other_dims = None):
    super().__init__()
    if net_type == 'deepset': 
      self.net_state = ActorDeepsetBlock(mid_dim, other_dims.shared_dim, other_dims.seperate_dim, other_dims.goal_dim, other_dims.num_goals)
    elif net_type == 'mlp':
      self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
      nn.Linear(mid_dim, mid_dim), nn.ReLU())
    else:
      raise NotImplementedError
    self.net_a_avg = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, action_dim))  # the average of action
    self.net_a_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, action_dim))  # the log_std of action
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
    a_log_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
    return a_log_std

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
  def __init__(self, mid_dim, state_dim, action_dim, net_type='mlp', other_dims=None):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                             nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                             nn.Linear(mid_dim, action_dim))

    # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
    self.a_std_log = nn.Parameter(torch.zeros(
      (1, action_dim)) - 0.5, requires_grad=True)
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
    log_prob = -(self.a_std_log + self.sqrt_2pi_log + delta)  # new_logprob

    return log_prob

  def get_logprob_entropy(self, state, action):
    a_avg = self.net(state)
    a_std = self.a_std_log.exp()

    delta = ((a_avg - action) / a_std).pow(2) * 0.5
    logprob = -(self.a_std_log + self.sqrt_2pi_log +
                delta).sum(1)  # new_logprob

    dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
    return logprob, dist_entropy

  def get_old_logprob(self, _action, noise):  # noise = action - a_noise
    delta = noise.pow(2) * 0.5
    return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

  @staticmethod
  def get_a_to_e(action):
    return action.tanh()


class Critic(nn.Module):
  def __init__(self, mid_dim, state_dim, action_dim):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                             nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                             nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                             nn.Linear(mid_dim, 1))

  def forward(self, state, action):
    return self.net(torch.cat((state, action), dim=1))  # q value


class CriticTwin(nn.Module):  # shared parameter
  def __init__(self, mid_dim, state_dim, action_dim):
    super().__init__()
    self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, mid_dim), nn.ReLU())  # concat(state, action)
    self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, 1))  # q1 value
    self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, 1))  # q2 value

  def forward(self, state, action):
    return torch.add(*self.get_q1_q2(state, action)) / 2.  # mean Q value

  def get_q_min(self, state, action):
    return torch.min(*self.get_q1_q2(state, action))  # min Q value

  def get_q1_q2(self, state, action):
    tmp = self.net_sa(torch.cat((state, action), dim=1))
    return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


class CriticREDq(nn.Module):  # modified REDQ (Randomized Ensemble Double Q-learning)
  def __init__(self, mid_dim, state_dim, action_dim, net_type = 'mlp', other_dims = None):
    super().__init__()
    self.critic_num = 8
    self.critic_list = list()
    for critic_id in range(self.critic_num):
      if net_type == 'deepset': 
        child_cri_net = CriticDeepset(mid_dim, action_dim, other_dims.shared_dim, other_dims.seperate_dim, other_dims.goal_dim, other_dims.num_goals)
      elif net_type == 'mlp':
        child_cri_net = Critic(mid_dim, state_dim, action_dim).net
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


class CriticREDQ(nn.Module):  # modified REDQ (Randomized Ensemble Double Q-learning)
  def __init__(self, mid_dim, state_dim, action_dim):
    super().__init__()
    self.critic_num = 8
    self.sample_num = 2
    self.critic_list = list()
    for critic_id in range(self.critic_num):
      child_cri_net = Critic(mid_dim, state_dim, action_dim).net
      setattr(self, f'critic{critic_id:02}', child_cri_net)
      self.critic_list.append(child_cri_net)

  def forward(self, state, action):
    # mean Q value
    return self.get_q_values(state, action).mean(dim=1, keepdim=True)

  def get_q_min(self, state, action):
    tensor_qs = self.get_q_values(state, action, self.sample_num)
    q_min = torch.min(tensor_qs, dim=1, keepdim=True)[0]  # min Q value
    return q_min

  def get_q_values(self, state, action, num=None):
    if num is None:
      num = self.critic_num
    tensor_sa = torch.cat((state, action), dim=1)
    idx = np.random.choice(self.critic_num, num, replace=False)
    tensor_qs = [self.critic_list[i](tensor_sa) for i in idx]
    tensor_qs = torch.cat(tensor_qs, dim=1)
    return tensor_qs  # multiple Q values


class CriticPPO(nn.Module):
  def __init__(self, mid_dim, state_dim, _action_dim, net_type='mlp', other_dims=None):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                             nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                             nn.Linear(mid_dim, 1))

  def forward(self, state):
    return self.net(state)  # advantage value


# TODO make it sequential 
class ActorDeepsetBlock(nn.Module):
  def __init__(self, mid_dim, shared_dim, seperate_dim, goal_dim, num_goals):
    # state_dim=[shared_dim, seperate_dim, goal_dim, num_goals]
    super().__init__()
    self.shared_dim = shared_dim
    self.seperate_dim = seperate_dim
    self.goal_dim = goal_dim
    self.num_goals = num_goals
    assert self.goal_dim % self.num_goals ==0, f'goal dim {self.goal_dim} should be divisible by num goals {self.num_goals}'
    self.single_goal_dim = self.goal_dim // self.num_goals
    assert self.seperate_dim % self.num_goals == 0, f'seperate dim {self.seperate_dim} should be divisible by num goals {self.num_goals}'
    self.single_seperate_dim = self.seperate_dim // self.num_goals
    self.fc1 = nn.Linear(self.shared_dim+self.single_seperate_dim+self.single_goal_dim, mid_dim)
    self.fc2 = nn.Linear(mid_dim, mid_dim)

  def forward(self, state):
    grip = state[..., :self.shared_dim]
    obj = state[..., self.shared_dim:self.shared_dim + self.seperate_dim].reshape(-1, self.num_goals, self.single_seperate_dim)
    g = state[..., self.shared_dim+self.seperate_dim:self.shared_dim + self.seperate_dim+self.goal_dim].reshape(-1, self.num_goals, self.single_goal_dim) 
    grip = grip.unsqueeze(1).repeat(1, self.num_goals, 1)
    x = torch.cat((grip, obj, g), -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return x.mean(dim=1)


class CriticDeepset(nn.Module):
  def __init__(self, mid_dim, action_dim, shared_dim, seperate_dim, goal_dim, num_goals):
    super().__init__()
    self.shared_dim = shared_dim
    self.seperate_dim = seperate_dim
    self.goal_dim = goal_dim
    self.num_goals = num_goals
    self.action_dim = action_dim
    assert self.goal_dim % self.num_goals ==0, f'goal dim {self.goal_dim} should be divisible by num goals {self.num_goals}'
    self.single_goal_dim = self.goal_dim // self.num_goals
    assert self.seperate_dim % self.num_goals == 0, f'seperate dim {self.seperate_dim} should be divisible by num goals {self.num_goals}'
    self.single_seperate_dim = self.seperate_dim // self.num_goals
    self.net_in = nn.Sequential(
      nn.Linear(self.shared_dim+self.single_seperate_dim+self.single_goal_dim+action_dim, mid_dim), nn.ReLU(),
      nn.Linear(mid_dim, mid_dim), nn.ReLU(),)
    self.net_out = nn.Sequential(
      nn.Linear(mid_dim, mid_dim), nn.ReLU(),
      nn.Linear(mid_dim, 1)
    )

  def forward(self, state, action=None):
    if action is None:
      action = state[..., -self.action_dim:]
    obj = state[..., self.shared_dim:self.shared_dim + self.seperate_dim].reshape(-1, self.num_goals, self.single_seperate_dim)
    g = state[..., self.shared_dim+self.seperate_dim:self.shared_dim + self.seperate_dim+self.goal_dim].reshape(-1, self.num_goals, self.single_goal_dim) 
    grip = state[..., :self.shared_dim].unsqueeze(1).repeat(1, self.num_goals, 1)
    action = action.unsqueeze(1).repeat(1, self.num_goals, 1)
    x = torch.cat((grip, obj, g, action), -1) # batch, obj, feature
    x = self.net_in(x).mean(dim=1)
    return self.net_out(x)