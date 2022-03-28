import torch
import numpy as np
import torch.nn as nn


class ActorSAC(nn.Module):
  def __init__(self, mid_dim, state_dim, action_dim):
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
