from attrdict import AttrDict
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv


class Actor(nn.Module):
  def __init__(self, cfg):
    self.cfg, self.EP = filter_cfg(cfg)
    super().__init__()
    if cfg.net_type == 'deepset':
      self.net = nn.Sequential(
        ActorDeepsetBlock(cfg),
        *[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()] *
        (self.cfg.net_layer-self.cfg.shared_net_layer-1),
        # *[nn.Linear(cfg.net_dim, cfg.net_dim),nn.ReLU()]*(self.cfg.net_layer-4),
        nn.Linear(cfg.net_dim, self.EP.action_dim))
    elif cfg.net_type == 'attn':
      self.net = nn.Sequential(
        ActorAttnBlock(cfg),
        *[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()] *
        (self.cfg.net_layer-self.cfg.shared_net_layer-1),
        nn.Linear(cfg.net_dim, self.EP.action_dim))
    elif cfg.net_type == 'mlp':
      self.net = nn.Sequential(
        nn.Linear(self.EP.state_dim, cfg.net_dim), nn.ReLU(),
        # nn.Linear(cfg.net_dim, cfg.net_dim),nn.ReLU(),
        # nn.Linear(cfg.net_dim, cfg.net_dim),nn.ReLU(),
        *[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()] * \
        (self.cfg.net_layer-2),
        nn.Linear(cfg.net_dim, self.EP.action_dim),
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
    self.cfg, self.EP = filter_cfg(cfg)
    super().__init__()
    self.net_state = nn.Sequential(nn.Linear(self.EP.state_dim, cfg.net_dim), nn.ReLU(),
                                   nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU(), )
    self.net_a_avg = nn.Sequential(nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU(),
                                   nn.Linear(cfg.net_dim, self.EP.action_dim))  # the average of action
    self.net_a_std = nn.Sequential(nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU(),
                                   nn.Linear(cfg.net_dim, self.EP.action_dim))  # the log_std of action
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
    self.cfg, self.EP = filter_cfg(cfg)
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
        nn.Linear(self.EP.state_dim, cfg.net_dim), nn.ReLU(),
        *[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()] *
        (self.cfg.net_layer-2),
      )
    else:
      raise NotImplementedError(f'net_type {cfg.net_type} not implemented')
    if self.cfg.shared_actor:
      assert self.EP.num_robots == 2, 'shared actor only works for 2 robots'
      self.net_a_avg = nn.Linear(
        cfg.net_dim, self.EP.per_action_dim)  # the average of action
      self.net_a_std = nn.Linear(
        cfg.net_dim, self.EP.per_action_dim)  # the log_std of action
    else:
      self.net_a_avg = nn.Linear(
        cfg.net_dim, self.EP.action_dim)  # the average of action
      self.net_a_std = nn.Linear(
        cfg.net_dim, self.EP.action_dim)  # the log_std of action
    self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
    self.soft_plus = nn.Softplus()

  def forward(self, state):
    if self.cfg.shared_actor or self.cfg.mirror_actor:
      state = torch.stack((state, state@self.EP.obs_rot_mat),dim=1).view(-1,state.shape[-1]) # [batch * 2, state_dim]
    tmp = self.net_state(state)
    a_avg = self.net_a_avg(tmp).tanh()
    if self.cfg.shared_actor:
      a_avg = a_avg.view(-1,self.EP.action_dim)
      a_avg @= self.EP.last_act_rot_mat
    elif self.cfg.mirror_actor:
      a_avg = a_avg.view(-1,2*self.EP.action_dim)
      a_avg @= self.EP.dual_act_rot_mat
      a_avg = a_avg.view(-1,2,self.EP.action_dim).mean(dim=1)
    return a_avg

  def get_action(self, state):
    if self.cfg.shared_actor or self.cfg.mirror_actor:
      state = torch.stack((state, state@self.EP.obs_rot_mat),dim=1).view(-1,state.shape[-1]) # [batch * 2, state_dim]
    t_tmp = self.net_state(state)
    a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
    a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
    act = torch.normal(a_avg, a_std).tanh()  # re-parameterize
    if self.cfg.shared_actor:
      act = act.view(-1,self.EP.action_dim)
      act @= self.EP.last_act_rot_mat
    elif self.cfg.mirror_actor:
      act = act.view(-1,2*self.EP.action_dim)
      act @= self.EP.dual_act_rot_mat
      act = act.view(-1,2,self.EP.action_dim).mean(dim=1)
    return act

  def get_a_log_std(self, state):
    if self.cfg.shared_actor or self.cfg.mirror_actor:
      state = torch.stack((state, state@self.EP.obs_rot_mat),dim=1).view(-1,state.shape[-1]) # [batch * 2, state_dim]
    t_tmp = self.net_state(state)
    a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
    if self.cfg.shared_actor:
      a_std = a_std.view(-1,2,self.EP.per_action_dim)
      a_std = a_std.view(a_std.shape[0], -1)
    elif self.cfg.mirror_actor:
      a_std = a_std.view(-1,2,self.EP.action_dim).mean(dim=1)
    return a_std

  def get_logprob(self, state, action):
    if self.cfg.shared_actor or self.cfg.mirror_actor:
      state = torch.stack((state, state@self.EP.obs_rot_mat),dim=1).view(-1,state.shape[-1]) # [batch * 2, state_dim]
    t_tmp = self.net_state(state)
    a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
    a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
    a_std = a_std_log.exp()
    a_noise = a_avg + a_std * torch.randn_like(a_avg, requires_grad=True)
    noise = a_noise - action 
    log_prob = a_std_log + self.log_sqrt_2pi + \
      noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
    log_prob += (np.log(2.) - a_noise - self.soft_plus(-2. *a_noise)) * 2.
    if self.cfg.shared_actor:
      log_prob = log_prob.view(-1,2,self.EP.per_action_dim)
      log_prob = log_prob.view(log_prob.shape[0], -1)
    elif self.cfg.mirror_actor:
      log_prob = log_prob.view(-1,2,self.EP.action_dim).mean(dim=1)
    return log_prob

  def get_action_logprob(self, state):
    if self.cfg.shared_actor or self.cfg.mirror_actor:
      state = torch.stack((state, state@self.EP.obs_rot_mat),dim=1).view(-1,state.shape[-1]) # [batch * 2, state_dim]
    t_tmp = self.net_state(state)
    a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
    a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
    a_std = a_std_log.exp()
    noise = torch.randn_like(a_avg, requires_grad=True)
    a_noise = a_avg + a_std * noise
    log_prob = a_std_log + self.log_sqrt_2pi + \
      noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
    log_prob += (np.log(2.) - a_noise - self.soft_plus(-2. *
                                                       a_noise)) * 2.  # better than below
    a_noise = a_noise.tanh()
    if self.cfg.shared_actor:
      a_noise = a_noise.view(-1, self.EP.action_dim)
      a_noise @= self.EP.last_act_rot_mat
      log_prob = log_prob.view(-1,2,self.EP.per_action_dim)
      log_prob = log_prob.view(log_prob.shape[0], -1)
    elif self.cfg.mirror_actor:
      a_noise = a_noise.view(-1,2*self.EP.action_dim)
      a_noise @= self.EP.dual_act_rot_mat
      a_noise = a_noise.view(-1,2,self.EP.action_dim).mean(dim=1)
      log_prob = log_prob.view(-1,2,self.EP.action_dim).mean(dim=1)
    log_prob = log_prob.sum(1, keepdim=True)
    return a_noise, log_prob


class ActorPPO(nn.Module):
  def __init__(self, cfg):
    self.cfg, self.EP = filter_cfg(cfg)
    super().__init__()
    self.net = nn.Sequential(nn.Linear(self.EP.state_dim, cfg.net_dim), nn.ReLU(),
                             nn.Linear(
      cfg.net_dim, cfg.net_dim), nn.ReLU(),
      nn.Linear(cfg.net_dim, self.EP.action_dim))

    # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
    self.a_std_log = nn.Parameter(torch.zeros(
      (1, self.EP.action_dim)) - 0., requires_grad=True)
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
    self.cfg, self.EP = filter_cfg(cfg)
    super().__init__()
    self.net = nn.Sequential(nn.Linear(self.EP.state_dim + self.EP.action_dim, cfg.net_dim), nn.ReLU(),
                             nn.Linear(
      cfg.net_dim, cfg.net_dim), nn.ReLU(),
      nn.Linear(
      cfg.net_dim, cfg.net_dim), nn.ReLU(),
      nn.Linear(cfg.net_dim, 1))

  def forward(self, state, action):
    return self.net(torch.cat((state, action), dim=1))  # q value


class CriticTwin(nn.Module):  # shared parameter
  def __init__(self, cfg):
    self.cfg, self.EP = filter_cfg(cfg)
    super().__init__()
    if self.cfg.net_type == 'deepset':
      self.net_sa = CriticDeepsetBlock(cfg)
    elif self.cfg.net_type == 'attn':
      self.net_sa = CriticAttnBlock(cfg)
    elif self.cfg.net_type == 'mlp':
      self.net_sa = nn.Sequential(nn.Linear(self.EP.state_dim + self.EP.action_dim, cfg.net_dim), nn.ReLU(),
                                  *[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()]*(self.cfg.shared_net_layer-1))  # concat(state, action)
    else:
      raise NotImplementedError
    if self.cfg.mirror_feature_reg_coef > 0: # TODO make it more elegant
      self.net_q1_body = nn.Sequential(*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()]*(self.cfg.net_layer-1-self.cfg.shared_net_layer))
      self.net_q1_out = nn.Linear(cfg.net_dim, 1)
      self.net_q1 = nn.Sequential(self.net_q1_body, self.net_q1_out)
      self.net_q2_body = nn.Sequential(*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()]*(self.cfg.net_layer-1-self.cfg.shared_net_layer))
      self.net_q2_out = nn.Linear(cfg.net_dim, 1)
      self.net_q2 = nn.Sequential(self.net_q1_body, self.net_q1_out)
    else:
      self.net_q1 = nn.Sequential(*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()]*(self.cfg.net_layer-1-self.cfg.shared_net_layer),
                                  nn.Linear(cfg.net_dim, 1))  # q1 value
      self.net_q2 = nn.Sequential(*[nn.Linear(cfg.net_dim, cfg.net_dim), nn.ReLU()]*(self.cfg.net_layer-1-self.cfg.shared_net_layer),
                                  nn.Linear(cfg.net_dim, 1))  # q2 value
      

  def forward(self, state, action):
    return torch.mean(self.get_q_all(state, action))

  def get_q_min(self, state, action):
    # min Q value
    return torch.min(self.get_q_all(state, action), dim=-1, keepdim=True)[0]

  def get_q_all(self, state, action, get_mirror_std=False, get_embedding_norm=False):
    if self.cfg.shared_critic:
      state = torch.stack((state, state@self.EP.obs_rot_mat),dim=1).view(-1, state.shape[-1])
      action = torch.stack((action, action@self.EP.act_rot_mat),dim=1).view(-1, action.shape[-1])
      tmp = self.net_sa(torch.cat((state, action), dim=-1))
      if get_embedding_norm:
        embedding1 = self.net_q1_body(tmp)
        q1 = self.net_q1_out(embedding1)
        embedding2 = self.net_q2_body(tmp)
        q2 = self.net_q2_out(embedding1)
        embedding_stack = torch.cat((embedding1, embedding2), dim=-1).view(-1,self.cfg.net_dim,2) 
        q_stack = torch.cat((q1, q2), dim=-1).view(-1,2,2) # [batch, 2(mirror), 2]
        embedding_norm = torch.norm(embedding_stack, dim=1)/self.cfg.net_dim
        return q_stack.mean(dim=1), embedding_norm
      else:
        q_stack = torch.cat((self.net_q1(tmp), self.net_q2(tmp)), dim=-1).view(-1,2,2) # [batch, 2(mirror), 2]
        if get_mirror_std:
          return q_stack.mean(dim=1), q_stack.std(dim=1)
        else:
          return q_stack.mean(dim=1)
    else:
      tmp = self.net_sa(torch.cat((state, action), dim=1))
      return torch.cat((self.net_q1(tmp), self.net_q2(tmp)), dim=-1)


class CriticRed(nn.Module):  # shared parameter
  def __init__(self, cfg):
    self.cfg, self.EP = filter_cfg(cfg)
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
        nn.Linear(self.EP.state_dim + self.EP.action_dim, cfg.net_dim), nn.ReLU(),
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
    self.cfg, self.EP = filter_cfg(cfg)
    super().__init__()
    self.net = nn.Sequential(nn.Linear(self.EP.state_dim, cfg.net_dim), nn.ReLU(),
                             nn.Linear(
      cfg.net_dim, cfg.net_dim), nn.ReLU(),
      nn.Linear(cfg.net_dim, 1))

  def forward(self, state):
    return self.net(state)  # advantage value


# TODO make it sequential
class ActorDeepsetBlock(nn.Module):
  def __init__(self, cfg):
    # state_dim=[shared_dim, seperate_dim, goal_dim, num_goals]
    self.cfg, self.EP = filter_cfg(cfg)
    super().__init__()
    self.shared_dim = self.EP.shared_dim
    self.seperate_dim = self.EP.seperate_dim
    self.goal_dim = self.EP.goal_dim
    self.num_goals = self.EP.num_goals
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
    self.cfg, self.EP = filter_cfg(cfg)
    super().__init__()
    self.shared_dim = self.EP.shared_dim
    self.seperate_dim = self.EP.seperate_dim
    self.goal_dim = self.EP.goal_dim
    self.num_goals = self.EP.num_goals
    assert self.goal_dim % self.num_goals == 0, f'goal dim {self.goal_dim} should be divisible by num goals {self.num_goals}'
    self.single_goal_dim = self.goal_dim // self.num_goals
    assert self.seperate_dim % self.num_goals == 0, f'seperate dim {self.seperate_dim} should be divisible by num goals {self.num_goals}'
    self.single_seperate_dim = self.seperate_dim // self.num_goals
    if self.cfg.actor_pool_type in ['cross', 'cross2']:
      self.query_embed = nn.Sequential(
        nn.Linear(self.shared_dim, cfg.net_dim), nn.ReLU())
      self.cross_attn = nn.MultiheadAttention(
        self.cfg.net_dim, self.cfg.n_head, dropout=0.0)
    self.embed = nn.Sequential(
      nn.Linear(self.shared_dim + self.single_seperate_dim +
                self.single_goal_dim, cfg.net_dim), nn.ReLU())
    self.enc = nn.Sequential(*[AttnEncoderLayer(self.cfg.net_dim, n_head=self.cfg.n_head, dim_ff=self.cfg.net_dim,
                                                pre_lnorm=True, dropout=0.0) for _ in range(self.cfg.shared_net_layer-1)])
    if self.cfg.actor_pool_type in ['bert', 'bert2']:
      self.bert_query = nn.parameter.Parameter(torch.randn(self.cfg.net_dim))
      self.bert_attn = nn.MultiheadAttention(
        self.cfg.net_dim, self.cfg.n_head, dropout=0.0)

  def forward(self, state):
    grip = state[..., :self.shared_dim]
    obj = state[..., self.shared_dim:self.shared_dim +
                self.seperate_dim].reshape(-1, self.num_goals, self.single_seperate_dim)
    g = state[..., self.shared_dim+self.seperate_dim:self.shared_dim +
              self.seperate_dim+self.goal_dim].reshape(-1, self.num_goals, self.single_goal_dim)
    if self.cfg.actor_pool_type in ['cross', 'cross2']:
      query = self.query_embed(grip).unsqueeze(0)
    grip = grip.unsqueeze(1).repeat(1, self.num_goals, 1)
    x = torch.cat((grip, obj, g), -1)
    x = self.embed(x).transpose(0, 1)  # Tensor(num_goals, num_envs, net_dim)
    if self.cfg.actor_pool_type == 'bert2':
      x = torch.cat((self.bert_query.repeat(1,x.shape[1],1), x), 0) # Tensor(num_goals+1, num_envs, net_dim)
    if self.cfg.actor_pool_type == 'cross2':
      x = torch.cat((query, x), 0) # Tensor(num_goals+1, num_envs, net_dim)
    token = self.enc(x)
    if self.cfg.actor_pool_type == 'mean':
      return token.mean(dim=0)
    elif self.cfg.actor_pool_type == 'max':
      return token.max(dim=0)[0]
    elif self.cfg.actor_pool_type == 'bert':
      return self.bert_attn(self.bert_query.tile(1, token.shape[1], 1), token, token)[0].squeeze(0)
    elif self.cfg.actor_pool_type == 'bert2':
      return self.bert_attn(token[[0]], token[1:], token[1:])[0].squeeze(0)
    elif self.cfg.actor_pool_type == 'cross':
      return self.cross_attn(query, token, token)[0].squeeze(0)
    elif self.cfg.actor_pool_type == 'cross2':
      return self.cross_attn(token[[0]], token[1:], token[1:])[0].squeeze(0)
    else:
      raise NotImplementedError


class CriticDeepsetBlock(nn.Module):
  def __init__(self, cfg):
    self.cfg, self.EP = filter_cfg(cfg)
    super().__init__()
    self.shared_dim = self.EP.shared_dim
    self.seperate_dim = self.EP.seperate_dim
    self.goal_dim = self.EP.goal_dim
    self.num_goals = self.EP.num_goals
    self.action_dim = self.EP.action_dim
    assert self.goal_dim % self.num_goals == 0, f'goal dim {self.goal_dim} should be divisible by num goals {self.num_goals}'
    self.single_goal_dim = self.goal_dim // self.num_goals
    assert self.seperate_dim % self.num_goals == 0, f'seperate dim {self.seperate_dim} should be divisible by num goals {self.num_goals}'
    self.single_seperate_dim = self.seperate_dim // self.num_goals
    self.net_in = nn.Sequential(
      nn.Linear(self.shared_dim+self.single_seperate_dim +
                self.single_goal_dim+self.EP.action_dim, cfg.net_dim), nn.ReLU(),
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
    self.cfg, self.EP = filter_cfg(cfg)
    super().__init__()
    self.shared_dim = self.EP.shared_dim
    self.seperate_dim = self.EP.seperate_dim
    self.goal_dim = self.EP.goal_dim
    self.num_goals = self.EP.num_goals
    self.action_dim = self.EP.action_dim
    assert self.goal_dim % self.num_goals == 0, f'goal dim {self.goal_dim} should be divisible by num goals {self.num_goals}'
    self.single_goal_dim = self.goal_dim // self.num_goals
    assert self.seperate_dim % self.num_goals == 0, f'seperate dim {self.seperate_dim} should be divisible by num goals {self.num_goals}'
    self.single_seperate_dim = self.seperate_dim // self.num_goals
    if self.cfg.actor_pool_type == 'cross':
      self.query_embed = nn.Sequential(
        nn.Linear(self.shared_dim+self.EP.action_dim, cfg.net_dim), nn.ReLU())
      self.cross_attn = nn.MultiheadAttention(
        self.cfg.net_dim, self.cfg.n_head, dropout=0.0)
    self.embed = nn.Sequential(
      nn.Linear(self.shared_dim + self.single_seperate_dim +
                self.single_goal_dim+self.EP.action_dim, cfg.net_dim), nn.ReLU())
    self.enc = nn.Sequential(*[AttnEncoderLayer(self.cfg.net_dim, n_head=self.cfg.n_head, dim_ff=self.cfg.net_dim,
                                                pre_lnorm=True, dropout=0.0) for _ in range(self.cfg.shared_net_layer-1)])
    if self.cfg.critic_pool_type == 'bert':
      self.bert_query = nn.parameter.Parameter(torch.randn(self.cfg.net_dim))
      self.bert_attn = nn.MultiheadAttention(
        self.cfg.net_dim, self.cfg.n_head, dropout=0.0)

  def forward(self, state, action=None):
    if action is None:
      action = state[..., -self.action_dim:]
    batch_shape = state.shape[:-1] # [batch] or [batch, stack]
    batch_shape_len = len(batch_shape)
    obj = state[..., self.shared_dim:self.shared_dim +
                self.seperate_dim].reshape(*batch_shape, self.num_goals, self.single_seperate_dim)
    g = state[..., self.shared_dim+self.seperate_dim:self.shared_dim +
              self.seperate_dim+self.goal_dim].reshape(*batch_shape, self.num_goals, self.single_goal_dim)
    if self.cfg.actor_pool_type == 'cross':
      grip = state[..., :self.shared_dim]
      query = self.query_embed(torch.cat([grip, action], dim=-1)).unsqueeze(0)
    grip = state[..., :self.shared_dim].unsqueeze(
      -2).repeat(*([1]*batch_shape_len), self.num_goals, 1)
    action = action.unsqueeze(-2).repeat(*([1]*batch_shape_len), self.num_goals, 1)
    x = torch.cat((grip, obj, g, action), -1)  # batch, obj, feature
    x = self.embed(x).transpose(0, 1)
    token = self.enc(x)
    if self.cfg.critic_pool_type == 'mean':
      return token.mean(dim=0)
    elif self.cfg.critic_pool_type == 'max':
      return token.max(dim=0)[0]
    elif self.cfg.critic_pool_type == 'bert':
      return self.bert_attn(self.bert_query.tile(1, token.shape[1], 1), token, token)[0].squeeze(0)
    elif self.cfg.critic_pool_type == 'cross':
      return self.cross_attn(query, token, token)[0].squeeze(0)
    else:
      raise NotImplementedError


class AttnEncoderLayer(nn.Module):
  """Adapted from: https://github.com/jwang0306/transformer-pytorch."""

  def __init__(self, hidden_size, n_head, dim_ff, pre_lnorm, dropout=0.0):
    super().__init__()
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


class GATEncoderLayer(nn.Module):
  """Graph Attention Network"""

  def __init__(self, hidden_size, n_head):
    super().__init__()
    self.gat1 = GATv2Conv(hidden_size, hidden_size, heads=n_head)
    self.gat2 = GATv2Conv(hidden_size*n_head, hidden_size, heads=n_head)

  def forward(self, x, edge_index):
    h = self.gat1(x, edge_index)
    h = F.elu(h)
    h = self.gat2(h, edge_index)
    return h, F.log_softmax(h, dim=1)

class CriticDeepset(nn.Module):
  def __init__(self, cfg):
    self.cfg, self.EP = filter_cfg(cfg)
    super().__init__()
    self.shared_dim = self.EP.shared_dim
    self.seperate_dim = self.EP.seperate_dim
    self.goal_dim = self.EP.goal_dim
    self.num_goals = self.EP.num_goals
    self.action_dim = self.EP.action_dim
    assert self.goal_dim % self.num_goals == 0, f'goal dim {self.goal_dim} should be divisible by num goals {self.num_goals}'
    self.single_goal_dim = self.goal_dim // self.num_goals
    assert self.seperate_dim % self.num_goals == 0, f'seperate dim {self.seperate_dim} should be divisible by num goals {self.num_goals}'
    self.single_seperate_dim = self.seperate_dim // self.num_goals
    self.net_in = nn.Sequential(
      nn.Linear(self.shared_dim+self.single_seperate_dim +
                self.single_goal_dim+self.EP.action_dim, cfg.net_dim), nn.ReLU(),
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
