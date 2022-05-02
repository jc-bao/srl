import torch
import numpy as np
from copy import deepcopy
import os
import logging
from attrdict import AttrDict
import gym
import wandb

import net
from replay_buffer import ReplayBuffer, ReplayBufferList


class AgentBase:
  def __init__(self, cfg=None):
    '''set up params'''
    self.cfg = cfg
    # device
    self.cfg.update(device=torch.device(f"cuda:{cfg.gpu_id}" if (
      torch.cuda.is_available() and (cfg.gpu_id >= 0)) else "cpu"))
    # eval steps
    if self.cfg.eval_steps is None:
      self.cfg.update(eval_steps=cfg.steps_per_rollout)
    # dir
    if cfg.wandb:
      self.cfg.update(
        cwd=f'{wandb.run.dir}/{cfg.name}_{cfg.project}_{cfg.env_name[4:]}')
    else:
      self.cfg.update(
        cwd=f'./{cfg.cwd}/{cfg.name}_{cfg.project}_{cfg.env_name[4:]}')
    os.makedirs(self.cfg.cwd, exist_ok=True)
    # update times
    if self.cfg.updates_per_rollout is None:
      self.cfg.update(updates_per_rollout=cfg.resue *
                      cfg.steps_per_rollout//cfg.batch_size)

    '''seed '''
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.set_default_dtype(torch.float32)

    '''env setup'''
    print('[Agent] env setup')
    self.env = gym.make(cfg.env_name, **cfg.env_kwargs)
    self.cfg.update(env_params=self.env.env_params())
    # alias for env_params
    self.EP = self.cfg.env_params
    # batch size
    if self.cfg.batch_size < self.EP.num_envs:
      print('[Agent] WARNING: batch_size < num_envs')

    '''set up actor critic'''
    print('[Agent] net setup')
    act_class = getattr(net, cfg.act_net, None)
    cri_class = getattr(net, cfg.critic_net, None)
    self.act = act_class(cfg).to(self.cfg.device)
    self.cri = cri_class(cfg).to(
      self.cfg.device) if cri_class else self.act
    self.act_target = deepcopy(
      self.act) if self.cfg.if_act_target else self.act
    self.cri_target = deepcopy(
      self.cri) if self.cfg.if_cri_target else self.cri

    self.act_optimizer = torch.optim.Adam(
      self.act.parameters(), self.cfg.lr)
    self.cri_optimizer = torch.optim.Adam(
      self.cri.parameters(), self.cfg.lr) if cri_class else self.act_optimizer

    '''function'''
    print('[Agent] data setup')
    self.criterion = torch.nn.SmoothL1Loss()
    # PER (Prioritized Experience Replay) for sparse reward
    if getattr(cfg, 'if_use_per', False):
      self.criterion = torch.nn.SmoothL1Loss(reduction='none')
      self.get_obj_critic = self.get_obj_critic_per
    else:
      self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
      self.get_obj_critic = self.get_obj_critic_raw

    '''record params'''
    self.total_step = 0
    self.r_max = -np.inf
    # params for tmp buffer to record traj info
    self.traj_idx = torch.arange(
      self.EP.num_envs, device=self.cfg.device)
    self.num_traj = self.EP.num_envs

    ''' data '''
    # tmp buffer
    self.buffer = ReplayBufferList(
      cfg) if 'PPO' in cfg.agent_name else ReplayBuffer(cfg)
    self.traj_list = torch.empty((self.EP.num_envs, self.EP.max_env_step,
                                 self.buffer.total_dim), device=self.cfg.device, dtype=torch.float32)

  def eval_vec_env(self, target_steps=None):
    # auto set target steps
    if target_steps is None:
      target_steps = self.cfg.eval_steps
    else:
      logging.warn(
        f'eval_env: target_steps is not None, forced to {target_steps}')
    # log data
    num_ep = torch.zeros(self.EP.num_envs,
                         device=self.cfg.device)
    ep_rew = torch.zeros(self.EP.num_envs,
                         device=self.cfg.device)
    ep_step = torch.zeros(self.EP.num_envs,
                          device=self.cfg.device)
    final_rew = torch.zeros(
      self.EP.num_envs, device=self.cfg.device)
    # reset
    ten_s = self.env.reset()
    if self.cfg.render:
      images = self.env.render(mode='rgb_array')
      videos = [[im] for im in images]
    # loop
    collected_steps = 0
    while collected_steps < target_steps:
      ten_a = self.act(ten_s).detach()
      ten_s_next, ten_rewards, ten_dones, _ = self.env.step(
        ten_a)  # different
      if self.cfg.render:
        images = self.env.render(mode='rgb_array')
        for vi, im in zip(videos, images):
          vi.append(im)
      ten_dones = ten_dones.type(torch.bool)
      num_ep[ten_dones] += 1
      ep_rew += ten_rewards
      ep_step += 1
      final_rew[ten_dones] += ten_rewards[ten_dones]
      collected_steps = (ep_step).sum()
      ten_s = ten_s_next
    # return
    video = None
    if self.cfg.render:
      videos = np.array(videos)
      video = np.concatenate(videos, axis=0)
      video = np.moveaxis(video, -1, 1)
    return AttrDict(
      steps=self.total_step,
      ep_rew=torch.mean(ep_rew/num_ep).item(),
      final_rew=torch.mean(final_rew/num_ep).item(),
      ep_steps=torch.mean(ep_step/num_ep).item(),
      video=video
    )

  def explore_vec_env(self, target_steps=None):
    # auto set target steps
    if target_steps is None:
      target_steps = self.cfg.steps_per_rollout
    else:
      logging.warn(
        f'explore: target_steps is not None, forced to {target_steps}')
    # reset tmp buffer status
    traj_start_ptr = torch.zeros(
      self.EP.num_envs, dtype=torch.long, device=self.cfg.device)
    traj_lens = torch.zeros(self.EP.num_envs,
                            dtype=torch.long, device=self.cfg.device)
    data_ptr = 0  # where to store data
    collected_steps = 0  # data added to buffer
    useless_steps = 0  # data explored but dropped
    # loop
    ten_s = self.env.reset()
    while collected_steps < target_steps:
      ten_a = self.act.get_action(ten_s).detach()  # different
      if isinstance(ten_a, tuple):
        ten_a, ten_n = ten_a  # record noise for no-policy
      ten_s_next, ten_rewards, ten_dones, ten_info = self.env.step(
        ten_a)  # different
      # preprocess info, add [1]trajectory index, [2]traj len, [3]to left
      ten_info = self.EP.info_updater(
        ten_info, AttrDict(traj_idx=self.traj_idx))
      # add data to tmp buffer
      self.traj_list[:, data_ptr, :] = torch.cat((
        ten_s,  # state
        ten_rewards.unsqueeze(1)*self.cfg.reward_scale,  # reward
        ((1-ten_dones)*self.cfg.gamma).unsqueeze(1),  # mask
        ten_a,  # action
        ten_info,  # info
      ), dim=-1)
      # update ptr for tmp buffer
      data_ptr = (data_ptr+1) % self.EP.max_env_step
      # update trajectory info
      traj_lens += 1
      done_idx = torch.where(ten_dones)[0]
      done_num_envs = torch.sum(ten_dones).type(torch.int32)
      # update traj index
      self.traj_idx[done_idx] = (
        self.num_traj+torch.arange(done_num_envs, device=self.cfg.device))
      self.num_traj += done_num_envs
      assert torch.max(self.traj_idx) + 1 == self.num_traj, \
        f'traj index {self.traj_idx} and num traj {self.num_traj} not match'
      # reset traj recorder and add extra traj info
      if done_idx.shape[0] > 0:
        # tile traj len for later use
        tiled_traj_len = traj_lens[done_idx].unsqueeze(1)\
          .tile(1, self.EP.max_env_step).float()
        # get data
        data = self.traj_list[done_idx]
        # calculate to left distance
        info_step = self.buffer.data_parser(data, 'info.step')
        # NOTE: not inplace op here
        self.traj_list[done_idx] = self.buffer.data_updater(
          data,
          AttrDict(
            info=AttrDict(
              tleft=tiled_traj_len - info_step,
              traj_len=tiled_traj_len))
        )
        # add to big buffer
        results = self.save_to_buffer(done_idx, traj_start_ptr, traj_lens)
        self.total_step += (results.collected_steps + results.useless_steps)
        collected_steps += results.collected_steps
        useless_steps += results.useless_steps
        # reset record params
        traj_start_ptr[done_idx] = (data_ptr+1) % self.EP.max_env_step
        traj_lens[done_idx] = 0
      # setup next state
      ten_s = ten_s_next

    return AttrDict(
      steps=self.total_step,
      collected_steps=collected_steps,
      useless_steps=useless_steps,
    )

  def save_to_buffer(self, done_idx, traj_start_ptr, traj_lens):
    traj_data = []
    useless_steps = 0
    for i in done_idx:  # TODO fix the one by one add traj process
      start_point = traj_start_ptr[i]
      end_point = (start_point + traj_lens[i]) % self.EP.max_env_step
      end_data = self.traj_list[i, (end_point-1) % self.EP.max_env_step]
      end_info = self.buffer.data_parser(end_data, 'info')
      end_info_dict = self.EP.info_parser(end_info)
      # TODO merge buffer and add parser
      # dropout unmoved experience
      if getattr(end_info_dict, 'early_termin', False):
        useless_steps += traj_lens[i]
        continue
      if start_point < end_point:
        traj_data.append(
          self.traj_list[i, start_point:end_point])
      else:
        traj_data.append(torch.cat((
          self.traj_list[i, start_point:],
          self.traj_list[i, :end_point]
        ), dim=0))
    if traj_data:
      traj_data = torch.cat(traj_data, dim=0)
      self.buffer.extend_buffer(traj_data)
    return AttrDict(
      collected_steps=len(traj_data),
      useless_steps=int(useless_steps)
    )

  def convert_trajectory(self, buf_items, last_done):
    buf_items = list(map(list, zip(*buf_items)))
    '''stack items'''
    buf_items[0] = torch.stack(buf_items[0])
    # action, info
    buf_items[3:] = [torch.stack(item) for item in buf_items[3:]]
    # action
    if len(buf_items[3].shape) == 2:
      buf_items[3] = buf_items[3].unsqueeze(2)
    # info
    if self.EP.num_envs > 1:
      # rew
      buf_items[1] = (torch.stack(buf_items[1]) *
                      self.cfg.reward_scale).unsqueeze(2)
      # mask
      buf_items[2] = ((1 - torch.stack(buf_items[2]))
                      * self.cfg.gamma).unsqueeze(2)
    else:
      buf_items[1] = (torch.tensor(buf_items[1], dtype=torch.float32) * self.reward_scale
                      ).unsqueeze(1).unsqueeze(2)
      buf_items[2] = ((1 - torch.tensor(buf_items[2], dtype=torch.float32)) * self.cfg.gamma
                      ).unsqueeze(1).unsqueeze(2)
    '''splice items'''
    # for j in range(len(buf_items)):
    #   cur_item = list()
    #   buf_item = buf_items[j]
    #   for env_i in range(self.EP.num_envs):
    #     last_step = last_done[env_i]
    #     cur_item.append(buf_item[:last_step, env_i])
    #   buf_items[j] = torch.vstack(cur_item)
    return buf_items

  def get_obj_critic_raw(self, buffer, batch_size):
    with torch.no_grad():
      trans = buffer.sample_batch(batch_size, her_rate=self.cfg.her_rate)
      next_a = self.act_target(trans.next_state)  # stochastic policy
      critic_targets: torch.Tensor = self.cri_target(trans.next_state, next_a)
      (next_q, min_indices) = torch.min(critic_targets, dim=1, keepdim=True)
      q_label = trans.rew.unsqueeze(-1)+ trans.mask.unsqueeze(-1) * next_q 
    q = self.cri(trans.state, trans.action)
    obj_critic = self.criterion(q, q_label)
    return obj_critic, trans.state 

  def get_obj_critic_per(self, buffer, batch_size):
    """
    Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

    :param buffer: the ReplayBuffer instance that stores the trajectories.
    :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
    :return: the loss of the network and states.
    """
    with torch.no_grad():
      reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
        batch_size)
      next_a = self.act_target(next_s)
      critic_targets: torch.Tensor = self.cri_target(next_s, next_a)
      # taking a minimum while preserving the dimension for possible twin critics
      (next_q, min_indices) = torch.min(critic_targets, dim=1, keepdim=True)
      q_label = reward + mask * next_q

    q = self.cri(state, action)
    td_error = self.criterion(q, q_label)
    obj_critic = (td_error * is_weights).mean()

    buffer.td_error_update(td_error.detach())
    return obj_critic, state

  @staticmethod
  def optimizer_update(optimizer, objective):
    optimizer.zero_grad()
    objective.backward()
    optimizer.step()

  @staticmethod
  def soft_update(target_net, current_net, tau):
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
      tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

  def save_or_load_agent(self, file_tag='', cwd=None, if_save=True):
    if cwd is None:
      cwd = self.cfg.cwd

    def load_torch_file(model_or_optim, _path):
      state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
      model_or_optim.load_state_dict(state_dict)

    name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optimizer),
                     ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optimizer), ]
    name_obj_list = [(name, obj)
                     for name, obj in name_obj_list if obj is not None]
    if if_save:
      for name, obj in name_obj_list:
        save_path = f"{cwd}/{file_tag+name}.pth"
        torch.save(obj.state_dict(), save_path)
    else:
      for name, obj in name_obj_list:
        save_path = f"{cwd}/{file_tag+name}.pth"
        load_torch_file(obj, save_path) if os.path.isfile(save_path) else None


class AgentSAC(AgentBase):
  def __init__(self, cfg):
    super().__init__(cfg=cfg)

    self.alpha_log = torch.tensor((-np.log(self.EP.action_dim) * np.e,), dtype=torch.float32,
                                  requires_grad=True, device=cfg.device)  # trainable parameter
    self.alpha_optim = torch.optim.Adam(
      (self.alpha_log,), lr=cfg.lr)
    self.target_entropy = np.log(self.EP.action_dim)

  def update_net(self):
    self.buffer.update_now_len()

    obj_critic = obj_actor = None
    for _ in range(1+int(self.buffer.now_len / self.buffer.max_len * self.cfg.updates_per_rollout)):
      '''objective of critic (loss function of critic)'''
      obj_critic, state = self.get_obj_critic(self.buffer, self.cfg.batch_size)
      self.optimizer_update(self.cri_optimizer, obj_critic)
      self.soft_update(self.cri_target, self.cri, self.cfg.soft_update_tau)

      '''objective of alpha (temperature parameter automatic adjustment)'''
      a_noise_pg, log_prob = self.act.get_action_logprob(
        state)  # policy gradient
      obj_alpha = (self.alpha_log * (log_prob -
                   self.target_entropy).detach()).mean()
      self.optimizer_update(self.alpha_optim, obj_alpha)

      '''objective of actor'''
      alpha = self.alpha_log.exp().detach()
      with torch.no_grad():
        self.alpha_log[:] = self.alpha_log.clamp(-20, 2)

      q_value_pg = self.cri(state, a_noise_pg)
      obj_actor = -(q_value_pg + log_prob * alpha).mean()
      self.optimizer_update(self.act_optimizer, obj_actor)
      # self.soft_update(self.act_target, self.act, self.cfg.soft_update_tau) # SAC don't use act_target network

    return AttrDict(
      critic_loss=obj_critic.item(),
      actor_loss=-obj_actor.item(),
      alpha_log=self.alpha_log.exp().detach().item()
    )

  def get_obj_critic_raw(self, buffer, batch_size):
    with torch.no_grad():
      trans = buffer.sample_batch(batch_size, her_rate=self.cfg.her_rate)
      next_a, next_log_prob = self.act_target.get_action_logprob(
        trans.next_state)  # stochastic policy
      next_q = self.cri_target.get_q_min(trans.next_state, next_a)

      alpha = self.alpha_log.exp().detach()
      q_label = trans.rew.unsqueeze(-1) + trans.mask.unsqueeze(-1) * \
        (next_q + next_log_prob * alpha)
    q1, q2 = self.cri.get_q1_q2(trans.state, trans.action)
    obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2
    return obj_critic, trans.state

  def get_obj_critic_per(self, buffer, batch_size):
    with torch.no_grad():
      reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
        batch_size)

      next_a, next_log_prob = self.act_target.get_action_logprob(
        next_s)  # stochastic policy
      next_q = self.cri_target.get_q_min(next_s, next_a)

      alpha = self.alpha_log.exp().detach()
      q_label = reward + mask * (next_q + next_log_prob * alpha)
    q1, q2 = self.cri.get_q1_q2(state, action)
    td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
    obj_critic = (td_error * is_weights).mean()

    buffer.td_error_update(td_error.detach())
    return obj_critic, state


# Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
class AgentModSAC(AgentSAC):
  def __init__(self, cfg):
    super().__init__(cfg)
    self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    self.lambda_a_log_std = getattr(cfg, 'lambda_a_log_std', 2 ** -4)

  def update_net(self, buffer):
    buffer.update_now_len()

    with torch.no_grad():  # H term
        # buf_state = buffer.sample_batch_r_m_a_s()[3]
      if buffer.prev_idx <= buffer.next_idx:
        buf_state = buffer.buf_state[buffer.prev_idx:buffer.next_idx]
      else:
        buf_state = torch.vstack((buffer.buf_state[buffer.prev_idx:],
                                  buffer.buf_state[:buffer.next_idx],))
      buffer.prev_idx = buffer.next_idx

      avg_a_log_std = self.act.get_a_log_std(
        buf_state).mean(dim=0, keepdim=True)
      avg_a_log_std = avg_a_log_std * \
        torch.ones((self.cfg.batch_size, 1), device=self.cfg.device)
      del buf_state

    alpha = self.alpha_log.exp().detach()
    update_a = 0
    obj_actor = torch.zeros(1)
    for update_c in range(1, int(2 + buffer.now_len * self.cfg.repeat_times / self.cfg.batch_size)):
      '''objective of critic (loss function of critic)'''
      obj_critic, state = self.get_obj_critic(buffer, self.cfg.batch_size)
      self.optimizer_update(self.cri_optimizer, obj_critic)
      self.soft_update(self.cri_target, self.cri, self.cfg.soft_update_tau)
      self.obj_c = 0.995 * self.obj_c + 0.005 * \
        obj_critic.item()  # for reliable_lambda

      a_noise_pg, logprob = self.act.get_action_logprob(
        state)  # policy gradient
      '''objective of alpha (temperature parameter automatic adjustment)'''
      obj_alpha = (self.alpha_log * (logprob -
                   self.target_entropy).detach()).mean()
      self.optimizer_update(self.alpha_optim, obj_alpha)
      with torch.no_grad():
        self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
      alpha = self.alpha_log.exp().detach()

      '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
      reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
      if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
      if if_update_a:  # auto TTUR
        update_a += 1

        obj_a_std = self.criterion(self.act.get_a_log_std(
          state), avg_a_log_std) * self.lambda_a_log_std

        q_value_pg = self.cri(state, a_noise_pg)
        obj_actor = -(q_value_pg + logprob * alpha).mean() + obj_a_std

        self.optimizer_update(self.act_optimizer, obj_actor)
        self.soft_update(self.act_target, self.act,
                         self.cfg.soft_update_tau)
    return self.obj_c, -obj_actor.item(), alpha.item()


# Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
class AgentREDqSAC(AgentSAC):
  def __init__(self, cfg):
    super().__init__(cfg)
    self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

  def get_obj_critic_raw(self, buffer, batch_size):
    with torch.no_grad():
      trans = buffer.sample_batch(batch_size, her_rate=self.cfg.her_rate)
      next_a, next_log_prob = self.act_target.get_action_logprob(
        trans.next_state)  # stochastic policy
      next_q = self.cri_target.get_q_min(trans.next_state, next_a)

      alpha = self.alpha_log.exp().detach()
      q_label = trans.rew.unsqueeze(-1) + trans.mask.unsqueeze(-1) * \
        (next_q + next_log_prob * alpha)
    qs = self.cri.get_q_values(trans.state, trans.action)
    obj_critic = self.criterion(qs, q_label * torch.ones_like(qs))
    return obj_critic, trans.state

  def get_obj_critic_per(self, buffer, batch_size):
    with torch.no_grad():
      reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
        batch_size)

      next_a, next_log_prob = self.act_target.get_action_logprob(
        next_s)  # stochastic policy
      next_q = self.cri_target.get_q_min(next_s, next_a)

      alpha = self.alpha_log.exp().detach()
      q_label = reward + mask * (next_q + next_log_prob * alpha)
    qs = self.cri.get_q_values(state, action)
    td_error = self.criterion(qs, q_label * torch.ones_like(qs)).mean(dim=1)
    obj_critic = (td_error * is_weights).mean()

    buffer.td_error_update(td_error.detach())
    return obj_critic, state

# Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)


class AgentREDQSAC(AgentSAC):
  def __init__(self, cfg):
    self.act_class = getattr(self, 'act_class', ActorFixSAC)
    self.cri_class = getattr(self, 'cri_class', CriticREDQ)
    self.repeat_q_times = 1
    super().__init__(cfg)
    self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

  def get_obj_critic_raw(self, buffer, batch_size):
    with torch.no_grad():
      reward, mask, action, state, next_s, info = buffer.sample_batch(
        batch_size)

      next_a, next_log_prob = self.act_target.get_action_logprob(
        next_s)  # stochastic policy
      next_q = self.cri_target.get_q_min(next_s, next_a)

      alpha = self.alpha_log.exp().detach()
      q_label = reward + mask * (next_q + next_log_prob * alpha)
    qs = self.cri.get_q_values(state, action)
    obj_critic = self.criterion(qs, q_label * torch.ones_like(qs))
    return obj_critic, state

  def update_net(self, buffer):
    buffer.update_now_len()

    obj_critic = obj_actor = None
    for _ in range(int(1 + buffer.now_len * self.cfg.repeat_times / self.cfg.batch_size)):
      for _ in range(self.repeat_q_times):
        '''objective of critic (loss function of critic)'''
        obj_critic, state = self.get_obj_critic(buffer, self.cfg.batch_size)
        self.optimizer_update(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri,
                         self.cfg.soft_update_tau)

      '''objective of alpha (temperature parameter automatic adjustment)'''
      a_noise_pg, log_prob = self.act.get_action_logprob(
        state)  # policy gradient
      obj_alpha = (self.alpha_log * (log_prob -
                   self.target_entropy).detach()).mean()
      self.optimizer_update(self.alpha_optim, obj_alpha)

      '''objective of actor'''
      alpha = self.alpha_log.exp().detach()
      with torch.no_grad():
        self.alpha_log[:] = self.alpha_log.clamp(-20, 2)

      q_value_pg = self.cri(state, a_noise_pg)
      obj_actor = -(q_value_pg + log_prob * alpha).mean()
      self.optimizer_update(self.act_optimizer, obj_actor)

    return obj_critic.item(), -obj_actor.item(), self.alpha_log.exp().detach().item()


class AgentDDPG(AgentBase):
  def __init__(self, cfg):
    super().__init__(cfg=cfg)
    self.act.explore_noise = getattr(
      cfg, 'explore_noise', 0.2)  # set for `get_action()`

  def update_net(self):
    self.buffer.update_now_len()
    obj_critic = obj_actor = None
    for _ in range(1+int(self.buffer.now_len / self.buffer.max_len * self.cfg.updates_per_rollout)):
      obj_critic, state = self.get_obj_critic(self.buffer, self.cfg.batch_size)
      self.optimizer_update(self.cri_optimizer, obj_critic)
      self.soft_update(self.cri_target, self.cri, self.cfg.soft_update_tau)

      action_pg = self.act(state)  # policy gradient
      obj_actor = -self.cri(state, action_pg).mean()
      self.optimizer_update(self.act_optimizer, obj_actor)
      self.soft_update(self.act_target, self.act, self.cfg.soft_update_tau)
    return AttrDict(
      critic_loss=obj_critic.item(),
      actor_loss=-obj_actor.item(),
    )


class AgentPPO(AgentBase):
  def __init__(self, cfg):
    super().__init__(cfg=cfg)
    if cfg.if_use_gae:
      self.get_reward_sum = self.get_reward_sum_gae
    else:
      self.get_reward_sum = self.get_reward_sum_raw

  def explore_vec_env(self, target_steps=None) -> list:
    # auto set target steps
    if target_steps is None:
      target_steps = self.cfg.steps_per_rollout
    else:
      logging.warn(
        f'explore: target_steps is not None, forced to {target_steps}')
    # TODO merge into base class explore fn
    traj_list = list()
    last_done = torch.zeros(
      self.EP.num_envs, dtype=torch.int, device=self.cfg.device)
    ten_s = self.env.reset()
    step_i = 0
    ten_dones = torch.zeros(
      self.EP.num_envs, dtype=torch.int, device=self.cfg.device)
    get_action = self.act.get_action
    get_a_to_e = self.act.get_a_to_e
    while step_i < target_steps:
      ten_a, ten_n = get_action(ten_s)  # different
      ten_s_next, ten_rewards, ten_dones, _ = self.env.step(get_a_to_e(ten_a))
      traj_list.append((ten_s.clone(), ten_rewards.clone(),
                       ten_dones.clone(), ten_a, ten_n))  # different
      step_i += self.EP.num_envs
      last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
      ten_s = ten_s_next
    self.total_step += step_i
    buf_items = self.convert_trajectory(traj_list, last_done)
    steps, mean_rew = self.buffer.update_buffer(buf_items)  # traj_list
    return AttrDict(
      steps=self.total_step,
      mean_rew=mean_rew.item(),
    )

  def update_net(self):
    with torch.no_grad():
      buf_state, buf_reward, buf_mask, buf_action, buf_noise = [
        ten.to(self.cfg.device) for ten in self.buffer]
        # ten.to(self.cfg.device).view(-1,ten.shape[-1]) for ten in self.buffer]
      buf_len = buf_state.shape[0]

      '''get buf_r_sum, buf_logprob'''
      # bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
      # buf_value = [self.cri_target(buf_state[i:i + bs])
      #              for i in range(0, buf_len, bs)]
      buf_value = self.cri_target(buf_state)
      # buf_value = torch.cat(buf_value, dim=0)
      buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

      buf_r_sum, buf_adv_v = self.get_reward_sum(
        buf_len, buf_reward, buf_mask, buf_value)  # detach()
      buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
      # buf_adv_v: buffer data of adv_v value
      del buf_noise

    '''update network'''
    obj_critic = None
    obj_actor = None
    # assert buf_len * \
    #   self.EP.num_envs >= self.cfg.batch_size, f'buf_len {buf_len}, self.cfg.batch_size {self.cfg.batch_size}'
    # batch_size_per_env = self.cfg.batch_size//self.EP.num_envs
    num_traj_per_batch = self.cfg.batch_size//buf_len # split traj by env number
    for i in range(self.cfg.updates_per_rollout):
      # indices = torch.randint(buf_len, size=(batch_size_per_env,), requires_grad=False, device=self.cfg.device)
      # indices = torch.arange(start=(self.cfg.batch_size*i), end=self.cfg.batch_size*(i+1), requires_grad=False, device=self.cfg.device)%buf_len
      indices = torch.arange(start=(num_traj_per_batch*i), end=num_traj_per_batch*(i+1), requires_grad=False, device=self.cfg.device)%self.EP.num_envs

      state = buf_state[:,indices]
      r_sum = buf_r_sum[:,indices]
      adv_v = buf_adv_v[:,indices].squeeze(-1)
      action = buf_action[:,indices]
      logprob = buf_logprob[:,indices]

      '''PPO: Surrogate objective of Trust Region'''
      new_logprob, obj_entropy = self.act.get_logprob_entropy(
        state, action)  # it is obj_actor
      ratio = (new_logprob - logprob.detach()).exp()
      surrogate1 = adv_v * ratio
      surrogate2 = adv_v * \
        ratio.clamp(1 - self.cfg.ratio_clip, 1 + self.cfg.ratio_clip)
      obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
      obj_actor = obj_surrogate + obj_entropy * self.cfg.lambda_entropy
      self.optimizer_update(self.act_optimizer, obj_actor)

      # critic network predicts the reward_sum (Q value) of state
      value = self.cri(state).squeeze(1)
      obj_critic = self.criterion(value, r_sum)
      self.optimizer_update(self.cri_optimizer, obj_critic)
      if self.cfg.if_cri_target:
        self.soft_update(self.cri_target, self.cri,
                         self.cfg.soft_update_tau)

    a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
    return AttrDict(
      critic_loss=obj_critic.item(),
      actor_loss=-obj_actor.item(),
      a_std_log=a_std_log.item()
    )

  def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value):
    buf_r_sum = torch.empty(buf_len, dtype=torch.float32,
                            device=self.cfg.device)  # reward sum

    pre_r_sum = 0
    for i in range(buf_len - 1, -1, -1):
      buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
      pre_r_sum = buf_r_sum[i]
    buf_adv_v = buf_r_sum - buf_value[:, 0]
    return buf_r_sum, buf_adv_v

  def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value):
    buf_r_sum = torch.empty((buf_len, self.EP.num_envs, 1), dtype=torch.float32,
                            device=self.cfg.device)  # old policy value
    buf_adv_v = torch.empty((buf_len, self.EP.num_envs, 1), dtype=torch.float32,
                            device=self.cfg.device)  # advantage value
    pre_r_sum = 0
    pre_adv_v = 0  # advantage value of previous step
    for i in range(buf_len - 1, -1, -1):  # Notice: mask = (1-done) * gamma
      buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
      pre_r_sum = buf_r_sum[i]

      buf_adv_v[i] = ten_reward[i] + ten_mask[i] * pre_adv_v - ten_value[i]
      pre_adv_v = ten_value[i] + buf_adv_v[i] * self.cfg.lambda_gae_adv
    return buf_r_sum, buf_adv_v


'''test bench'''
if __name__ == '__main__':
  agent_base = AgentDDPG(1, 1, 1, 1, cfg=AttrDict(
    eval_gap=0, eval_steps_per_env=1, cwd=None))
