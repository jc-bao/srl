from cgitb import reset
from gc import get_stats
from re import S
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
    # dir
    if cfg.wandb:
      self.cfg.update(
        cwd=f'{wandb.run.dir}')
    else:
      self.cfg.update(
        cwd=f'results/{cfg.cwd}')
    os.makedirs(self.cfg.cwd, exist_ok=True)

    '''seed '''
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.set_default_dtype(torch.float32)

    '''env setup'''
    print('[Agent] env setup')
    self.env_kwargs = cfg.env
    self.env_kwargs.sim_device_id = self.cfg.gpu_id
    self.env_kwargs.rl_device_id = self.cfg.gpu_id
    if not self.cfg.render:
      self.env_kwargs.num_cameras = 0
    # handle special case: change goal number
    if self.cfg.curri is not None and 'num_goals' in self.cfg.curri:
      self.env_kwargs.num_goals = self.cfg.curri.num_goals.now
      print(f'[Env] change number of goals from {self.env_kwargs.num_goals} to {self.cfg.curri.num_goals.now} in the init...')
    self.env = gym.make(cfg.env_name, **self.env_kwargs)
    self.cfg.update(env_params=self.env.env_params(), env_cfg=self.env.cfg)
    reset_params = AttrDict()
    if self.cfg.curri is not None:
      for k, v in self.cfg.curri.items():
        reset_params[k] = v['now']
      self.env.reset(config=reset_params)
      print(f'[Env] reset to {reset_params}')
    # alias for env_params
    self.EP = self.cfg.env_params
    # rollout steps
    if self.cfg.steps_per_rollout is None:
      self.cfg.update(steps_per_rollout=int(
        self.EP.num_envs * self.EP.early_termin_step * 1.5))
      print(
        f'[Params] change step per rollout to {self.cfg.steps_per_rollout}')
    # eval steps
    if self.cfg.eval_eps is None:
      self.cfg.update(eval_eps=self.EP.num_envs)
      print(f'[Params] change epoches per eval to {self.cfg.eval_eps}')
    # update times
    if self.cfg.updates_per_rollout is None:
      self.cfg.update(updates_per_rollout=cfg.reuse *
                      cfg.steps_per_rollout//cfg.batch_size)

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
    self.total_save = 0
    self.r_max = -np.inf
    # params for tmp buffer to record traj info
    self.traj_idx = torch.arange(
      self.EP.num_envs, device=self.cfg.device)
    self.num_traj = self.EP.num_envs
    if self.cfg.wandb:
      wandb.config.update(self.cfg, allow_val_change=True)

    ''' data '''
    # tmp buffer
    self.buffer = ReplayBufferList(
      cfg) if 'PPO' in cfg.agent_name else ReplayBuffer(cfg)
    self.traj_list = torch.empty((self.EP.num_envs, self.EP.max_env_step,
                                 self.buffer.total_dim), device=self.cfg.device, dtype=torch.float32)

  def eval_vec_env(self, target_eps=None, render=False):
    # auto set target steps
    if target_eps is None:
      target_eps = self.cfg.eval_eps
    else:
      logging.warn(
        f'eval_env: target_steps is not None, forced to {target_eps}')
    # log data
    num_ep = torch.zeros(self.EP.num_envs,
                         device=self.cfg.device)
    ep_rew = torch.zeros(self.EP.num_envs,
                         device=self.cfg.device)
    ep_step = torch.zeros(self.EP.num_envs,
                          device=self.cfg.device)
    final_rew = torch.zeros(
      self.EP.num_envs, device=self.cfg.device)
    success_rate = torch.zeros(
      self.EP.num_envs, device=self.cfg.device)
    handover_success_rate = torch.zeros(
      (2, self.EP.num_goals+1), device=self.cfg.device)
    handover_num_ep = torch.ones((2, self.EP.num_goals+1),
                                 device=self.cfg.device)
    # reset
    ten_s, ten_rewards, ten_dones, ten_info = self.env.reset()
    if self.cfg.render and render:
      images = self.env.render(mode='rgb_array')
      videos = [[im] for im in images]
    # loop
    collected_eps = 0
    while collected_eps < target_eps or (num_ep < 1).any():
      ten_a = self.act(ten_s, self.EP.info_parser(ten_info, 'goal_mask')).detach()
      ten_s_next, ten_rewards, ten_dones, ten_info = self.env.step(
        ten_a)  # different
      if self.cfg.render and render:
        images = self.env.render(mode='rgb_array')
        for vi, im in zip(videos, images):
          vi.append(im)
      ten_dones = ten_dones.type(torch.bool)
      ten_goal_num = self.EP.info_parser(ten_info, 'goal_mask').sum(dim=-1)
      num_ep[ten_dones] += 1
      ep_rew += ten_rewards
      ep_step += 1
      final_rew[ten_dones] += ten_rewards[ten_dones]
      success_rate[ten_dones] += self.EP.info_parser(
        ten_info[ten_dones], 'success')
      try:
        for goal_id in range(2):
          goal_num = int(self.env.cfg.current_num_goals) + goal_id
          for i in range(self.EP.num_goals+1):
            now_done = ten_dones & (self.env.num_handovers == i) & (ten_goal_num == goal_num)
            handover_num_ep[goal_id, i] += now_done.sum()
            handover_success_rate[goal_id, i] += self.EP.info_parser(
              ten_info[now_done], 'success').sum()
      except Exception as e:
        print(f'{e}, fail to calcuate handover success rate')
      collected_eps = num_ep.sum()
      ten_s = ten_s_next
    # return
    video = None
    if self.cfg.render and render:
      videos = np.array(videos)
      video = np.concatenate(videos, axis=0)
      video = np.moveaxis(video, -1, 1)
    # curriculum
    final_rew = torch.mean(final_rew/num_ep).item()
    success_rate = torch.mean(success_rate/num_ep).item()
    reset_params = {}
    handover_success_rate /= handover_num_ep
    ho_success_dict = {}
    for goal_id in range(2):
      goal_num = int(self.env.cfg.current_num_goals) + goal_id
      for i in range(goal_num+1):
        ho_success_dict[f'handover_{i}_{goal_num}_success_rate'] = handover_success_rate[goal_id, i].item()
    results = AttrDict(
      steps=self.total_step,
      ep_rew=torch.mean(ep_rew / num_ep).item(),
      final_rew=final_rew,
      success_rate=success_rate,
      **ho_success_dict,
      ep_steps=torch.mean(ep_step / num_ep).item(),
      video=video)  # record curriculum
    if self.cfg.curri is not None:
      for k, v in self.cfg.curri.items():
        if eval(v['indicator']) > v['bar'] and abs(v['now'] - v['end']) > abs(v['step']/2):
          self.cfg.curri[k]['now'] += v['step']
        reset_params[k] = self.cfg.curri[k]['now']
        '''
        Old version of goal number curriculum, need to flush out buffer
        if 'num_goals' in reset_params and reset_params[
            'num_goals'] != self.env.cfg.num_goals:
          self.env_kwargs.num_goals = reset_params['num_goals']
          print(f'[Env] change num_goals to {self.env_kwargs.num_goals}, rebuild env...')
          self.env.close()
          del self.env
          self.env = gym.make(self.cfg.env_name, **self.env_kwargs)
          self.cfg.update(env_params=self.env.env_params(), env_cfg=self.env.cfg)
          self.EP = self.cfg.env_params
          print(f'[Agent] change num_goals to {self.env_kwargs.num_goals}, rebuild buffer...')
          del self.buffer
          del self.traj_list
          self.buffer = ReplayBufferList(
            self.cfg) if 'PPO' in self.cfg.agent_name else ReplayBuffer(self.cfg)
          self.traj_list = torch.empty((self.EP.num_envs, self.EP.max_env_step,
            self.buffer.total_dim), device=self.cfg.device, dtype=torch.float32)
          self.act.EP = self.EP
          self.cri.EP = self.EP
          self.act_target.EP = self.EP
          self.cri_target.EP = self.EP
          if self.cfg.curri.num_goals.change_other_back:
            for k, v in self.cfg.curri.items():
              if k != 'num_goals':
                print(f'[Curri] change {k} to {v["init"]}')
                self.cfg.curri[k]['now'] = self.cfg.curri[k]['init']
                reset_params[k] = self.cfg.curri[k]['now']
        '''
      self.env.reset(config=reset_params)
    results.update(curri=reset_params)
    return results

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
    num_ep = torch.zeros(
      self.EP.num_envs, dtype=torch.long, device=self.cfg.device)
    traj_lens = torch.zeros(self.EP.num_envs,
                            dtype=torch.long, device=self.cfg.device)
    data_ptr = 0  # where to store data
    collected_steps = 0  # data added to buffer
    useless_steps = 0  # data explored but dropped
    # loop
    s, rew, done, info = self.env.reset()
    act = self.act.get_action(s, self.EP.info_parser(info, 'goal_mask')).detach()
    while collected_steps < target_steps or (num_ep < 1).any():
      # setup next state
      s, rew, done, info = self.env.step(act)  # different
      act = self.act.get_action(s, self.EP.info_parser(info, 'goal_mask')).detach()

      # update buffer
      num_ep[done.type(torch.bool)] += 1
      # preprocess info, add [1]trajectory index, [2]traj len, [3]to left
      info = self.EP.info_updater(info, AttrDict(traj_idx=self.traj_idx))
      # add data to tmp buffer
      self.traj_list[:, data_ptr, :] = torch.cat((
        s,  # state(t)
        rew.unsqueeze(1)*self.cfg.reward_scale,  # reward(t)
        ((1-done)*self.cfg.gamma).unsqueeze(1),  # mask(t)
        act,  # action(t)
        info,  # info(t+1)
      ), dim=-1)
      # update ptr for tmp buffer
      data_ptr = (data_ptr+1) % self.EP.max_env_step
      # update trajectory info
      traj_lens += 1
      done_idx = torch.where(done)[0]
      done_num_envs = torch.sum(done).type(torch.int32)
      # update traj index
      self.traj_idx[done_idx] = (
        self.num_traj+torch.arange(done_num_envs, device=self.cfg.device))
      self.num_traj += done_num_envs
      assert torch.max(self.traj_idx) + 1 == self.num_traj, \
        f'traj index {self.traj_idx} and num traj {self.num_traj} not match'
      # reset traj recorder and add extra traj info
      if done_idx.shape[0] > 0:
        # tile traj len for later use
        tiled_traj_len = traj_lens[done_idx].unsqueeze(
          1).tile(1, self.EP.max_env_step).float()
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
              traj_len=tiled_traj_len)))
        # add to data buffer
        results = self.save_to_buffer(
          done_idx, traj_start_ptr, traj_lens)
        self.total_step += (results.collected_steps +
                            results.useless_steps)
        collected_steps += results.collected_steps
        useless_steps += results.useless_steps
        # reset record params
        traj_start_ptr[done_idx] = data_ptr
        # traj_start_ptr[done_idx] = data_ptr
        traj_lens[done_idx] = 0

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
      # dropout unmoved experience
      if getattr(end_info_dict, 'early_termin', False) and self.cfg.dropout_early_termin:
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
      # tleft = self.buffer.data_parser(traj_data, 'info.tleft')
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
      next_a = self.act_target(trans.next_state, self.EP.info_parser(trans.info, 'goal_mask'))  # stochastic policy
      critic_targets: torch.Tensor = self.cri_target(
        trans.next_state, next_a)
      (next_q, min_indices) = torch.min(
        critic_targets, dim=1, keepdim=True)
      q_label = trans.rew.unsqueeze(-1) + \
        trans.mask.unsqueeze(-1) * next_q
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
      (next_q, min_indices) = torch.min(
        critic_targets, dim=1, keepdim=True)
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

    name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optimizer),
                     ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optimizer), ]
    name_obj_list = [(name, obj)
                     for name, obj in name_obj_list if obj is not None]
    if if_save:
      data = {'step': self.total_step, 'curri': self.cfg.curri, 'total_save': self.total_save}
      for name, obj in name_obj_list:
        data[name] = obj.state_dict()
      last_save_path = f"{cwd}/{file_tag}_{self.total_save}.pth"
      if os.path.exists(last_save_path):
        os.remove(last_save_path)  # remove this file to save space
      self.total_save += 1
      save_path = f"{cwd}/{file_tag}_{self.total_save}.pth"
      torch.save(data, save_path)
      if self.cfg.wandb:
        wandb.save(save_path, base_path=cwd, policy="now")  # upload now
    else:
      if self.cfg.wid is not None:
        if self.cfg.load_project is None:
          self.cfg.load_project = self.cfg.project
        save_path = wandb.restore(
          f'{self.cfg.load_folder}{file_tag}.pth', f'{self.cfg.entity}/{self.cfg.load_project}/{self.cfg.wid}').name
      elif self.cfg.load_path is not None:
        save_path = self.cfg.load_path
      with open(save_path, 'rb') as f:
        data = torch.load(f, map_location=lambda storage, loc: storage)
      if self.cfg.resume_mode == 'continue':
        self.total_step = data['step']
        self.total_save = data['total_save']
      if self.cfg.load_curri is None:
        self.cfg.load_curri = (self.cfg.resume_mode == 'continue')
      print('[Load] load curri:', self.cfg.load_curri)
      if self.cfg.load_curri:
        for k, v in data['curri'].items():
          if k in self.cfg.curri:
            print(f'[Load] set {k} to {v["now"]}')
            self.cfg['curri'][k]['now'] = v['now']
      for name, obj in name_obj_list:
        obj.load_state_dict(data[name])


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

    obj_critic = obj_actor = torch.zeros(1, device=self.cfg.device)[0]
    for _ in range(int(self.buffer.now_len / self.buffer.max_len * self.cfg.updates_per_rollout)):
      '''objective of critic (loss function of critic)'''
      obj_critic, trans = self.get_obj_critic(
        self.buffer, self.cfg.batch_size)
      self.optimizer_update(self.cri_optimizer, obj_critic)
      self.soft_update(self.cri_target, self.cri,
                       self.cfg.soft_update_tau)

      '''objective of alpha (temperature parameter automatic adjustment)'''
      a_noise_pg, log_prob = self.act.get_action_logprob(
        trans.state, self.EP.info_parser(trans.info, 'goal_mask'))  # policy gradient
      obj_alpha = (self.alpha_log * (log_prob -
                                     self.target_entropy).detach()).mean()
      self.optimizer_update(self.alpha_optim, obj_alpha)

      '''objective of actor'''
      alpha = self.alpha_log.exp().detach()
      with torch.no_grad():
        self.alpha_log[:] = self.alpha_log.clamp(-20, 2)

      q_value_pg = self.cri(trans.state, a_noise_pg, self.EP.info_parser(trans.info, 'goal_mask'))
      obj_actor = -(q_value_pg + log_prob * alpha).mean()
      self.optimizer_update(self.act_optimizer, obj_actor)
      # SAC don't use act_target network
      self.soft_update(self.act_target, self.act, self.cfg.soft_update_tau)

    return AttrDict(
      critic_loss=obj_critic.item(),
      actor_loss=-obj_actor.item(),
      alpha_log=self.alpha_log.exp().detach().item(),
      ag_random_relabel_rate=self.buffer.ag_random_relabel_rate.item(),
      g_random_relabel_rate=self.buffer.g_random_relabel_rate.item(),
    )

  def get_obj_critic_raw(self, buffer, batch_size):
    with torch.no_grad():
      trans = buffer.sample_batch(batch_size, her_rate=self.cfg.her_rate)
      mask = self.EP.info_parser(trans.info, 'goal_mask')
      next_a, next_log_prob = self.act_target.get_action_logprob(
        trans.next_state, mask)  # stochastic policy
      next_q = self.cri_target.get_q_min(trans.next_state, next_a, mask=mask)

      alpha = self.alpha_log.exp().detach()
      q_label = trans.rew.unsqueeze(-1) + trans.mask.unsqueeze(-1) * \
        (next_q + next_log_prob * alpha)
    if self.cfg.mirror_q_reg_coef > 0:
      qs, q_std = self.cri.get_q_all(trans.state, trans.action, get_mirror_std=True, mask=mask)
      obj_critic = self.criterion(
        qs, q_label * torch.ones_like(qs)) + q_std.mean() * self.cfg.mirror_q_reg_coef
    elif self.cfg.mirror_feature_reg_coef > 0:
      qs, feature_norm = self.cri.get_q_all(trans.state, trans.action, get_embedding_norm=True, mask=mask)
      obj_critic = self.criterion(
        qs, q_label * torch.ones_like(qs)) + feature_norm.mean() * self.cfg.mirror_feature_reg_coef
    else:
      qs = self.cri.get_q_all(trans.state, trans.action, mask=mask)
      obj_critic = self.criterion(qs, q_label * torch.ones_like(qs))
    return obj_critic, trans


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
      obj_critic, state = self.get_obj_critic(
        buffer, self.cfg.batch_size)
      self.optimizer_update(self.cri_optimizer, obj_critic)
      self.soft_update(self.cri_target, self.cri,
                       self.cfg.soft_update_tau)
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
    td_error = self.criterion(
      qs, q_label * torch.ones_like(qs)).mean(dim=1)
    obj_critic = (td_error * is_weights).mean()

    buffer.td_error_update(td_error.detach())
    return obj_critic, state


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
        obj_critic, state = self.get_obj_critic(
          buffer, self.cfg.batch_size)
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
      obj_critic, state = self.get_obj_critic(
        self.buffer, self.cfg.batch_size)
      self.optimizer_update(self.cri_optimizer, obj_critic)
      self.soft_update(self.cri_target, self.cri,
                       self.cfg.soft_update_tau)

      action_pg = self.act(state)  # policy gradient
      obj_actor = -self.cri(state, action_pg).mean()
      self.optimizer_update(self.act_optimizer, obj_actor)
      self.soft_update(self.act_target, self.act,
                       self.cfg.soft_update_tau)
    return AttrDict(
      critic_loss=obj_critic.item(),
      actor_loss=-obj_actor.item(),
    )


class AgentTD3(AgentDDPG):
  def __init__(self, cfg):
    super().__init__(cfg=cfg)

  def update_net(self) -> tuple:
    self.buffer.update_now_len()
    obj_critic = obj_actor = None
    for update_c in range(1+int(self.buffer.now_len / self.buffer.max_len * self.cfg.updates_per_rollout)):
      obj_critic, state = self.get_obj_critic(
        self.buffer, self.cfg.batch_size)
      self.optimizer_update(self.cri_optimizer, obj_critic)

      if update_c % self.cfg.policy_update_gap == 0:  # delay update
        action_pg = self.act(state)  # policy gradient
        obj_actor = -self.cri_target(state, action_pg).mean()
        self.optimizer_update(self.act_optimizer, obj_actor)
      if update_c % self.cfg.update_freq == 0:  # delay update
        self.soft_update(self.cri_target, self.cri,
                         self.cfg.soft_update_tau)
        self.soft_update(self.act_target, self.act,
                         self.cfg.soft_update_tau)
    return AttrDict(
      critic_loss=obj_critic.item()/2,
      actor_loss=-obj_actor.item(),
      ag_random_relabel_rate=self.buffer.ag_random_relabel_rate.item(),
      g_random_relabel_rate=self.buffer.g_random_relabel_rate.item(),
    )

  def get_obj_critic_raw(self, buffer, batch_size):
    with torch.no_grad():
      trans = buffer.sample_batch(batch_size, her_rate=self.cfg.her_rate)
      next_a = self.act_target.get_action_noise(
        trans.next_state, self.cfg.policy_noise)
      next_q = self.cri_target.get_q_min(trans.next_state, next_a)
      q_label = trans.rew.unsqueeze(-1) + trans.mask.unsqueeze(-1) * next_q
    # q1, q2 = self.cri.get_q_all(trans.state, trans.action)
    # obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
    qs = self.cri.get_q_all(trans.state, trans.action)
    obj_critic = self.criterion(qs, q_label * torch.ones_like(qs))
    return obj_critic, trans.state

  def get_obj_critic_per(self, buffer, batch_size):
    with torch.no_grad():
      reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
        batch_size
      )
      next_a = self.act_target.get_action_noise(
        next_s, self.policy_noise
      )  # policy noise
      next_q = torch.min(
        *self.cri_target.get_q_all(next_s, next_a)
      )  # twin critics
      q_label = reward + mask * next_q

    q1, q2 = self.cri.get_q_all(state, action)
    td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
    obj_critic = (td_error * is_weights).mean()

    buffer.td_error_update(td_error.detach())
    return obj_critic, state


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
      ten_s_next, ten_rewards, ten_dones, _ = self.env.step(
        get_a_to_e(ten_a))
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
      buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / \
        (buf_adv_v.std() + 1e-5)
      # buf_adv_v: buffer data of adv_v value
      del buf_noise

    '''update network'''
    obj_critic = None
    obj_actor = None
    # assert buf_len * \
    #   self.EP.num_envs >= self.cfg.batch_size, f'buf_len {buf_len}, self.cfg.batch_size {self.cfg.batch_size}'
    # batch_size_per_env = self.cfg.batch_size//self.EP.num_envs
    num_traj_per_batch = self.cfg.batch_size//buf_len  # split traj by env number
    for i in range(self.cfg.updates_per_rollout):
      # indices = torch.randint(buf_len, size=(batch_size_per_env,), requires_grad=False, device=self.cfg.device)
      # indices = torch.arange(start=(self.cfg.batch_size*i), end=self.cfg.batch_size*(i+1), requires_grad=False, device=self.cfg.device)%buf_len
      indices = torch.arange(start=(num_traj_per_batch*i), end=num_traj_per_batch*(
        i+1), requires_grad=False, device=self.cfg.device) % self.EP.num_envs

      state = buf_state[:, indices]
      r_sum = buf_r_sum[:, indices]
      adv_v = buf_adv_v[:, indices].squeeze(-1)
      action = buf_action[:, indices]
      logprob = buf_logprob[:, indices]

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

      buf_adv_v[i] = ten_reward[i] + \
        ten_mask[i] * pre_adv_v - ten_value[i]
      pre_adv_v = ten_value[i] + buf_adv_v[i] * self.cfg.lambda_gae_adv
    return buf_r_sum, buf_adv_v


'''test bench'''
if __name__ == '__main__':
  agent_base = AgentDDPG(1, 1, 1, 1, cfg=AttrDict(
    eval_gap=0, eval_steps_per_env=1, cwd=None))
