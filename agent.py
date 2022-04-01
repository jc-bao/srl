from turtle import done
import torch
import numpy as np
from copy import deepcopy
import os
import time
from attrdict import AttrDict

from net import ActorSAC, ActorFixSAC, CriticTwin, CriticREDq, CriticREDQ


class AgentBase:
  def __init__(self, net_dim: int, state_dim: int, action_dim: int, max_env_step: int, info_dim: int, goal_dim: int = 0, gpu_id=0, args=None):
    self.gamma = getattr(args, 'gamma', 0.99)
    self.env_num = getattr(args, 'env_num', 1)
    self.batch_size = getattr(args, 'batch_size', 128)
    self.repeat_times = getattr(args, 'repeat_times', 1.)
    self.reward_scale = getattr(args, 'reward_scale', 1.)
    self.lambda_gae_adv = getattr(args, 'lambda_entropy', 0.98)
    self.if_use_old_traj = getattr(args, 'if_use_old_traj', False)  # ?
    self.soft_update_tau = getattr(args, 'soft_update_tau', 2 ** -8)
    self.her_rate = getattr(args, 'her_rate', 0)
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.goal_dim = goal_dim
    self.max_env_step = max_env_step
    self.info_dim = info_dim

    if_act_target = getattr(args, 'if_act_target', False)
    if_cri_target = getattr(args, 'if_cri_target', False)
    if_off_policy = getattr(args, 'if_off_policy', True)
    learning_rate = getattr(args, 'learning_rate', 2 ** -12)

    self.states = None
    self.device = torch.device(f"cuda:{gpu_id}" if (
      torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    # self.traj_list = [[list() for _ in range(5 if if_off_policy else 6)]
    #                   for _ in range(self.env_num)]  # for `self.explore_vec_env()`

    act_class = getattr(self, 'act_class', None)
    cri_class = getattr(self, 'cri_class', None)
    self.act = act_class(net_dim, state_dim, action_dim).to(self.device)
    self.cri = cri_class(net_dim, state_dim, action_dim).to(
      self.device) if cri_class else self.act
    self.act_target = deepcopy(self.act) if if_act_target else self.act
    self.cri_target = deepcopy(self.cri) if if_cri_target else self.cri

    self.act_optimizer = torch.optim.Adam(self.act.parameters(), learning_rate)
    self.cri_optimizer = torch.optim.Adam(
      self.cri.parameters(), learning_rate) if cri_class else self.act_optimizer

    '''function'''
    self.criterion = torch.nn.SmoothL1Loss()

    if self.env_num == 1:
      self.explore_env = self.explore_one_env
    else:
      self.explore_env = self.explore_vec_env

    if getattr(args, 'if_use_per', False):  # PER (Prioritized Experience Replay) for sparse reward
      self.criterion = torch.nn.SmoothL1Loss(reduction='none')
      self.get_obj_critic = self.get_obj_critic_per
    else:
      self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
      self.get_obj_critic = self.get_obj_critic_raw

    '''record params'''
    self.total_step = 0
    self.last_eval_step = -1
    self.eval_gap = args.eval_gap
    self.eval_steps = args.eval_steps
    self.cwd = args.cwd
    self.start_time = time.time()
    self.r_max = -np.inf
    self.traj_idx = torch.arange(self.env_num, device=self.device)
    self.num_traj = self.env_num
    self.traj_list = torch.empty((self.env_num, self.max_env_step, state_dim +
                                 2+action_dim+info_dim), device=self.device, dtype=torch.float32)
    self.useless_step = 0

  def explore_one_env(self, env, target_steps_per_env) -> list:
    traj_list = list()
    last_done = [0, ]
    state = self.states[0]

    step_i = 0
    done = False
    while step_i < target_steps_per_env or not done:
      ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
      ten_a = self.act.get_action(
        ten_s.to(self.device)).detach().cpu()  # different
      next_s, reward, done, _ = env.step(ten_a[0].numpy())  # different

      traj_list.append((ten_s, reward, done, ten_a))  # different

      step_i += 1
      state = env.reset() if done else next_s

    self.states[0] = state
    last_done[0] = step_i
    return self.convert_trajectory(traj_list, last_done)  # traj_list

  def explore_vec_env(self, env, target_steps, eval_mode=False, random_mode=False, buffer=None) -> list:
    if eval_mode:
      num_ep = torch.zeros(self.env_num, device=self.device)
      ep_rew = torch.zeros(self.env_num, device=self.device)
      ep_step = torch.zeros(self.env_num, device=self.device)
      final_rew = torch.zeros(self.env_num, device=self.device)
    last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
    self.states = env.reset()
    ten_s = self.states

    step_i = 0
    ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
    traj_lens = torch.zeros(self.env_num, dtype=torch.long, device=self.device)
    traj_start_ptr = torch.zeros(
      self.env_num, dtype=torch.long, device=self.device)
    data_ptr = 0
    collected_steps = 0
    while collected_steps < target_steps:
      step_i += 1
      if random_mode:
        ten_a = torch.randn(self.env_num, self.action_dim, device=self.device)
      elif eval_mode:
        ten_a = self.act(ten_s).detach()
      else:
        ten_a = self.act.get_action(ten_s).detach()  # different
      ten_s_next, ten_rewards, ten_dones, ten_info = env.step(
        ten_a)  # different
      if eval_mode:
        done_idx = torch.where(ten_dones)[0]
        num_ep[done_idx] += 1
        ep_rew += ten_rewards
        ep_step += 1
        final_rew[done_idx] += ten_rewards[done_idx]
        collected_steps = (ep_step).sum()
      else:
        # preprocess info, add done, trajectory index, traj len, to left
        ten_info = torch.cat((ten_info, ten_dones.unsqueeze(1), self.traj_idx.unsqueeze(1),
                              torch.zeros((self.env_num, 2), device=self.device)), dim=-1)
        self.traj_list[:, data_ptr, :] = torch.cat((
          ten_s, ten_rewards.unsqueeze(
            1)*self.reward_scale, ((1-ten_dones)*self.gamma).unsqueeze(1),
          ten_a, ten_info), dim=-1)
        data_ptr = (data_ptr+1) % self.max_env_step

        traj_lens += 1

        # handle done
        done_idx = torch.where(ten_dones)[0]
        last_done[done_idx] = step_i  # behind `step_i+=1`
        done_env_num = torch.sum(ten_dones).type(torch.int32)
        self.traj_idx[done_idx] = (
          self.num_traj+torch.arange(done_env_num, device=self.device))
        self.num_traj += done_env_num  # do it after update traj_idx
        assert torch.max(
          self.traj_idx)+1 == self.num_traj, f'traj index {self.traj_idx} and num traj {self.num_traj} not match'
        # reset traj recorder
        if done_idx.shape[0] > 0:
          # add traj len, to left to info
          # traj len
          tiled_traj_len = traj_lens[done_idx].unsqueeze(
            1).tile(1, self.max_env_step).float()
          self.traj_list[done_idx, :, -2] = tiled_traj_len
          # to left = len - step
          self.traj_list[done_idx, :, -
                         1] = (tiled_traj_len - self.traj_list[done_idx, :, -self.info_dim+1])
          # mask

          # for env_idx in done_idx:
          #   for env_step in range(traj_lens[env_idx]):
          #     traj_list_idx = len(traj_list) + env_step - traj_lens[env_idx]
          #     # number of transitions, not equals to last index
          #     traj_list[traj_list_idx][-1][env_idx, -2] = traj_lens[env_idx]
          #     # to left length e.g. [3,2,1]. use randint(1,3) to get sample idx
          #     traj_list[traj_list_idx][-1][..., -1] = traj_lens[env_idx] - env_step
          state_traj = []
          other_traj = []
          for i in done_idx:  # TODO fix the one by one add traj process
            start_point = traj_start_ptr[i]
            end_point = (start_point + traj_lens[i]) % self.max_env_step
            ag_start = self.traj_list[i, start_point][self.state_dim +
                                                      2+self.info_dim:self.state_dim+4+self.info_dim]
            ag_end = self.traj_list[i, end_point][self.state_dim +
                                                  2+self.info_dim:self.state_dim+4+self.info_dim]
            # dropout unmoved exp
            if torch.norm(ag_start - ag_end) < 1e-2 and ~eval_mode:
              self.useless_step += traj_lens[i]
              pass
            if start_point < end_point:
              state_traj.append(
                self.traj_list[i, start_point:end_point, :self.state_dim])
              other_traj.append(
                self.traj_list[i, start_point:end_point, self.state_dim:])
            # elif end_point == 0:
            #   state_traj.append(self.traj_list[i, start_point:, :self.state_dim])
            #   other_traj.append(self.traj_list[i, start_point:, self.state_dim:])
            else:
              state_traj.append(torch.cat((
                self.traj_list[i, start_point:, :self.state_dim],
                self.traj_list[i, :end_point, :self.state_dim]
              ), dim=0))
              other_traj.append(torch.cat((
                self.traj_list[i, start_point:, self.state_dim:],
                self.traj_list[i, :end_point, self.state_dim:]
              ), dim=0))
          state_traj = torch.cat(state_traj, dim=0)
          other_traj = torch.cat(other_traj, dim=0)
          buffer.extend_buffer(state_traj, other_traj)
          self.total_step += state_traj.shape[0]
          collected_steps += state_traj.shape[0] 
          traj_start_ptr[done_idx] = (data_ptr+1) % self.max_env_step
          traj_lens[done_idx] = 0

      ten_s = ten_s_next

    if eval_mode:
      ep_rew /= num_ep
      final_rew /= num_ep
      ep_step /= num_ep
      return torch.mean(ep_rew), torch.mean(final_rew), torch.mean(ep_step)
    else:
      return self.total_step, self.useless_step

  def evaluate_save(self, env) -> (bool, bool):

    if self.total_step - self.last_eval_step < self.eval_gap:
      return None
    else:
      self.last_eval_step = self.total_step

      '''evaluate first time'''
      r_avg, r_final, s_avg = self.explore_vec_env(
        env, self.eval_steps, eval_mode=True)
      r_avg = r_avg.detach().cpu().numpy()
      s_avg = s_avg.detach().cpu().numpy()
      r_final = r_final.detach().cpu().numpy()

      '''save the policy network'''
      if_save = r_avg > self.r_max
      if if_save:  # save checkpoint with highest episode return
        self.r_max = r_avg  # update max reward (episode return)

        act_path = f"{self.cwd}/actor_{self.total_step:08}_{self.r_max:09.3f}.pth"
        # save policy and print
        # save policy network in *.pth
        torch.save(self.act.state_dict(), act_path)
        print(f"{self.total_step:8.2e}{self.r_max:8.2f} |")

    return AttrDict(
      step=self.total_step,
      r_avg=r_avg,
      r_final=r_final,
      s_avg=s_avg,
      used_time=time.time()-self.start_time
    )

  def convert_trajectory(self, buf_items, last_done):  # [ElegantRL.2022.01.01]
    # assert len(buf_items) == step_i
    # assert len(buf_items[0]) in {4, 5}
    # assert len(buf_items[0][0]) == self.env_num
    # state, reward, done, action, noise
    buf_items = list(map(list, zip(*buf_items)))
    # assert len(buf_items) == {4, 5}
    # assert len(buf_items[0]) == step
    # assert len(buf_items[0][0]) == self.env_num

    '''stack items'''
    buf_items[0] = torch.stack(buf_items[0])
    # action, info
    buf_items[3:] = [torch.stack(item) for item in buf_items[3:]]

    # action
    if len(buf_items[3].shape) == 2:
      buf_items[3] = buf_items[3].unsqueeze(2)

    # info

    if self.env_num > 1:
      # rew
      buf_items[1] = (torch.stack(buf_items[1]) *
                      self.reward_scale).unsqueeze(2)
      # mask
      buf_items[2] = ((1 - torch.stack(buf_items[2]))
                      * self.gamma).unsqueeze(2)
    else:
      buf_items[1] = (torch.tensor(buf_items[1], dtype=torch.float32) * self.reward_scale
                      ).unsqueeze(1).unsqueeze(2)
      buf_items[2] = ((1 - torch.tensor(buf_items[2], dtype=torch.float32)) * self.gamma
                      ).unsqueeze(1).unsqueeze(2)
    # assert all([buf_item.shape[:2] == (step, self.env_num) for buf_item in buf_items])

    '''splice items'''
    for j in range(len(buf_items)):
      cur_item = list()
      buf_item = buf_items[j]

      for env_i in range(self.env_num):
        last_step = last_done[env_i]

        pre_item = self.traj_list[env_i][j]
        if len(pre_item):
          cur_item.append(pre_item)

        cur_item.append(buf_item[:last_step, env_i])

        if self.if_use_old_traj:
          self.traj_list[env_i][j] = buf_item[last_step:, env_i]

      buf_items[j] = torch.vstack(cur_item)

    # on-policy:  buf_item = [states, rewards, dones, actions, noises]
    # off-policy: buf_item = [states, rewards, dones, actions, info]
    # buf_items = [buf_item, ...]
    return buf_items

  def get_obj_critic_raw(self, buffer, batch_size):
    """
    Calculate the loss of networks with **uniform sampling**.

    :param buffer: the ReplayBuffer instance that stores the trajectories.
    :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
    :return: the loss of the network and states.
    """
    with torch.no_grad():
      reward, mask, action, state, next_s, info = buffer.sample_batch(
        batch_size, her_rate=self.her_rate)
      next_a = self.act_target(next_s)
      critic_targets: torch.Tensor = self.cri_target(next_s, next_a)
      (next_q, min_indices) = torch.min(critic_targets, dim=1, keepdim=True)
      q_label = reward + mask * next_q
    q = self.cri(state, action)
    obj_critic = self.criterion(q, q_label)

    return obj_critic, state

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

  def save_or_load_agent(self, cwd, if_save):
    def load_torch_file(model_or_optim, _path):
      state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
      model_or_optim.load_state_dict(state_dict)

    name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optimizer),
                     ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optimizer), ]
    name_obj_list = [(name, obj)
                     for name, obj in name_obj_list if obj is not None]
    if if_save:
      for name, obj in name_obj_list:
        save_path = f"{cwd}/{name}.pth"
        torch.save(obj.state_dict(), save_path)
    else:
      for name, obj in name_obj_list:
        save_path = f"{cwd}/{name}.pth"
        load_torch_file(obj, save_path) if os.path.isfile(save_path) else None


class AgentSAC(AgentBase):
  def __init__(self, net_dim, state_dim, action_dim, max_env_step, goal_dim=0, info_dim=0, gpu_id=0, args=None):
    self.if_off_policy = True
    self.act_class = getattr(self, 'act_class', ActorSAC)
    self.cri_class = getattr(self, 'cri_class', CriticTwin)
    super().__init__(net_dim, state_dim, action_dim, max_env_step=max_env_step,
                     goal_dim=goal_dim, info_dim=info_dim, gpu_id=gpu_id, args=args)

    self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                  requires_grad=True, device=self.device)  # trainable parameter
    self.alpha_optim = torch.optim.Adam(
      (self.alpha_log,), lr=args.learning_rate)
    self.target_entropy = np.log(action_dim)

  def update_net(self, buffer):
    buffer.update_now_len()

    obj_critic = obj_actor = None
    for _ in range(int(1 + buffer.now_len * self.repeat_times / self.batch_size)):
      '''objective of critic (loss function of critic)'''
      obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
      self.optimizer_update(self.cri_optimizer, obj_critic)
      self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

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
      # self.soft_update(self.act_target, self.act, self.soft_update_tau) # SAC don't use act_target network

    return obj_critic.item(), -obj_actor.item(), self.alpha_log.exp().detach().item()

  def get_obj_critic_raw(self, buffer, batch_size):
    with torch.no_grad():
      reward, mask, action, state, next_s, info = buffer.sample_batch(
        batch_size, her_rate=self.her_rate)

      next_a, next_log_prob = self.act_target.get_action_logprob(
        next_s)  # stochastic policy
      next_q = self.cri_target.get_q_min(next_s, next_a)

      alpha = self.alpha_log.exp().detach()
      q_label = reward + mask * (next_q + next_log_prob * alpha)
    q1, q2 = self.cri.get_q1_q2(state, action)
    obj_critic = (self.criterion(q1, q_label) +
                  self.criterion(q2, q_label)) / 2
    return obj_critic, state

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
  def __init__(self, net_dim, state_dim, action_dim, max_env_step, goal_dim=0, info_dim=0, gpu_id=0, args=None):
    self.act_class = getattr(self, 'act_class', ActorFixSAC)
    self.cri_class = getattr(self, 'cri_class', CriticTwin)
    super().__init__(net_dim, state_dim, action_dim,
                     max_env_step, goal_dim, info_dim, gpu_id, args)
    self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    self.lambda_a_log_std = getattr(args, 'lambda_a_log_std', 2 ** -4)

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
        torch.ones((self.batch_size, 1), device=self.device)
      del buf_state

    alpha = self.alpha_log.exp().detach()
    update_a = 0
    obj_actor = torch.zeros(1)
    for update_c in range(1, int(2 + buffer.now_len * self.repeat_times / self.batch_size)):
      '''objective of critic (loss function of critic)'''
      obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
      self.optimizer_update(self.cri_optimizer, obj_critic)
      self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
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
        self.soft_update(self.act_target, self.act, self.soft_update_tau)
    return self.obj_c, -obj_actor.item(), alpha.item()


# Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
class AgentREDqSAC(AgentSAC):
  def __init__(self, net_dim, state_dim, action_dim, max_env_step, goal_dim=0, info_dim=0, gpu_id=0, args=None):
    self.act_class = getattr(self, 'act_class', ActorFixSAC)
    self.cri_class = getattr(self, 'cri_class', CriticREDq)
    super().__init__(net_dim, state_dim, action_dim, max_env_step, goal_dim, info_dim, gpu_id, args)
    self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

  def get_obj_critic_raw(self, buffer, batch_size):
    with torch.no_grad():
      reward, mask, action, state, next_s, info = buffer.sample_batch(batch_size, her_rate=self.her_rate)

      next_a, next_log_prob = self.act_target.get_action_logprob(
        next_s)  # stochastic policy
      next_q = self.cri_target.get_q_min(next_s, next_a)

      alpha = self.alpha_log.exp().detach()
      q_label = reward + mask * (next_q + next_log_prob * alpha)
    qs = self.cri.get_q_values(state, action)
    obj_critic = self.criterion(qs, q_label * torch.ones_like(qs))
    return obj_critic, state

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
  def __init__(self, net_dim, state_dim, action_dim, max_env_step, goal_dim=0, info_dim=0, gpu_id=0, args=None):
    self.act_class = getattr(self, 'act_class', ActorFixSAC)
    self.cri_class = getattr(self, 'cri_class', CriticREDQ)
    self.repeat_q_times = 1
    super().__init__(net_dim, state_dim, action_dim, max_env_step, goal_dim, info_dim, gpu_id, args)
    self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

  def get_obj_critic_raw(self, buffer, batch_size):
    with torch.no_grad():
      reward, mask, action, state, next_s, info = buffer.sample_batch(batch_size)

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
    for _ in range(int(1 + buffer.now_len * self.repeat_times / self.batch_size)):
      for _ in range(self.repeat_q_times):
        '''objective of critic (loss function of critic)'''
        obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
        self.optimizer_update(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

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
  def __init__(self, net_dim, state_dim, action_dim, max_env_step, goal_dim=0, info_dim=0, gpu_id=0, args=None):
    self.if_off_policy = True
    self.act_class = getattr(self, 'act_class', ActorSAC)
    self.cri_class = getattr(self, 'cri_class', CriticTwin)
    super().__init__(net_dim, state_dim, action_dim, max_env_step=max_env_step,
                     goal_dim=goal_dim, info_dim=info_dim, gpu_id=gpu_id, args=args)
    self.act.explore_noise = getattr(
      args, 'explore_noise', 0.1)  # set for `get_action()`

  def update_net(self, buffer):
    buffer.update_now_len()
    obj_critic = obj_actor = None
    for _ in range(int(1 + buffer.now_len * self.repeat_times / self.batch_size)):
      obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
      self.optimizer_update(self.cri_optimizer, obj_critic)
      self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

      action_pg = self.act(state)  # policy gradient
      obj_actor = -self.cri(state, action_pg).mean()
      self.optimizer_update(self.act_optimizer, obj_actor)
      self.soft_update(self.act_target, self.act, self.soft_update_tau)
    return obj_critic.item(), -obj_actor.item()




'''test bench'''
if __name__ == '__main__':
  agent_base = AgentDDPG(1, 1, 1, 1, args=AttrDict(
    eval_gap=0, eval_steps_per_env=1, cwd=None))
