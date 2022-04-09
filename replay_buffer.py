import os
import numpy as np
import numpy.random as rd
import torch
from attrdict import AttrDict


class ReplayBuffer:  # for off-policy
  def __init__(self, cfg):
    self.now_len = 0
    self.next_idx = 0
    self.prev_idx = 0
    self.if_full = False
    self.max_len = cfg.buffer_size
    self.cfg = cfg
    self.EP = cfg.env_params
    self.device = cfg.device

    # reward_dim + mask_dim + action_dim + info
    self.total_other_dim = 1 + 1 + self.EP.action_dim + self.EP.info_dim
    self.total_dim = self.total_other_dim+self.EP.state_dim
    self.data = torch.empty(
      (self.max_len, self.EP.state_dim+self.total_other_dim), dtype=torch.float32, device=self.device)

  def extend_buffer(self, traj_data):  # CPU array to CPU array
    size = len(traj_data)
    next_idx = self.next_idx + size
    if next_idx > self.max_len:
      self.data[self.next_idx:self.max_len] = traj_data[:self.max_len - self.next_idx]
      self.if_full = True
      next_idx = next_idx - self.max_len
      self.data[0:next_idx] = traj_data[-next_idx:]
    else:
      self.data[self.next_idx:next_idx] = traj_data
    self.next_idx = next_idx

  def sample_batch(self, batch_size, her_rate=0, indices = None) -> tuple:
    if indices is None:
      indices = torch.randint(self.now_len - 1, size=(batch_size,), dtype=torch.long, device=self.device)
    else:
      batch_size = indices.shape[0] 
    # filter done state
    trans =self.data[indices] 
    trans_dict = self.data_parser(trans)
    filtered_local_indices = trans_dict.mask > 1e-3
    # filter final state
    indices = indices[filtered_local_indices]
    trans = trans[filtered_local_indices]
    trans_dict = self.data_parser(trans)
    next_trans = self.data[indices+1]
    next_trans_dict = self.data_parser(next_trans)
    # get state
    if her_rate > 0:
      indices_her_local_bool = torch.rand(size=indices.shape, device=self.device) < her_rate
      info_dict = self.EP.info_parser(trans_dict.info[indices_her_local_bool])
      tleft = info_dict.tleft.long()
      indices_her_global = indices[indices_her_local_bool]
      # get local variables
      idx_shift = (torch.rand(tleft.shape, device=self.device)*(tleft)).long()
      fut_trans = self.data[(indices_her_global+idx_shift)%self.max_len]
      fut_info = self.data_parser(fut_trans).info
      fut_ag = self.EP.info_parser(fut_info).ag
      # NOTE: to sample next state
      # relabel
      trans_dict.state[indices_her_local_bool] = self.EP.obs_updater(trans_dict.state[indices_her_local_bool], AttrDict(g=fut_ag))
      next_trans_dict.state[indices_her_local_bool] = self.EP.obs_updater(next_trans_dict.state[indices_her_local_bool], AttrDict(g=fut_ag))
      # recompute
      trans_dict.rew[indices_her_local_bool] = self.EP.compute_reward(info_dict.ag, fut_ag, None)
    return AttrDict(
      rew = trans_dict.rew, 
      mask = trans_dict.mask, # mask
      action = trans_dict.action, # action
      state = trans_dict.state, # state
      next_state = next_trans_dict.state, # next_state
      info = trans_dict.info) # info

  def sample_batch_r_m_a_s(self):
    if self.prev_idx <= self.next_idx:
      r = self.data[self.prev_idx:self.next_idx, 0:1]
      m = self.data[self.prev_idx:self.next_idx, 1:2]
      a = self.data[self.prev_idx:self.next_idx, 2:]
      s = self.buf_state[self.prev_idx:self.next_idx]
    else:
      r = torch.vstack((self.data[self.prev_idx:, 0:1],
                        self.data[:self.next_idx, 0:1]))
      m = torch.vstack((self.data[self.prev_idx:, 1:2],
                        self.data[:self.next_idx, 1:2]))
      a = torch.vstack((self.data[self.prev_idx:, 2:],
                        self.data[:self.next_idx, 2:]))
      s = torch.vstack((self.buf_state[self.prev_idx:],
                        self.buf_state[:self.next_idx],))
    self.prev_idx = self.next_idx
    return r, m, a, s  # reward, mask, action, state

  def update_now_len(self):
    self.now_len = self.max_len if self.if_full else self.next_idx

  def save_or_load_history(self, cwd, if_save, buffer_id=0):
    save_path = f"{cwd}/replay_{buffer_id}.npz"

    if if_save:
      self.update_now_len()
      state_dim = self.buf_state.shape[1]
      other_dim = self.data.shape[1]
      buf_state = np.empty((self.max_len, state_dim),
                           dtype=np.float16)  # sometimes np.uint8
      buf_other = np.empty((self.max_len, other_dim), dtype=np.float16)

      temp_len = self.max_len - self.now_len
      buf_state[0:temp_len] = self.buf_state[self.now_len:self.max_len].detach(
      ).cpu().numpy()
      buf_other[0:temp_len] = self.data[self.now_len:self.max_len].detach(
      ).cpu().numpy()

      buf_state[temp_len:] = self.buf_state[:self.now_len].detach().cpu().numpy()
      buf_other[temp_len:] = self.data[:self.now_len].detach().cpu().numpy()

      np.savez_compressed(save_path, buf_state=buf_state, buf_other=buf_other)
      print(f"| ReplayBuffer save in: {save_path}")
    elif os.path.isfile(save_path):
      buf_dict = np.load(save_path)
      buf_state = buf_dict['buf_state']
      buf_other = buf_dict['buf_other']

      buf_state = torch.as_tensor(
        buf_state, dtype=torch.float32, device=self.device)
      buf_other = torch.as_tensor(
        buf_other, dtype=torch.float32, device=self.device)
      self.extend_buffer(buf_state, buf_other)
      self.update_now_len()
      print(f"| ReplayBuffer load: {save_path}")

  def data_parser(self, data):
    action_start = self.EP.state_dim+2
    info_start = action_start+self.EP.action_dim
    return AttrDict(
      # state part
      state = data[..., :self.EP.state_dim],
      rew = data[..., self.EP.state_dim], 
      mask = data[..., self.EP.state_dim+1],
      action = data[..., action_start:info_start],
      info = data[..., info_start:info_start+self.EP.info_dim],
    )

class ReplayBufferList(list):  # for on-policy
  def __init__(self, cfg=None):
    list.__init__(self)

  def update_buffer(self, traj_list):
    cur_items = list(map(list, zip(*traj_list)))
    self[:] = [torch.cat(item, dim=0) for item in cur_items]

    steps = self[1].shape[0]
    r_exp = self[1].mean().item()
    return steps, r_exp
