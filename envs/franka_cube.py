import numpy as np
import os
import yaml
import gym
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
import torch
from attrdict import AttrDict
import pathlib

from .vec_task import VecTask


class FrankaCube(VecTask):
	def __init__(self, cfg_file='../isaac_configs/FrankaCube.yaml', **kwargs):
		# get config and setup base class
		cfg_path = pathlib.Path(__file__).parent.resolve()/cfg_file
		with open(cfg_path) as config_file:
			try:
				cfg = AttrDict(yaml.load(config_file, Loader=yaml.SafeLoader))
			except yaml.YAMLError as exc:
				print(exc)
		cfg.update(**kwargs) # overwrite params
		super().__init__(cfg)
		
		# spaces
		# block space
		self.torch_block_space = torch.distributions.uniform.Uniform(
			low=torch.tensor([[-0.3,-0.2,self.cfg.block_size/2]], device=self.device).repeat(self.cfg.num_goals, 1), 
			high=torch.tensor([[0.,0.2,self.cfg.block_size/2+0.001]], device=self.device).repeat(self.cfg.num_goals,1))
		# robot space
		self.torch_robot_space = torch.distributions.uniform.Uniform(
			low=torch.tensor([-0.35,-0.25,self.cfg.block_size/2], device=self.device),
			high=torch.tensor([0.05,0.25,self.cfg.block_size/2+0.25], device=self.device))
		self.grip_pos_mean = self.torch_robot_space.mean
		self.grip_pos_std = self.torch_robot_space.stddev
		# goal space
		self.torch_goal_space = torch.distributions.uniform.Uniform(
			low=torch.tensor([[-0.3,-0.2,self.cfg.block_size/2]], device=self.device).repeat(self.cfg.num_goals, 1), 
			high=torch.tensor([[0.,0.2,self.cfg.block_size/2+0.2]], device=self.device).repeat(self.cfg.num_goals,1))
		self.goal_mean = self.torch_goal_space.mean
		self.goal_std = self.torch_goal_space.stddev
		self.block_states_mean = torch.zeros(13, device=self.device, dtype=torch.float)
		self.block_states_mean[:3] = self.torch_goal_space.mean
		self.block_states_mean[3:6] = self.hand_vel_mean
		self.block_states_std = torch.ones(13, device=self.device, dtype=torch.float) 
		self.block_states_std[:3] = self.torch_goal_space.stddev
		self.block_states_std[3:6] = self.hand_vel_std

		# indices
		self.global_indices = torch.arange(
			self.num_envs * (1 + self.cfg.num_goals), dtype=torch.int32, device=self.device
		).view(self.num_envs, -1)

		self.reset()

	def _create_ground_plane(self):
		plane_params = gymapi.PlaneParams()
		plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
		self.gym.add_ground(self.sim, plane_params)

	def _create_envs(self, num_envs, spacing, num_per_row):
		# default values
		# colors
		self.colors = [gymapi.Vec3(*np.random.rand(3)) for _ in range(self.cfg.num_goals)]
		# finger shift 
		self.finger_shift = to_torch(self.cfg.finger_shift, device=self.device)
		# joint pos
		self.franka_default_dof_pos = to_torch(
			[-0.5709,  0.5089, -0.2695, -2.0247,  0.2203,  2.5064,  1.3707,  0.0200, 0.0200],
			device=self.device,
		)
		self.franka_default_orn = to_torch(
			[[0.924, -0.383, 0., 0.]], device = self.device).repeat(self.num_envs,1)

		lower = gymapi.Vec3(-spacing, -spacing, 0.0)
		upper = gymapi.Vec3(spacing, spacing, spacing)

		asset_root = os.path.join(
			os.path.dirname(os.path.abspath(__file__)),self.cfg.asset.assetRoot)
		franka_asset_file = self.cfg.asset.assetFileNameFranka

		# load franka asset
		asset_options = gymapi.AssetOptions()
		asset_options.flip_visual_attachments = True
		asset_options.fix_base_link = True
		asset_options.collapse_fixed_joints = True
		asset_options.disable_gravity = True
		asset_options.thickness = 0.001
		asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
		asset_options.use_mesh_materials = True
		franka_asset = self.gym.load_asset(
			self.sim, asset_root, franka_asset_file, asset_options
		)
		franka_dof_stiffness = to_torch(
			[400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6],
			dtype=torch.float,
			device=self.device,
		)
		franka_dof_damping = to_torch(
			[80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2],
			dtype=torch.float,
			device=self.device,
		)
		self.num_franka_bodies = self.gym.get_asset_rigid_body_count(
			franka_asset)
		self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
		self.franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
		self.franka_hand_index = int(self.franka_link_dict["panda_link7"])
		print("num franka bodies: ", self.num_franka_bodies)
		print("num franka dofs: ", self.num_franka_dofs)

		# set franka dof properties
		franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
		self.franka_dof_lower_limits = []
		self.franka_dof_upper_limits = []
		for i in range(self.num_franka_dofs):
			franka_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
			if self.cfg.physics_engine == gymapi.SIM_PHYSX:
				franka_dof_props["stiffness"][i] = franka_dof_stiffness[i]
				franka_dof_props["damping"][i] = franka_dof_damping[i]
			else:
				franka_dof_props["stiffness"][i] = 7000.0
				franka_dof_props["damping"][i] = 50.0

			self.franka_dof_lower_limits.append(franka_dof_props["lower"][i])
			self.franka_dof_upper_limits.append(franka_dof_props["upper"][i])

		self.franka_dof_lower_limits = to_torch(
			self.franka_dof_lower_limits, device=self.device
		)
		self.franka_dof_upper_limits = to_torch(
			self.franka_dof_upper_limits, device=self.device
		)
		self.franka_dof_speed_scales = torch.ones_like(
			self.franka_dof_lower_limits)
		self.franka_dof_speed_scales[[7, 8]] = 0.1
		franka_dof_props["effort"][7] = 200
		franka_dof_props["effort"][8] = 200

		# create block assets
		box_opts = gymapi.AssetOptions()
		box_opts.density = 400
		block_asset = self.gym.create_box(
			self.sim, self.cfg.block_length, self.cfg.block_size, self.cfg.block_size, box_opts)

		franka_start_pose = gymapi.Transform()
		franka_start_pose.p = gymapi.Vec3(-0.6, -0.4, 0.0)
		franka_start_pose.r = gymapi.Quat(0.0, 0.0, np.sqrt(2)/2, np.sqrt(2)/2)

		# compute aggregate size
		num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
		num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
		num_block_bodies = self.gym.get_asset_rigid_body_count(block_asset)
		num_block_shapes = self.gym.get_asset_rigid_shape_count(block_asset)
		max_agg_bodies = (
			num_franka_bodies + self.cfg.num_goals * num_block_bodies
		)
		max_agg_shapes = (
			num_franka_shapes + self.cfg.num_goals * num_block_shapes
		)

		self.frankas = []
		self.default_block_states = []
		self.prop_start = []
		self.envs = []

		for i in range(self.num_envs):
			# create env instance
			env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

			if self.cfg.aggregate_mode >= 3:
				self.gym.begin_aggregate(
					env_ptr, max_agg_bodies, max_agg_shapes, True)

			# Key: create Panda
			franka_actor = self.gym.create_actor(
				env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0
			)
			self.gym.set_actor_dof_properties(
				env_ptr, franka_actor, franka_dof_props)

			if self.cfg.aggregate_mode == 2:
				self.gym.begin_aggregate(
					env_ptr, max_agg_bodies, max_agg_shapes, True)

			if self.cfg.aggregate_mode == 1:
				self.gym.begin_aggregate(
					env_ptr, max_agg_bodies, max_agg_shapes, True)

			# create blocks
			if self.cfg.num_goals > 0:
				self.prop_start.append(self.gym.get_sim_actor_count(self.sim))

				xmin = -self.cfg.block_size * (self.cfg.num_goals - 1)
				self.block_handles = []
				for j in range(self.cfg.num_goals):
					block_state_pose = gymapi.Transform()
					block_state_pose.p.x = xmin + j * 2 * self.cfg.block_size
					block_state_pose.p.y = 0
					block_state_pose.p.z = 0
					block_state_pose.r = gymapi.Quat(0, 0, 0, 1)
					handle = self.gym.create_actor(
						env_ptr, block_asset, block_state_pose, "block{}".format(j), i, 0, 0,)
					self.block_handles.append(handle)
					self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.colors[j])
					self.default_block_states.append(
						[block_state_pose.p.x, block_state_pose.p.y, block_state_pose.p.z,
						 block_state_pose.r.x, block_state_pose.r.y, block_state_pose.r.z,
						 block_state_pose.r.w, ] + [0]*6
					)
			if self.cfg.aggregate_mode > 0:
				self.gym.end_aggregate(env_ptr)

			self.envs.append(env_ptr)
			self.frankas.append(franka_actor)

		# set control data
		self.hand_handle = self.gym.find_actor_rigid_body_handle(
			env_ptr, franka_actor, "panda_link7"
		)
		self.lfinger_handle = self.gym.find_actor_rigid_body_handle(
			env_ptr, franka_actor, "panda_leftfinger"
		)
		self.rfinger_handle = self.gym.find_actor_rigid_body_handle(
			env_ptr, franka_actor, "panda_rightfinger"
		)
		self.default_block_states = to_torch(
			self.default_block_states, device=self.device, dtype=torch.float
		).view(self.num_envs, self.cfg.num_goals, 13)
		self.init_data()

	def init_data(self):
		"""Generate System Initial Data
		"""
		hand = self.gym.find_actor_rigid_body_handle(
			self.envs[0], self.frankas[0], "panda_link7"
		)
		lfinger = self.gym.find_actor_rigid_body_handle(
			self.envs[0], self.frankas[0], "panda_leftfinger"
		)
		rfinger = self.gym.find_actor_rigid_body_handle(
			self.envs[0], self.frankas[0], "panda_rightfinger"
		)

		hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
		lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
		rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

		finger_pose = gymapi.Transform()
		finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
		finger_pose.r = lfinger_pose.r

		hand_pose_inv = hand_pose.inverse()
		grasp_pose_axis = 1
		franka_local_grasp_pose = hand_pose_inv * finger_pose
		franka_local_grasp_pose.p += gymapi.Vec3(
			*get_axis_params(0.04, grasp_pose_axis)
		)
		self.franka_local_grasp_pos = to_torch(
			[
				franka_local_grasp_pose.p.x,
				franka_local_grasp_pose.p.y,
				franka_local_grasp_pose.p.z,
			],
			device=self.device,
		).repeat((self.num_envs, 1))
		self.franka_local_grasp_rot = to_torch(
			[
				franka_local_grasp_pose.r.x,
				franka_local_grasp_pose.r.y,
				franka_local_grasp_pose.r.z,
				franka_local_grasp_pose.r.w,
			],
			device=self.device,
		).repeat((self.num_envs, 1))

		self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat(
			(self.num_envs, 1)
		)
		self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat(
			(self.num_envs, 1)
		)

		self.franka_lfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
		self.franka_rfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
		self.franka_lfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
		self.franka_rfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)

		# dof
		_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
		self.jacobian = gymtorch.wrap_tensor(_jacobian)
		self.j_eef = self.jacobian[:,
															 self.hand_handle - 1, :, :self.franka_hand_index]

		# root
		actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
		self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
			self.num_envs, -1, 13)
		# object observation
		self.block_states = self.root_state_tensor[:, 1:]
			
		# joint pos
		dof_state_tensor = self.gym.acquire_dof_state_tensor(
			self.sim)  
		self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
		self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
		self.franka_dof_targets = torch.zeros(
			(self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
		)
		self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
			:, : self.num_franka_dofs
		]
		self.franka_dof_pos = self.franka_dof_state[..., 0]
		self.franka_dof_vel = self.franka_dof_state[..., 1]
		self.finger_width = self.franka_dof_pos[:, self.franka_hand_index]
		self.finger_width_mean = 0.02
		self.finger_width_std = 0.04 / 12**0.5

		# object pos
		rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
		self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
			self.num_envs, -1, 13
		)  
		# robot observation
		self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
		self.hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
		self.hand_vel = self.rigid_body_states[:, self.hand_handle][:, 7:10]
		self.hand_vel_mean = 0
		self.hand_vel_std = 2*self.cfg.max_vel / 12**0.5
		self.num_bodies = self.rigid_body_states.shape[1]
		# store for visualization
		self.franka_lfinger_pos = self.rigid_body_states[:,
																										 self.lfinger_handle][:, 0:3]
		self.franka_rfinger_pos = self.rigid_body_states[:,
																										 self.rfinger_handle][:, 0:3]
		self.franka_lfinger_rot = self.rigid_body_states[:,
																										 self.lfinger_handle][:, 3:7]
		self.franka_rfinger_rot = self.rigid_body_states[:,
																										 self.rfinger_handle][:, 3:7]
		self.grip_pos = (self.franka_lfinger_pos+self.franka_rfinger_pos)/2 + self.finger_shift

	def compute_reward(self, ag, dg, info, normed = True):
		if normed: 
			ag = ag*self.goal_std + self.goal_mean
			dg = dg*self.goal_std + self.goal_mean
		if self.cfg.reward_type == 'sparse':
			return -torch.mean((torch.norm(ag.reshape(-1, self.cfg.num_goals, 3)-dg.reshape(-1, self.cfg.num_goals, 3),dim=-1) > self.cfg.err).type(torch.float32), dim=-1)
		elif self.cfg.reward_type == 'dense':
			return 1-torch.mean(torch.norm(ag.reshape(-1, self.cfg.num_goals, 3)-dg.reshape(-1, self.cfg.num_goals, 3),dim=-1), dim=-1)/self.cfg.num_goals

	def pre_physics_step(self, actions, use_init_pos=False):
		reset_idx = self.reset_buf.clone().type(torch.bool)
		done_env_num = reset_idx.sum()
		# reset goals
		self.goal_buf[reset_idx] = self.torch_goal_space.sample((done_env_num,)).reshape(-1, self.cfg.goal_dim)
		# reset blocks
		if done_env_num > 0:
			block_indices = self.global_indices[reset_idx, 1:].flatten()
			# set to default pos
			self.block_states[reset_idx] = self.default_block_states[reset_idx]
			in_hand = torch.rand((self.num_envs,),device=self.device) < self.cfg.inhand_rate
			random_idx = reset_idx&~in_hand
			inhand_idx = reset_idx&in_hand
			# change to hand or random posz
			if random_idx.any(): 
				self.block_states[random_idx,:,:3] = self.torch_block_space.sample((random_idx.sum(),))
			if inhand_idx.any():
				self.block_states[inhand_idx,torch.randint(self.cfg.num_goals, (1,), device=self.device)[0],:3] = self.grip_pos[inhand_idx]
			self.gym.set_actor_root_state_tensor_indexed(
				self.sim,
				gymtorch.unwrap_tensor(self.root_state_tensor),
				gymtorch.unwrap_tensor(block_indices),
				len(block_indices),)
		# reset state buf
		self.reset_buf[reset_idx] = 0
		self.progress_buf[reset_idx] = 0
		self.success_step_buf[reset_idx] = 0
		# set action here
		self.actions = torch.clip(actions.clone().to(self.device), -self.cfg.clip_actions, self.cfg.clip_actions)
		orn_err = self.orientation_error(self.franka_default_orn, self.hand_rot)
		pos_err = self.actions[..., :3] * self.cfg.dt * self.cfg.max_vel
		# clip with bound
		if self.cfg.bound_robot:
			pos_err = torch.clip(pos_err+self.grip_pos, self.torch_robot_space.low, self.torch_robot_space.high) - self.grip_pos
		# reset action (Note: when reset, the action is zero, to init obj in hand) 
		pos_err[reset_idx] = 0
		# pos_err[reset_idx] = self.torch_robot_space.sample((done_env_num,)) - self.hand_pos[reset_idx]
		# if done_env_num > 0:
		# 	pos_err[reset_idx] = self.block_states[reset_idx,0,:3] - self.hand_pos[reset_idx]
		dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
		self.franka_dof_targets[:, :self.franka_hand_index] = self.franka_dof_pos.squeeze(
			-1)[:, :self.franka_hand_index] + self.control_ik(dpose)
		# grip
		grip_acts = (self.actions[..., 3] + 1) * 0.02
		# reset gripper
		grip_acts[reset_idx] = 0
		self.franka_dof_targets[:, self.franka_hand_index:
														self.franka_hand_index+2] = grip_acts.unsqueeze(1).repeat(1, 2)
		# limit
		self.franka_dof_targets[~reset_idx, :self.num_franka_dofs] = tensor_clamp(
			self.franka_dof_targets[~reset_idx,
															: self.num_franka_dofs], self.franka_dof_lower_limits, self.franka_dof_upper_limits)
		self.franka_dof_targets[reset_idx & use_init_pos, :self.num_franka_dofs] = self.franka_default_dof_pos
		# Deploy actions
		self.gym.set_dof_position_target_tensor(
			self.sim, gymtorch.unwrap_tensor(self.franka_dof_targets))

	def post_physics_step(self):
		# update state data
		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_rigid_body_state_tensor(self.sim)
		self.gym.refresh_jacobian_tensors(self.sim)

		# update obs, rew, done, info
		self.grip_pos = (self.franka_lfinger_pos+self.franka_rfinger_pos)/2 + self.finger_shift
		normed_block_states = (self.block_states - self.block_states_mean)/self.block_states_std
		self.obs_buf = torch.cat((
			(self.grip_pos-self.grip_pos_mean)/self.grip_pos_std, # mid finger
			(self.hand_vel-self.hand_vel_mean)/self.hand_vel_std, 
			(self.finger_width.unsqueeze(-1)-self.finger_width_mean)/self.finger_width_std, # robot
			normed_block_states[..., 3:].view(self.num_envs, -1), # objects
			normed_block_states[...,:3].view(self.num_envs, -1), # achieved goal NOTE make sure it is close to end
			(self.goal_buf.view(self.num_envs, -1)-self.goal_mean)/self.goal_std
			),dim=-1)
		# rew
		self.rew_buf = self.compute_reward(self.block_states[..., :3], self.goal_buf, None, normed=False)
		# reset
		self.progress_buf += 1
		success_env = self.rew_buf > self.cfg.success_bar
		self.success_step_buf[~success_env] = self.progress_buf[~success_env]
		self.reset_buf = ((self.progress_buf >= (self.cfg.max_steps - 1)) | (self.progress_buf > self.success_step_buf + self.cfg.extra_steps)).type(torch.float)
		# info
		self.info_buf = torch.cat((
			success_env.type(torch.float).unsqueeze(-1),
			self.progress_buf.type(torch.float).unsqueeze(-1),
			normed_block_states[...,:3].view(self.num_envs,3*self.cfg.num_goals).type(torch.float),
		), dim=-1)

		# debug viz
		if self.viewer and self.cfg.debug_viz:
			self.gym.clear_lines(self.viewer)

			for i in range(self.num_envs):
				# draw finger mid
				finger_mid = (self.franka_lfinger_pos[i] + self.franka_rfinger_pos[i])/2 + self.finger_shift
				px = ((finger_mid + quat_apply(self.franka_lfinger_rot[i],
					to_torch([1, 0, 0], device=self.device) * 0.2,)).cpu().numpy())
				py = ((finger_mid + quat_apply(self.franka_lfinger_rot[i],
					to_torch([0, 1, 0], device=self.device) * 0.2,)).cpu().numpy())
				pz = ((finger_mid + quat_apply(self.franka_lfinger_rot[i],
					to_torch([0, 0, 1], device=self.device) * 0.2,)).cpu().numpy())
				p0 = finger_mid.cpu().numpy()
				self.gym.add_lines(self.viewer,self.envs[i],1,
					[p0[0], p0[1], p0[2], px[0], px[1], px[2]],[1, 0, 0],)
				self.gym.add_lines(self.viewer,self.envs[i],1,
					[p0[0], p0[1], p0[2], py[0], py[1], py[2]],[0, 1, 0],)
				self.gym.add_lines(self.viewer,self.envs[i],1,
					[p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],[0, 0, 1],)

				# draw goals
				for j in range(self.cfg.num_goals):
					sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * np.pi, 0, 0)
					sphere_pose = gymapi.Transform(r=sphere_rot)
					sphere_geom = gymutil.WireframeSphereGeometry(0.025, 12, 12, sphere_pose, 
						color=(self.colors[j].x, self.colors[j].y, self.colors[j].z))
					pos = gymapi.Transform()
					pos.p.x, pos.p.y, pos.p.z = self.goal_buf[i, j*3+0], self.goal_buf[i, j*3+1], self.goal_buf[i, j*3+2] 
					gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pos)
					# self.gym.add_lines(self.viewer, self.envs[i], 1, sphere_geom, [0,0,0.1])
				# draw goal space
				low = self.torch_goal_space.low[0]
				high = self.torch_goal_space.high[0]
				mean = (high+low)/2
				pos = gymapi.Transform()
				pos.p.x, pos.p.y, pos.p.z = mean[0], mean[1], mean[2] 
				box_geom = gymutil.WireframeBoxGeometry(high[0]-low[0], high[1]-low[1], high[2]-low[2], color=(0,0,1))
				gymutil.draw_lines(box_geom, self.gym, self.viewer, self.envs[i], pos)


	def control_ik(self, dpose):
		# solve damped least squares
		j_eef_T = torch.transpose(self.j_eef, 1, 2)
		lmbda = torch.eye(6, device=self.device) * (self.cfg.damping ** 2)
		u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda)
				 @ dpose).view(self.num_envs, self.franka_hand_index)
		return u

	def orientation_error(self, desired, current):
		cc = quat_conjugate(current)
		q_r = quat_mul(desired, cc)
		return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

	def ezpolicy(self, obs):
		up_step=5
		reach_step=20
		grasp_step=23
		end_step=33
		pos = obs[..., :3]
		obj = obs[..., 7:10].view(self.num_envs, self.cfg.num_goals, 3)
		goal = obs[..., 20:23].view(self.num_envs, self.cfg.num_goals, 3)
		action = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
		for env_id in range(self.num_envs):
			pos_now = pos[env_id]
			for goal_id in range(self.cfg.num_goals):
				obj_now = obj[env_id, goal_id] 
				goal_now = goal[env_id, goal_id]
				o2g = torch.norm(obj_now-goal_now)
				r2o = torch.norm(pos_now - obj_now)
				reached = o2g < self.cfg.err
				attached = r2o < self.cfg.err 
				if self.progress_buf[env_id] < up_step:
					action[env_id, 2] = 1
				elif up_step <= self.progress_buf[env_id] < reach_step:
					action[env_id, :3] = (obj_now - pos_now)/r2o*0.5
					action[env_id, 3] = 1
				elif reach_step <= self.progress_buf[env_id] < grasp_step:
					action[env_id, 3] = -1
				elif grasp_step <= self.progress_buf[env_id] < end_step:	
					action[env_id, :3] = (goal_now - obj_now)/o2g*0.5
		return action

	def update_config(self, cfg):
		cfg.update(
			# dim
			state_dim = cfg.shared_dim + cfg.per_seperate_dim * cfg.num_goals + cfg.per_goal_dim*cfg.num_goals, 
			info_dim = cfg.info_dim + cfg.num_goals*cfg.per_goal_dim,
			seperate_dim = cfg.per_seperate_dim * cfg.num_goals,
			goal_dim = cfg.per_goal_dim * cfg.num_goals,
			# device
			# when not need render, close graph device, else share with sim
			graphics_device_id = -1 if (~cfg.enable_camera_sensors and cfg.headless) else cfg.sim_device_id,
			sim_device = f'cuda:{cfg.sim_device_id}' if cfg.sim_device_id >= 0 else 'cpu',
			rl_device = f'cuda:{cfg.rl_device_id}' if cfg.rl_device_id >= 0 else 'cpu',
			# isaac 
			physics_engine = getattr(gymapi, cfg.physics_engine),
			# steps
			max_steps = cfg.base_steps*cfg.num_goals,
		)
		return cfg

	def env_params(self):
		return AttrDict(
			# dims
			action_dim = self.cfg.action_dim, 
			state_dim = self.cfg.state_dim, 
			shared_dim = self.cfg.shared_dim, 
			seperate_dim = self.cfg.seperate_dim,
			goal_dim = self.cfg.goal_dim, 
			info_dim = self.cfg.info_dim, # is_success, step, achieved_goal
			# numbers
			num_goals = self.cfg.num_goals, 
			num_envs = self.cfg.num_envs,
			max_env_step = self.cfg.max_steps,
			# functions
			reward_fn = self.compute_reward,
		)

gym.register(id='PandaPNP-v0', entry_point=FrankaCube)

if __name__ == '__main__':
	'''
	run random policy
	'''
	env = gym.make('PandaPNP-v0', num_envs=2, headless=False, inhand_rate=1)
	obs = env.reset()
	while True:
		# act = torch.tensor([[1,-0.01,1,0],[1,-0.01,1,0]])
		act = torch.rand((env.num_envs,4), device='cuda:0')*2-1
		# act[..., -1] = 0
		# act = env.ezpolicy(obs)
		obs, rew, done, info = env.step(act)