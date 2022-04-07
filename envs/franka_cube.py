import numpy as np
import os, sys, time
import yaml
import gym
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
import torch
from attrdict import AttrDict
import pathlib

class FrankaCube(gym.Env):
	def __init__(self, cfg_file='isaac_configs/FrankaCube.yaml', **kwargs):
		# get config and setup base class
		cfg_path = pathlib.Path(__file__).parent.resolve()/cfg_file
		with open(cfg_path) as config_file:
			try:
				cfg = AttrDict(yaml.load(config_file, Loader=yaml.SafeLoader))
			except yaml.YAMLError as exc:
				print(exc)
		cfg.update(**kwargs) # overwrite params from args
		self.cfg = self.update_config(cfg) # auto update sim params
		self.device = self.cfg.sim_device

		# setup isaac
		self.gym = gymapi.acquire_gym()
		self.sim = self.gym.create_sim(
			self.cfg.sim_device_id, self.cfg.graphics_device_id, self.cfg.physics_engine, self.cfg.sim_params)
		self._create_ground_plane()
		self._create_envs()
		self.gym.prepare_sim(self.sim)
		self.set_viewer()
		
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

		# indices
		self.global_indices = torch.arange(
			self.cfg.num_envs * (1 + self.cfg.num_goals*2), dtype=torch.int32, device=self.device
		).view(self.cfg.num_envs, -1)

		self.reset()

	def set_viewer(self):
		self.enable_viewer_sync = True
		self.viewer = None
		# if running with a viewer, set up keyboard shortcuts and camera
		if self.cfg.headless == False:
			# subscribe to keyboard shortcuts
			camera_setting = gymapi.CameraProperties()
			self.viewer = self.gym.create_viewer(self.sim, camera_setting)
			cam_pos = gymapi.Vec3(1.5, 1.5, 0)
			look_at = gymapi.Vec3(0.5, 1, 0)
			self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, look_at)
			self.gym.subscribe_viewer_keyboard_event(
				self.viewer, gymapi.KEY_ESCAPE, "QUIT")
			self.gym.subscribe_viewer_keyboard_event(
				self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
			# set the camera position based on up axis
			sim_params = self.gym.get_sim_params(self.sim)
			if sim_params.up_axis == gymapi.UP_AXIS_Z:
				cam_pos = gymapi.Vec3(1.0, -1.0, 1.0)
				cam_target = gymapi.Vec3(-0.2, 0.0, 0.0)
			else:
				cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
				cam_target = gymapi.Vec3(10.0, 0.0, 15.0)
			self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

	def _create_ground_plane(self):
		plane_params = gymapi.PlaneParams()
		plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
		self.gym.add_ground(self.sim, plane_params)

	def _create_envs(self):
		# default values
		num_per_row = int(np.sqrt(self.cfg.num_envs))
		# colors
		self.colors = [gymapi.Vec3(*np.random.rand(3)) for _ in range(self.cfg.num_goals)]
		# finger shift 
		self.finger_shift = to_torch(self.cfg.finger_shift, device=self.device)
		# joint pos
		self.franka_default_dof_pos = to_torch(
			[-0.4050, 0.2139, -0.2834, -2.3273,  0.1036,  2.5304,  1.5893,  0.0200, 0.0200],
			device=self.device,)
		self.franka_default_orn = to_torch(
			[[0.924, -0.383, 0., 0.]], device = self.device).repeat(self.cfg.num_envs,1)
		lower = gymapi.Vec3(-self.cfg.env_spacing, -self.cfg.env_spacing, 0.0)
		upper = gymapi.Vec3(*([self.cfg.env_spacing]*3))

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
		goal_opts = gymapi.AssetOptions()
		goal_opts.density = 0
		goal_opts.disable_gravity = True
		goal_opts.fix_base_link = True
		goal_asset = self.gym.load_asset(
			self.sim, asset_root, self.cfg.asset.assetFileNameSphere, goal_opts)

		franka_start_pose = gymapi.Transform()
		franka_start_pose.p = gymapi.Vec3(-0.5, -0.4, 0.0)
		franka_start_pose.r = gymapi.Quat(0.0, 0.0, np.sqrt(2)/2, np.sqrt(2)/2)

		# compute aggregate size
		num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
		num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
		num_block_bodies = self.gym.get_asset_rigid_body_count(block_asset)
		num_block_shapes = self.gym.get_asset_rigid_shape_count(block_asset)
		num_goal_bodies = self.gym.get_asset_rigid_body_count(goal_asset)
		num_goal_shapes = self.gym.get_asset_rigid_shape_count(goal_asset)
		max_agg_bodies = (
			num_franka_bodies + self.cfg.num_goals * (num_block_bodies+num_goal_bodies))
		max_agg_shapes = (
			num_franka_shapes + self.cfg.num_goals * (num_block_shapes+num_goal_shapes))

		self.cameras = []
		self.frankas = []
		self.default_block_states = []
		self.prop_start = []
		self.envs = []

		for i in range(self.cfg.num_envs):
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
				self.goal_handles = []
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
						 block_state_pose.r.w, ] + [0]*6)
				for j in range(self.cfg.num_goals):
					goal_state_pose = gymapi.Transform()
					goal_state_pose.p.x = xmin + j * 2 * self.cfg.block_size
					goal_state_pose.p.y = 0
					goal_state_pose.p.z=0.2
					goal_state_pose.r = gymapi.Quat(0, 0, 0, 1) 
					goal_state_pose = goal_state_pose
					handle = self.gym.create_actor(
						env_ptr, goal_asset, goal_state_pose, "goal{}".format(j), i+self.cfg.num_envs, 0, 0,)
					self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, self.colors[j])
					self.goal_handles.append(handle)
			if self.cfg.aggregate_mode > 0:
				self.gym.end_aggregate(env_ptr)
			self.envs.append(env_ptr)
			self.frankas.append(franka_actor)
		for j in range(self.cfg.num_cameras):
			# create camera
			camera_properties = gymapi.CameraProperties()
			camera_properties.width = 320
			camera_properties.height = 200
			h1 = self.gym.create_camera_sensor(self.envs[j], camera_properties)
			camera_position = gymapi.Vec3(0, -1, 0.3)
			camera_target = gymapi.Vec3(0, 0, 0)
			self.gym.set_camera_location(h1, self.envs[j], camera_position, camera_target)
			self.cameras.append(h1)
		# set control data
		self.hand_handle = self.gym.find_actor_rigid_body_handle(
			env_ptr, franka_actor, "panda_link7")
		self.lfinger_handle = self.gym.find_actor_rigid_body_handle(
			env_ptr, franka_actor, "panda_leftfinger")
		self.rfinger_handle = self.gym.find_actor_rigid_body_handle(
			env_ptr, franka_actor, "panda_rightfinger")
		self.default_block_states = to_torch(
			self.default_block_states, device=self.device, dtype=torch.float
		).view(self.cfg.num_envs, self.cfg.num_goals, 13)
		self.init_data()

	def init_data(self):
		"""Generate System Initial Data
		"""
		# general buffers 
		self.progress_buf = torch.zeros(self.cfg.num_envs, device=self.device, dtype=torch.long)
		self.reset_buf = torch.zeros(self.cfg.num_envs, device=self.device, dtype=torch.bool)
		self.success_step_buf = torch.zeros(self.cfg.num_envs, device=self.device, dtype=torch.long)

		hand = self.gym.find_actor_rigid_body_handle(
			self.envs[0], self.frankas[0], "panda_link7")
		lfinger = self.gym.find_actor_rigid_body_handle(
			self.envs[0], self.frankas[0], "panda_leftfinger")
		rfinger = self.gym.find_actor_rigid_body_handle(
			self.envs[0], self.frankas[0], "panda_rightfinger")

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
			*get_axis_params(0.04, grasp_pose_axis))
		self.franka_local_grasp_pos = to_torch(
			[
				franka_local_grasp_pose.p.x,
				franka_local_grasp_pose.p.y,
				franka_local_grasp_pose.p.z,
			],
			device=self.device,
		).repeat((self.cfg.num_envs, 1))
		self.franka_local_grasp_rot = to_torch(
			[
				franka_local_grasp_pose.r.x,
				franka_local_grasp_pose.r.y,
				franka_local_grasp_pose.r.z,
				franka_local_grasp_pose.r.w,
			],
			device=self.device,
		).repeat((self.cfg.num_envs, 1))

		self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat(
			(self.cfg.num_envs, 1)
		)
		self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat(
			(self.cfg.num_envs, 1)
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
			self.cfg.num_envs, -1, 13)
		# object observation
		self.block_states = self.root_state_tensor[:, 1:1+self.cfg.num_goals]
		# goal
		self.goal = self.root_state_tensor[:, 1+self.cfg.num_goals:1+self.cfg.num_goals*2, :3]
			
		# joint pos
		dof_state_tensor = self.gym.acquire_dof_state_tensor(
			self.sim)  
		self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
		self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.cfg.num_envs
		self.franka_dof_targets = torch.zeros(
			(self.cfg.num_envs, self.num_dofs), dtype=torch.float, device=self.device
		)
		self.franka_dof_state = self.dof_state.view(self.cfg.num_envs, -1, 2)[
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
			self.cfg.num_envs, -1, 13)  
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

	def reset(self):
		for _ in range(10):
			self.reset_buf[:] = True
			act = torch.zeros((self.cfg.num_envs, 4), device=self.device, dtype=torch.float)
			self.step(act, use_init_pos=True)
		return self.step(act)[0]

	def step(self, actions: torch.Tensor, use_init_pos=False):
		# apply actions
		self.pre_physics_step(actions, use_init_pos=use_init_pos)
		# step physics and render each frame
		for i in range(self.cfg.control_freq_inv):
			self.gym.simulate(self.sim)
		if self.device == "cpu":
			self.gym.fetch_results(self.sim, True)
		# compute observations, rewards, resets, ...
		return self.post_physics_step()

	def pre_physics_step(self, actions, use_init_pos=False):
		reset_idx = self.reset_buf.clone()
		done_env_num = reset_idx.sum()
		# reset goals
		self.goal[reset_idx] = self.torch_goal_space.sample((done_env_num,))

		# reset blocks
		if done_env_num > 0: 
			block_indices = self.global_indices[reset_idx, 1:].flatten()
			# set to default pos
			self.block_states[reset_idx] = self.default_block_states[reset_idx]
			in_hand = torch.rand((self.cfg.num_envs,),device=self.device) < self.cfg.inhand_rate
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

		# update state buffers
		self.progress_buf += 1

		# update obs, rew, done, info
		self.grip_pos = (self.franka_lfinger_pos+self.franka_rfinger_pos)/2 + self.finger_shift
		obs = torch.cat((
			(self.grip_pos-self.grip_pos_mean)/self.grip_pos_std, # mid finger
			(self.hand_vel-self.hand_vel_mean)/self.hand_vel_std, 
			(self.finger_width.unsqueeze(-1)-self.finger_width_mean)/self.finger_width_std, # robot
			self.block_states[..., 3:].view(self.cfg.num_envs, -1), # objects
			((self.block_states[...,:3]-self.goal_mean)/self.goal_std).view(self.cfg.num_envs, -1), # achieved goal NOTE make sure it is close to end
			((self.goal-self.goal_mean)/self.goal_std).view(self.cfg.num_envs, -1)
			),dim=-1)
		# rew
		rew = self.compute_reward(self.block_states[..., :3], self.goal, None, normed=False)
		# reset
		success_env = rew > self.cfg.success_bar
		self.success_step_buf[~success_env] = self.progress_buf[~success_env]
		self.reset_buf = ((self.progress_buf >= (self.cfg.max_steps - 1)) | (self.progress_buf > self.success_step_buf + self.cfg.extra_steps))
		done = self.reset_buf.clone().type(torch.float)
		# info
		info = torch.cat((
			success_env.type(torch.float).unsqueeze(-1),
			self.progress_buf.type(torch.float).unsqueeze(-1),
			((self.block_states[...,:3]-self.goal_mean)/self.goal_std).view(self.cfg.num_envs,3*self.cfg.num_goals).type(torch.float),
		), dim=-1)

		# debug viz
		if self.viewer and self.cfg.debug_viz:
			self.gym.clear_lines(self.viewer)

			for i in range(self.cfg.num_envs):
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
				# for j in range(self.cfg.num_goals):
				# 	sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * np.pi, 0, 0)
				# 	sphere_pose = gymapi.Transform(r=sphere_rot)
				# 	sphere_geom = gymutil.WireframeSphereGeometry(0.025, 12, 12, sphere_pose, 
				# 		color=(self.colors[j].x, self.colors[j].y, self.colors[j].z))
				# 	pos = gymapi.Transform()
				# 	pos.p.x, pos.p.y, pos.p.z = self.goal[i, j, 0], self.goal[i, j, 1], self.goal[i, j, 2] 
				# 	gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pos)
					# self.gym.add_lines(self.viewer, self.envs[i], 1, sphere_geom, [0,0,0.1])
				# draw goal space
				low = self.torch_goal_space.low[0]
				high = self.torch_goal_space.high[0]
				mean = (high+low)/2
				pos = gymapi.Transform()
				pos.p.x, pos.p.y, pos.p.z = mean[0], mean[1], mean[2] 
				box_geom = gymutil.WireframeBoxGeometry(high[0]-low[0], high[1]-low[1], high[2]-low[2], color=(0,0,1))
				gymutil.draw_lines(box_geom, self.gym, self.viewer, self.envs[i], pos)

		return obs, rew, done, info

	def render(self, mode='rgb_array'):
		"""Draw the frame to the viewer, and check for keyboard events."""
		if self.viewer and mode == 'human':
			# check for window closed
			if self.gym.query_viewer_has_closed(self.viewer):
				sys.exit()
			# check for keyboard events
			for evt in self.gym.query_viewer_action_events(self.viewer):
				if evt.action == "QUIT" and evt.value > 0:
					sys.exit()
				elif evt.action == "toggle_viewer_sync" and evt.value > 0:
					self.enable_viewer_sync = not self.enable_viewer_sync
			# fetch results
			if self.device != "cpu":
				self.gym.fetch_results(self.sim, True)
			# step graphics
			if self.enable_viewer_sync:
				self.gym.step_graphics(self.sim)
				self.gym.draw_viewer(self.viewer, self.sim, True)
				# Wait for dt to elapse in real time.
				# This synchronizes the physics simulation with the rendering rate.
				self.gym.sync_frame_time(self.sim)
			else:
				self.gym.poll_viewer_events(self.viewer)
		elif mode == 'rgb_array':
			if self.device != "cpu":
				self.gym.fetch_results(self.sim, True)
			self.gym.step_graphics(self.sim)
			self.gym.render_all_camera_sensors(self.sim)
			images = []
			for idx, handle in enumerate(self.cameras):
				image = self.gym.get_camera_image(self.sim, self.envs[idx], handle, gymapi.IMAGE_COLOR)
				images.append(image.reshape((image.shape[0], -1, 4)))
			return images

	def control_ik(self, dpose):
		# solve damped least squares
		j_eef_T = torch.transpose(self.j_eef, 1, 2)
		lmbda = torch.eye(6, device=self.device) * (self.cfg.damping ** 2)
		u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda)
				 @ dpose).view(self.cfg.num_envs, self.franka_hand_index)
		return u

	def orientation_error(self, desired, current):
		cc = quat_conjugate(current)
		q_r = quat_mul(desired, cc)
		return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

	def ezpolicy(self, obs):
		up_step=5
		reach_step=30
		grasp_step=33
		end_step=50
		pos = obs[..., :3]*self.grip_pos_std+self.grip_pos_mean
		obj = obs[..., 17:20].view(self.cfg.num_envs, self.cfg.num_goals, 3)*self.goal_std+self.goal_mean
		goal = obs[..., 20:23].view(self.cfg.num_envs, self.cfg.num_goals, 3)*self.goal_std+self.goal_mean
		action = torch.zeros((self.cfg.num_envs, 4), device=self.device, dtype=torch.float32)
		for env_id in range(self.cfg.num_envs):
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
					# action[env_id, :3] = (torch.tensor([-0.15,0,0.1],device=self.device) - pos_now)
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
			graphics_device_id = -1 if ((not cfg.enable_camera_sensors) and cfg.headless) else cfg.sim_device_id,
			sim_device = f'cuda:{cfg.sim_device_id}' if cfg.sim_device_id >= 0 else 'cpu',
			rl_device = f'cuda:{cfg.rl_device_id}' if cfg.rl_device_id >= 0 else 'cpu',
			# isaac 
			physics_engine = getattr(gymapi, cfg.physics_engine),
			# steps
			max_steps = cfg.base_steps*cfg.num_goals,
		)
		sim_params = gymapi.SimParams()
		if cfg.up_axis not in ["z", "y"]:
			msg = f"Invalid physics up-axis: {cfg.up_axis}"
			print(msg)
			raise ValueError(msg)
		if cfg["up_axis"] == "z":
			sim_params.up_axis = gymapi.UP_AXIS_Z
		else:
			sim_params.up_axis = gymapi.UP_AXIS_Y
		sim_params.dt = cfg.dt
		sim_params.num_client_threads = cfg.get("num_client_threads", 0)
		sim_params.use_gpu_pipeline = cfg.use_gpu_pipeline
		sim_params.substeps = cfg.get("substeps", 2)
		# assign gravity
		sim_params.gravity = gymapi.Vec3(*cfg["gravity"])
		# configure physics parameters
		if cfg.physics_engine == gymapi.SIM_PHYSX:
			if "physx" in cfg:
				for opt in cfg["physx"].keys():
					if opt == "contact_collection":
						setattr(sim_params.physx,opt,
							gymapi.ContactCollection(cfg["physx"][opt]),)
					else:
						setattr(sim_params.physx, opt, cfg["physx"][opt])
		else:
			if "flex" in cfg:
				for opt in cfg["flex"].keys():
					setattr(sim_params.flex, opt, cfg["flex"][opt])
		# return the configured params
		cfg.sim_params = sim_params
		return cfg

	def close(self):
		self.gym.destroy_viewer(self.viewer)
		self.gym.destroy_sim(self.sim)

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
	from PIL import Image
	import numpy as np
	'''
	run random policy
	'''
	env = gym.make('PandaPNP-v0', num_envs=4, headless=True, enable_camera_sensors=True)
	obs = env.reset()
	start = time.time()
	for _ in range(1):
		# act = torch.tensor([[1,-0.01,1,0],[1,-0.01,1,0]])
		# act = torch.rand((env.cfg.num_envs,4), device='cuda:0')*2-1
		# act[..., 0] += 0.5
		act = env.ezpolicy(obs)
		obs, rew, done, info = env.step(act)
		images = env.render(mode='rgb_array')
		Image.fromarray(images[0]).save('foo.png')

	print(time.time()-start)
	env.close()