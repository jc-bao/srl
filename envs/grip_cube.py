from socket import EAI_OVERFLOW
import numpy as np
import os
import sys
import time
import yaml
import gym
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
import torch
from attrdict import AttrDict
import pathlib
import math
# import pytorch_kinematics as pk


class gripCube(gym.Env):
	def __init__(self, cfg_file='Grip.yaml', **kwargs):
		# get config and setup base class
		self.cfg_path = pathlib.Path(__file__).parent.resolve()/'configs'
		with open(self.cfg_path/cfg_file) as config_file:
			try:
				cfg = AttrDict(yaml.load(config_file, Loader=yaml.SafeLoader))
			except yaml.YAMLError as exc:
				print(exc)
		cfg.update(**kwargs)  # overwrite params from args
		self.cfg = self.update_config(cfg)  # auto update sim params
		self.device = self.cfg.sim_device

		# setup ik solver
		# self.chain = pk.build_serial_chain_from_urdf(open("./isaac_assets/urdf/grip_description/robots/grip_panda.urdf").read(), "panda_hand").to(device = self.device)

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
			low=torch.tensor([-self.cfg.goal_space[0]/2, -self.cfg.goal_space[1]/2, self.cfg.block_size/2],device=self.device),
			high=torch.tensor([self.cfg.goal_space[0]/2, self.cfg.goal_space[1]/2, self.cfg.block_size/2+0.001], device=self.device))
		# robot space
		self.torch_robot_space = torch.distributions.uniform.Uniform(
			low=torch.tensor([-self.cfg.robot_gap/2, -self.cfg.goal_space[1]/1.5, self.cfg.block_size/2],
											 device=self.device),
			high=torch.tensor([self.cfg.robot_gap/2, self.cfg.goal_space[1]/1.5, self.cfg.block_size/2+self.cfg.goal_space[2]*1.5], device=self.device))
		# goal space
		if self.cfg.goal_space[2] > 0.01 and self.cfg.num_robots > 1 and self.cfg.num_goals > 1:
			print('[Env] Warn: multi robot, multi goal, goal height > 0.01')
		self.torch_goal_space = torch.distributions.uniform.Uniform(
			low=torch.tensor([-self.cfg.goal_space[0]/2, -self.cfg.goal_space[1]/2, self.cfg.block_size/2], device=self.device),
			high=torch.tensor([self.cfg.goal_space[0]/2, self.cfg.goal_space[1]/2, self.cfg.block_size/2+self.cfg.goal_space[2]], device=self.device))
		if self.cfg.norm_method == 'default':
			self.single_goal_mean = self.torch_goal_space.mean
			self.single_goal_std = self.torch_goal_space.stddev
			self.goal_mean = self.torch_goal_space.mean
			self.goal_std = self.torch_goal_space.stddev 
		elif self.cfg.norm_method == '0-1':
			self.single_goal_mean = self.torch_goal_space.mean
			self.single_goal_std = torch.tensor(np.array(self.cfg.goal_space)/2, dtype=torch.float, device=self.device) 
			self.goal_mean = torch.tensor([0,0,self.cfg.table_size[2]+self.cfg.goal_space[2]/2+self.cfg.block_size/2], device=self.device) 
			self.goal_std = torch.tensor([self.cfg.robot_gap/2+self.cfg.goal_space[0]/2,self.cfg.goal_space[1]/2,self.cfg.goal_space[2]/2], device=self.device)
		else: 
			raise NotImplementedError('norm_method not implemented')

		# indices
		self.global_indices = torch.arange(
			self.cfg.num_envs * (self.cfg.num_robots*2 + self.cfg.num_goals*2), dtype=torch.int32, device=self.device
		).view(self.cfg.num_envs, -1)

		# rotation metrix
		robot_pos_rot_mat = torch.tensor([
			[0,0,0,-1,0,0],
			[0,0,0,0,-1,0],
			[0,0,0,0,0,1],
			[-1,0,0,0,0,0],
			[0,-1,0,0,0,0],
			[0,0,1,0,0,0],
		], device=self.device)
		pos_rot_mat = torch.tensor([
			[-1,0,0],
			[0,-1,0],
			[0,0,1]
		], device=self.device)
		quat_rot_mat = torch.tensor([
			[0,-1,0,0],
			[1,0,0,0],
			[0,0,0,1],
			[0,0,1,0],
		], device=self.device)
		block_other_mat = torch.block_diag(*([quat_rot_mat]+[pos_rot_mat]*2))
		self.obs_rot_mat = torch.block_diag(*([robot_pos_rot_mat]*2+[torch.tensor([[0,1],[1,0]],device=self.device)]+[block_other_mat]*self.cfg.num_goals+[pos_rot_mat]*2*self.cfg.num_goals))

		self.reset()

	def set_viewer(self):
		self.enable_viewer_sync = True
		self.viewer = None
		# if running with a viewer, set up keyboard shortcuts and camera
		if self.cfg.headless == False:
			# subscribe to keyboard shortcuts
			camera_setting = gymapi.CameraProperties()
			self.viewer = self.gym.create_viewer(self.sim, camera_setting)
			cam_pos = gymapi.Vec3(1, -1, 1)
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
				cam_target = gymapi.Vec3(0.0, 0.0, 0.4)
			else:
				cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
				cam_target = gymapi.Vec3(10.0, 0.0, 15.0)
			self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

	def _create_ground_plane(self):
		plane_params = gymapi.PlaneParams()
		plane_params.distance = 0.0
		plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
		self.gym.add_ground(self.sim, plane_params)

	def _create_envs(self):
		# default values
		num_per_row = int(np.sqrt(self.cfg.num_envs))
		# colors
		self.colors = [gymapi.Vec3(*np.random.rand(3))
									 for _ in range(self.cfg.num_goals)]
		# finger shift
		self.finger_shift = to_torch(self.cfg.finger_shift, device=self.device)
		# joint pos
		lower = gymapi.Vec3(-self.cfg.env_spacing, -self.cfg.env_spacing, 0.0)
		upper = gymapi.Vec3(*([self.cfg.env_spacing]*3))

		asset_root = os.path.join(
			os.path.dirname(os.path.abspath(__file__)), self.cfg.asset.assetRoot)
		grip_asset_file = self.cfg.asset.assetFileNamegrip
		# load grip asset
		asset_options = gymapi.AssetOptions()
		asset_options.flip_visual_attachments = True
		asset_options.fix_base_link = False
		asset_options.collapse_fixed_joints = True
		asset_options.disable_gravity = True
		asset_options.thickness = 0.001
		asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
		asset_options.use_mesh_materials = True
		grip_asset = self.gym.load_asset(
			self.sim, asset_root, grip_asset_file, asset_options
		)
		grip_dof_stiffness = to_torch(
			[1.0e6, 1.0e6],
			dtype=torch.float,
			device=self.device,
		)
		grip_dof_damping = to_torch(
			[1.0e2, 1.0e2],
			dtype=torch.float,
			device=self.device,
		)
		self.num_grip_bodies = self.gym.get_asset_rigid_body_count(
			grip_asset)
		self.num_grip_dofs = self.gym.get_asset_dof_count(grip_asset)
		self.grip_link_dict = self.gym.get_asset_rigid_body_dict(grip_asset)
		print("num grip bodies: ", self.num_grip_bodies)
		print("num grip dofs: ", self.num_grip_dofs)

		# set grip dof properties
		grip_dof_props = self.gym.get_asset_dof_properties(grip_asset)
		self.grip_dof_lower_limits = []
		self.grip_dof_upper_limits = []
		for i in range(self.num_grip_dofs):
			grip_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
			if self.cfg.physics_engine == gymapi.SIM_PHYSX:
				grip_dof_props["stiffness"][i] = grip_dof_stiffness[i]
				grip_dof_props["damping"][i] = grip_dof_damping[i]
			else:
				grip_dof_props["stiffness"][i] = 7000.0
				grip_dof_props["damping"][i] = 50.0

			self.grip_dof_lower_limits.append(grip_dof_props["lower"][i])
			self.grip_dof_upper_limits.append(grip_dof_props["upper"][i])

		self.grip_dof_lower_limits = to_torch(
			self.grip_dof_lower_limits, device=self.device
		)
		self.grip_dof_upper_limits = to_torch(
			self.grip_dof_upper_limits, device=self.device
		)
		self.grip_dof_speed_scales = torch.ones_like(
			self.grip_dof_lower_limits)
		self.grip_dof_speed_scales[[0, 1]] = 0.1
		grip_dof_props["effort"][0] = 200
		grip_dof_props["effort"][1] = 200

		# create block assets
		box_opts = gymapi.AssetOptions()
		box_opts.density = 100
		box_opts.angular_damping = 100
		box_opts.linear_damping = 10
		box_opts.thickness = 0.005
		if self.cfg.lock_block_orn:
			block_asset = self.gym.load_asset(
				self.sim, asset_root, 'urdf/cube.urdf', box_opts)
		else:
			block_asset = self.gym.create_box(
				self.sim, self.cfg.block_length, self.cfg.block_size, self.cfg.block_size, box_opts)
		goal_opts = gymapi.AssetOptions()
		goal_opts.density = 0
		goal_opts.disable_gravity = True
		goal_opts.fix_base_link = True
		goal_asset = self.gym.load_asset(
			self.sim, asset_root, self.cfg.asset.assetFileNameSphere, goal_opts)
		# create table assets
		table_opts = gymapi.AssetOptions()
		table_opts.disable_gravity = True
		table_opts.fix_base_link = True
		table_opts.density = 400
		table_asset = self.gym.create_box(
			self.sim, self.cfg.table_size[0], self.cfg.table_size[1], self.cfg.table_size[2], table_opts)


		grip_start_poses = []
		self.grip_roots = []
		self.origin_shift = []
		for grip_id in range(self.cfg.num_robots):
			side = grip_id % 2 * 2 - 1
			x = self.cfg.robot_gap*(grip_id+0.5-self.cfg.num_robots*0.5)
			grip_start_pose = gymapi.Transform()
			pos = [x, side*self.cfg.robot_y, self.cfg.table_size[2]]
			grip_start_pose.p = gymapi.Vec3(*pos)
			pos[1] = 0
			self.origin_shift.append(pos)
			rot = [0.0, 0.0, np.sqrt(2)/2, -side*np.sqrt(2)/2]
			grip_start_pose.r = gymapi.Quat(*rot)
			grip_start_poses.append(grip_start_pose)
			self.grip_roots.append([*pos, *rot])
		self.origin_shift = to_torch(self.origin_shift, device=self.device)
		self.grip_roots = to_torch(self.grip_roots, device=self.device)
		self.default_grip_pos = self.origin_shift.unsqueeze(0).repeat(self.cfg.num_envs,1,1).clone()

		# compute aggregate size
		num_grip_bodies = self.gym.get_asset_rigid_body_count(grip_asset)
		num_grip_shapes = self.gym.get_asset_rigid_shape_count(grip_asset)
		num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
		num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
		num_block_bodies = self.gym.get_asset_rigid_body_count(block_asset)
		num_block_shapes = self.gym.get_asset_rigid_shape_count(block_asset)
		num_goal_bodies = self.gym.get_asset_rigid_body_count(goal_asset)
		num_goal_shapes = self.gym.get_asset_rigid_shape_count(goal_asset)
		max_agg_bodies = (
			(num_grip_bodies+num_table_bodies)*self.cfg.num_robots+ self.cfg.num_goals * (num_block_bodies+num_goal_bodies))
		max_agg_shapes = (
			(num_grip_shapes+num_table_shapes)*self.cfg.num_robots + self.cfg.num_robots + self.cfg.num_goals * (num_block_shapes+num_goal_shapes))

		self.cameras = []
		self.grips = []
		self.default_block_states = []
		self.prop_start = []
		self.envs = []

		for i in range(self.cfg.num_envs):
			# create env instance
			env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
			if self.cfg.aggregate_mode >= 3:
				self.gym.begin_aggregate(
					env_ptr, max_agg_bodies, max_agg_shapes, True)
			grip_actors = []
			for grip_id in range(self.cfg.num_robots):
				grip_pos = grip_start_poses[grip_id]
				# Key: create Panda
				grip_actor = self.gym.create_actor(
					env_ptr, grip_asset, grip_pos, f"grip{grip_id}", i, 0, i
				)
				self.gym.set_actor_dof_properties(
					env_ptr, grip_actor, grip_dof_props)
				grip_actors.append(grip_actor)
			for grip_id in range(self.cfg.num_robots):
				grip_pos = grip_start_poses[grip_id]
				# create table
				table_state_pose = gymapi.Transform()
				table_state_pose.p = gymapi.Vec3(grip_pos.p.x, 0, self.cfg.table_size[2]/2)
				table_state_pose.r = gymapi.Quat(0, 0, 0, 1)
				table_actor = self.gym.create_actor(
						env_ptr, table_asset, table_state_pose, "table{}".format(grip_id), i, 0, 0,)
				prop = gymapi.RigidShapeProperties()
				prop.friction = self.cfg.friction 
				self.gym.set_actor_rigid_shape_properties(
					env_ptr, table_actor, [prop])
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
					block_state_pose.p.z = self.cfg.table_size[2]
					block_state_pose.r = gymapi.Quat(0, 0, 0, 1)
					handle = self.gym.create_actor(
						env_ptr, block_asset, block_state_pose, "block{}".format(j), i, 0, 0,)
					self.block_handles.append(handle)
					self.gym.set_rigid_body_color(
						env_ptr, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.colors[j])
					self.default_block_states.append(
						[block_state_pose.p.x, block_state_pose.p.y, block_state_pose.p.z,
						 block_state_pose.r.x, block_state_pose.r.y, block_state_pose.r.z,
						 block_state_pose.r.w, ] + [0]*6)
				for j in range(self.cfg.num_goals):
					goal_state_pose = gymapi.Transform()
					goal_state_pose.p.x = xmin + j * 2 * self.cfg.block_size
					goal_state_pose.p.y = 0
					goal_state_pose.p.z = 0.2
					goal_state_pose.r = gymapi.Quat(0, 0, 0, 1)
					goal_state_pose = goal_state_pose
					handle = self.gym.create_actor(
						env_ptr, goal_asset, goal_state_pose, "goal{}".format(j), i+self.cfg.num_envs, 0, 0,)
					self.gym.set_rigid_body_color(
						env_ptr, handle, 0, gymapi.MESH_VISUAL, self.colors[j])
					self.goal_handles.append(handle)
			if self.cfg.aggregate_mode > 0:
				self.gym.end_aggregate(env_ptr)
			self.envs.append(env_ptr)
			self.grips.append(grip_actors)
		for j in range(self.cfg.num_cameras):
			# create camera
			camera_properties = gymapi.CameraProperties()
			camera_properties.width = 320
			camera_properties.height = 200
			h1 = self.gym.create_camera_sensor(self.envs[j], camera_properties)
			camera_position = gymapi.Vec3(1, -1, 1)
			camera_target = gymapi.Vec3(0, 0, 0)
			self.gym.set_camera_location(
				h1, self.envs[j], camera_position, camera_target)
			self.cameras.append(h1)
		# set control data
		self.hand_handles = [self.gym.find_actor_rigid_body_handle(
			env_ptr, r, "panda_hand") for r in grip_actors]
		self.lfinger_handles = [self.gym.find_actor_rigid_body_handle(
			env_ptr, r, "panda_leftfinger") for r in grip_actors]
		self.rfinger_handles = [self.gym.find_actor_rigid_body_handle(
			env_ptr, r, "panda_rightfinger") for r in grip_actors]
		self.default_block_states = to_torch(
			self.default_block_states, device=self.device, dtype=torch.float
		).view(self.cfg.num_envs, self.cfg.num_goals, 13)
		self.init_data()

	def init_data(self):
		"""Generate System Initial Data
		"""
		# general buffers
		self.progress_buf = torch.zeros(
			self.cfg.num_envs, device=self.device, dtype=torch.long)
		self.reset_buf = torch.zeros(
			self.cfg.num_envs, device=self.device, dtype=torch.bool)
		self.success_step_buf = torch.zeros(
			self.cfg.num_envs, device=self.device, dtype=torch.long)

		# joint pos
		dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
		self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
		self.num_dofs = self.gym.get_sim_dof_count(self.sim) // (self.cfg.num_envs*self.cfg.num_robots)
		self.grip_dof_targets = torch.zeros(
			(self.cfg.num_envs, self.cfg.num_robots, self.num_dofs), dtype=torch.float, device=self.device
		)
		self.grip_dof_states = self.dof_state.view(self.cfg.num_envs, self.cfg.num_robots, -1, 2)[
			:,:, : self.num_grip_dofs
		]
		self.grip_dof_poses = self.grip_dof_states[..., 0]
		self.grip_dof_vels = self.grip_dof_states[..., 1]
		self.finger_widths = self.grip_dof_poses[:,:,0]

		self.finger_width_mean = 0.02
		self.finger_width_std = 0.04 / 12**0.5

		# root
		actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(
			self.sim)
		self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
			self.cfg.num_envs, -1, 13)
		# object observation
		self.block_states = self.root_state_tensor[:, self.cfg.num_robots*2:self.cfg.num_robots*2+self.cfg.num_goals]
		# goal
		self.init_ag = torch.zeros_like(self.block_states[..., :3], device=self.device, dtype=torch.float)
		self.init_ag_normed = torch.zeros_like(self.init_ag, device=self.device, dtype=torch.float) 
		self.last_step_ag = torch.zeros_like(self.block_states[..., :3], device=self.device, dtype=torch.float)
		self.ag_unmoved_steps = torch.zeros((self.cfg.num_envs, self.cfg.num_goals,), device=self.device, dtype=torch.float)
		self.goal = self.root_state_tensor[:, self.cfg.num_robots*2 +
																			 self.cfg.num_goals:self.cfg.num_robots*2+self.cfg.num_goals*2, :3]
		self.goal_workspace = torch.zeros((self.cfg.num_envs, self.cfg.num_goals), device=self.device, dtype=torch.long)
		self.block_workspace = torch.zeros((self.cfg.num_envs, self.cfg.num_goals), device=self.device, dtype=torch.long)
		self.num_os_goal = torch.zeros((self.cfg.num_envs,), device=self.device, dtype=torch.long)
		# table
		self.table_states = self.root_state_tensor[:, self.cfg.num_robots:self.cfg.num_robots*2]

		# object pos
		rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
		self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
			self.cfg.num_envs, -1, 13)
		# robot observation
		self.hand_vel_mean = 0
		self.hand_vel_std = 2*self.cfg.max_vel / 12**0.5
		self.num_bodies = self.rigid_body_states.shape[1]
		# store for obs 
		self.hand_pos = []
		self.hand_vel = []
		self.hand_rot = []
		self.grip_lfinger_poses = []
		self.grip_rfinger_poses = []
		self.grip_lfinger_rots = []
		self.grip_rfinger_rots = []
		for i in range(self.cfg.num_robots):
			# NOTE do not use self.lfinger_handles to indice as it will copy the tensor
			self.hand_pos.append(self.rigid_body_states[:, self.hand_handles[i]][..., :3])
			self.hand_rot.append(self.rigid_body_states[:, self.hand_handles[i]][..., 3:7])
			self.hand_vel.append(self.rigid_body_states[:, self.hand_handles[i]][..., 7:10])
			self.grip_lfinger_poses.append(self.rigid_body_states[:,
																											self.lfinger_handles[i]][..., 0:3])
			self.grip_rfinger_poses.append(self.rigid_body_states[:,
																											self.rfinger_handles[i]][..., 0:3])
			self.grip_lfinger_rots.append(self.rigid_body_states[:,
																											self.lfinger_handles[i]][..., 3:7])
			self.grip_rfinger_rots.append(self.rigid_body_states[:,
																											self.rfinger_handles[i]][..., 3:7])
		self.grip_pos = (torch.stack(self.grip_lfinger_poses,dim=1) +
										 torch.stack(self.grip_rfinger_poses,dim=1))/2 + self.finger_shift
		self.grip_states = self.root_state_tensor[:, :self.cfg.num_robots]
		self.hand_pos_tensor = torch.stack(self.hand_pos, dim=1)
		self.target_hand_pos = self.hand_pos_tensor.clone()

	def compute_reward(self, ag, dg, info, normed=True):
		ag = ag.view(-1, self.cfg.num_goals, 3)
		dg = dg.view(-1, self.cfg.num_goals, 3)
		if normed:
			ag = ag*self.goal_std + self.goal_mean
			dg = dg*self.goal_std + self.goal_mean
		if self.cfg.reward_type == 'sparse':
			return -torch.mean((torch.norm(ag-dg, dim=-1) > self.cfg.err).type(torch.float32), dim=-1)
		elif self.cfg.reward_type == 'sparse+':
			return torch.mean((torch.norm(ag-dg, dim=-1) < self.cfg.err).type(torch.float32), dim=-1)
		elif self.cfg.reward_type == 'dense':
			distances = torch.norm(ag-dg, dim=-1)
			distance_rew = 0.5*(1-torch.mean(distances,dim=-1)*5/self.cfg.num_robots)/(self.cfg.num_goals)
			reach_rew = 0.5*torch.mean((distances<self.cfg.err).float(), dim=-1)
			return distance_rew+reach_rew 
		elif self.cfg.reward_type == 'dense+':
			# distance to obj
			dist2obj = torch.norm(ag-info.grip_pos, dim=-1) # TODO multi robot
			dist2obj_rew = 0.1*(1-torch.mean(dist2obj,dim=-1)*5)/self.cfg.num_goals
			reach_obj_rew = 0.1*torch.mean((dist2obj<self.cfg.err).float(), dim=-1)
			reach_obj_rew = 0.1*torch.mean((dist2obj<self.cfg.err/2).float(), dim=-1)
			reach_obj_rew = 0.1*torch.mean((dist2obj<self.cfg.err/4).float(), dim=-1)
			reach_obj_rew = 0.1*torch.mean((dist2obj<self.cfg.err/8).float(), dim=-1)
			# distance to goal
			dist2g = torch.norm(ag-dg, dim=-1)
			dist2g_rew = 0.25*(1-torch.mean(dist2g,dim=-1)*5)/self.cfg.num_goals
			reach_g_rew = 0.25*torch.mean((dist2g<self.cfg.err).float(), dim=-1)
			return dist2obj_rew+reach_obj_rew+dist2g_rew+reach_g_rew


	def reset(self, config={}):
		# change params
		'''manualy set attribute to certain value'''
		for k, v in config.items():
			if k in self.cfg:
				v_old = self.cfg[k]
				self.cfg[k] = v
				print(f'[Curriculum] change {k} from {v_old} to {v}')
				# TODO move this to general update
			else:
				print(f'[Curriculum] config has no attribute {k}')
		# step first to init params
		act = torch.zeros((self.cfg.num_envs, 4*self.cfg.num_robots), device=self.device, dtype=torch.float)
		obs, _, _, _ = self.step(act) # TODO try to remove this
		self.reset_buf[:] = True
		obs, rew, done, info = self.step(act)
		self.progress_buf[:] = 0  # NOTE: make sure step start from 0
		self.default_grip_pos = self.grip_pos.clone()
		return obs, rew, done, info

	def step(self, actions: torch.Tensor):
		# apply actions
		reset_idx = self.reset_buf.clone()
		done_env_num = reset_idx.sum()
		# reset goals
		while True and done_env_num > 0:
			if self.cfg.goal_sample_mode == 'uniform':
				extra_goal_ws = torch.randint(self.cfg.num_robots,size=(done_env_num.item(),self.cfg.num_goals), device=self.device).repeat(self.cfg.extra_goal_sample,1,1)
			elif self.cfg.goal_sample_mode == 'bernoulli': # TODO extend to multi arm scenario
				extra_goal_ws = torch.randint(self.cfg.num_robots,size=(done_env_num.item(),1), device=self.device).repeat(self.cfg.extra_goal_sample,1,self.cfg.num_goals)
				goal_ws_shift = torch.bernoulli(torch.ones((done_env_num.item(),self.cfg.num_goals-1), device=self.device, dtype=torch.float)*self.cfg.goal_os_rate).long()
				self.num_os_goal[reset_idx] = goal_ws_shift.sum(dim=-1)
				extra_goal_ws_shift = goal_ws_shift.repeat(self.cfg.extra_goal_sample,1,1)
				extra_goal_ws[...,1:] += extra_goal_ws_shift
				extra_goal_ws %= self.cfg.num_goals
			sampled_goal = self.torch_goal_space.sample((self.cfg.extra_goal_sample, done_env_num,self.cfg.num_goals))
			goal_dift = torch.tensor([0,0,self.cfg.block_size/2], device=self.device)
			sampled_goal = (sampled_goal - goal_dift)*self.cfg.goal_scale + goal_dift
			extra_goals = sampled_goal + \
				self.origin_shift[extra_goal_ws.flatten()].view(self.cfg.extra_goal_sample, done_env_num, self.cfg.num_goals, 3)
			goal_dist = torch.abs(extra_goals.unsqueeze(-3) - extra_goals.unsqueeze(-2))
			satisfied_idx = ((goal_dist[...,0] > self.cfg.block_length*1.2) | \
				(goal_dist[..., 1] > self.cfg.block_size*2) | \
						torch.eye(self.cfg.num_goals, device=self.device, dtype=torch.bool)).all(dim=-1).all(dim=-1)
			if satisfied_idx.sum() >= done_env_num:
				self.goal[reset_idx] = extra_goals[satisfied_idx][:done_env_num]
				self.goal_workspace[reset_idx] = extra_goal_ws[satisfied_idx][:done_env_num]
				break
		multi_goal_in_same_ws = torch.zeros((self.cfg.num_envs,), device=self.device, dtype=torch.bool)
		for i in range(self.cfg.num_robots):
			multi_goal_in_same_ws |= ((self.goal_workspace==i).sum(dim=-1) > 1)
		# reset tables
		if self.cfg.num_robots == 2:
			new_gap = self.cfg.table_size[0] + self.cfg.table_gap + torch.rand((done_env_num), device=self.device) * self.cfg.rand_table_gap
			self.table_states[reset_idx,0,0] = -new_gap/2
			self.table_states[reset_idx,1,0] = new_gap/2
		# reset blocks
		block_indices = torch.tensor([], device=self.device)
		if done_env_num > 0:
			block_indices = self.global_indices[reset_idx, self.cfg.num_robots:].flatten()
			# set to default pos
			self.block_states[reset_idx] = self.default_block_states[reset_idx]
			in_hand = torch.rand((self.cfg.num_envs,),
													 device=self.device) < self.cfg.inhand_rate
			self.inhand_idx = reset_idx & in_hand
			while True and done_env_num > 0:
				if self.cfg.obj_sample_mode == 'uniform':
					extra_block_ws = torch.randint(self.cfg.num_robots,size=(done_env_num.item(),self.cfg.num_goals), device=self.device).repeat(self.cfg.extra_goal_sample,1,1)
				elif self.cfg.obj_sample_mode == 'bernoulli': # TODO extend to multi arm scenario
					tiled_goal_ws = self.goal_workspace[reset_idx].repeat(self.cfg.extra_goal_sample,1,1)
					extra_block_ws = tiled_goal_ws + torch.bernoulli(torch.ones(tiled_goal_ws.shape[1:], device=self.device, dtype=torch.float)*self.cfg.os_rate).long().repeat(self.cfg.extra_goal_sample,1,1)
					extra_block_ws %= self.cfg.num_robots
				elif self.cfg.obj_sample_mode == 'task_distri':
					rand_number = torch.rand((done_env_num,), device=self.device)
					block_ws = torch.zeros((done_env_num, self.cfg.num_goals), device=self.device, dtype=torch.long)
					now_prob = self.cfg.task_distri[0] 
					for i in range(1, self.cfg.num_goals+1):
						block_ws[now_prob<=rand_number<now_prob+self.cfg.task_distri[i], :i] = 1
						now_prob += self.cfg.task_distri[i]
					block_ws += self.goal_workspace[reset_idx] 
					extra_block_ws = block_ws.repeat(self.cfg.extra_goal_sample,1,1)
					extra_block_ws %= self.cfg.num_robots
				else:
					raise NotImplementedError
				# TODO fix this
				sampled_ag = self.torch_block_space.sample((self.cfg.extra_goal_sample, done_env_num,self.cfg.num_goals))
				goal_dift = torch.tensor([0,0,self.cfg.block_size/2], device=self.device)
				sampled_ag = (sampled_ag - goal_dift)*self.cfg.goal_scale + goal_dift
				extra_ags = sampled_ag + \
					self.origin_shift[extra_block_ws.flatten()].view(self.cfg.extra_goal_sample, done_env_num, self.cfg.num_goals, 3)
				ag_dist = torch.abs(extra_ags.unsqueeze(-3) - extra_ags.unsqueeze(-2))
				satisfied_idx = ((ag_dist[...,0] > self.cfg.block_length*1.2) | \
					(ag_dist[..., 1] > self.cfg.block_size*2) | \
							torch.eye(self.cfg.num_goals, device=self.device, dtype=torch.bool)).all(dim=-1).all(dim=-1)
				if satisfied_idx.sum() >= done_env_num:
					self.init_ag[reset_idx] = extra_ags[satisfied_idx][:done_env_num]
					self.block_workspace[reset_idx] = extra_block_ws[satisfied_idx][:done_env_num]
					break	
			self.num_handovers = (self.block_workspace != self.goal_workspace).sum(dim=-1)
			self.last_step_ag[reset_idx] = self.init_ag[reset_idx]
			self.ag_unmoved_steps[reset_idx] = 0
			if self.inhand_idx.any():
				# choosed_block = torch.randint(self.cfg.num_goals, (1,), device=self.device)[0]
				# NOTE can only choose block 0 in hand now TODO fix it
				choosed_block = 0 
				choosed_robot = torch.randint(high=self.cfg.num_robots,size=(self.inhand_idx.sum().item(),))
				self.init_ag[self.inhand_idx, choosed_block] = self.default_grip_pos[self.inhand_idx, choosed_robot] + \
					(torch.rand_like(self.default_grip_pos[self.inhand_idx, choosed_robot], device=self.device) - 0.5) * to_torch([self.cfg.block_length*0.7, 0., 0.], device=self.device)
				if self.cfg.num_goals > 1 and self.cfg.num_robots > 1 and torch.rand(1)[0] < 0.5:
					choosed_block = (choosed_block+1)%self.cfg.num_goals 
					choosed_robot = (choosed_robot+1)%self.cfg.num_robots
					self.init_ag[self.inhand_idx, choosed_block] = self.default_grip_pos[self.inhand_idx, choosed_robot] + \
						(torch.rand_like(self.default_grip_pos[self.inhand_idx, choosed_robot], device=self.device) - 0.5) * to_torch([self.cfg.block_length*0.7, 0., 0.], device=self.device)
			self.block_states[reset_idx,:,:3] = self.init_ag[reset_idx]
			self.init_ag_normed[reset_idx] = ((self.init_ag[reset_idx]-self.goal_mean)/self.goal_std)
			# change some goal to the ground
			ground_goal_idx = reset_idx & ((torch.rand((self.cfg.num_envs,),device=self.device) < self.cfg.goal_ground_rate) | multi_goal_in_same_ws | (self.num_handovers > 0))
			self.goal[ground_goal_idx, :, -1] = self.cfg.table_size[2]+self.cfg.block_size/2
		# reset state buf
		self.reset_buf[reset_idx] = 0
		self.progress_buf[reset_idx] = 0
		self.success_step_buf[reset_idx] = 0
		# set action here
		self.actions = torch.clip(
			actions.clone().view(self.cfg.num_envs,self.cfg.num_robots,self.cfg.per_action_dim).to(self.device)+self.cfg.action_shift, 
			-self.cfg.clip_actions, self.cfg.clip_actions)
		delta_pos = self.actions[..., :3] * self.cfg.dt * self.cfg.control_freq_inv * self.cfg.max_vel / self.cfg.control_freq_inv
		filtered_pos_target = self.hand_pos_tensor.clone()
		# step physics and render each frame
		for i in range(self.cfg.control_freq_inv):
			# setup control params
			filtered_pos_target += delta_pos 
			if self.cfg.bound_robot:
				filtered_pos_target = torch.clip(filtered_pos_target, self.torch_robot_space.low,
														self.torch_robot_space.high)
			self.grip_states[...,:3] = filtered_pos_target
			# grip
			self.grip_dof_targets += (self.actions[..., [3]] * self.cfg.dt * self.cfg.max_grip_vel).repeat(1, 1, 2)
			# limit
			self.grip_dof_targets[..., :self.num_grip_dofs] = tensor_clamp(
				self.grip_dof_targets[...,
																: self.num_grip_dofs], self.grip_dof_lower_limits, self.grip_dof_upper_limits)
			# Deploy actions
			# set action
			act_indices = self.global_indices[~reset_idx, :self.cfg.num_robots].flatten()
			self.gym.set_dof_position_target_tensor_indexed(
				self.sim, 
				gymtorch.unwrap_tensor(self.grip_dof_targets),
				gymtorch.unwrap_tensor(act_indices),
				act_indices.shape[0])
			# change to hand or random pos
			self.gym.set_actor_root_state_tensor_indexed(
				self.sim,
				gymtorch.unwrap_tensor(self.root_state_tensor),
				gymtorch.unwrap_tensor(block_indices),
				len(block_indices),)
			block_indices = torch.tensor([], device=self.device, dtype=torch.int32)
			# simulate
			self.gym.simulate(self.sim)
			# update state data
			self.gym.refresh_actor_root_state_tensor(self.sim)
			self.gym.refresh_dof_state_tensor(self.sim)
			self.gym.refresh_rigid_body_state_tensor(self.sim)
			self.hand_pos_tensor = torch.stack(self.hand_pos, dim=1)
			self.target_hand_pos[reset_idx] = self.hand_pos_tensor[reset_idx] 
		if not self.cfg.headless:
			self.render(mode='human')
		if self.device == "cpu":
			self.gym.fetch_results(self.sim, True)
		# compute observations, rewards, reset, ...
		# update state buffer
		self.progress_buf += 1

		# update obs, rew, done, info
		self.grip_pos = (torch.stack(self.grip_lfinger_poses,dim=1) +
										 torch.stack(self.grip_rfinger_poses,dim=1))/2 + self.finger_shift
		grip_pos_normed = (self.grip_pos-self.goal_mean)/self.goal_std
		hand_vel_normed = (torch.stack(self.hand_vel,dim=1)-self.hand_vel_mean)/self.hand_vel_std
		finger_widths_normed = (self.finger_widths.unsqueeze(-1)-self.finger_width_mean) / self.finger_width_std
		block_pos_normed = (self.block_states[..., :3]-self.goal_mean) / self.goal_std # CHECK multi robot
		goal_normed = (self.goal-self.goal_mean)/self.goal_std
		obs = torch.cat((
			grip_pos_normed.view(self.cfg.num_envs, self.cfg.num_robots*3),  # mid finger
			hand_vel_normed.view(self.cfg.num_envs, self.cfg.num_robots*3),
			finger_widths_normed.view(self.cfg.num_envs, self.cfg.num_robots),  # robot
			self.block_states[..., 3:].reshape(self.cfg.num_envs, -1),  # objects
			# achieved goal NOTE make sure it is close to end
			block_pos_normed.view(self.cfg.num_envs, self.cfg.num_goals*3),
			goal_normed.view(self.cfg.num_envs, self.cfg.num_goals*3),
		), dim=-1)

		# rew
		rew = self.compute_reward(
			self.block_states[..., :3], self.goal, AttrDict(grip_pos=self.grip_pos), normed=False)
		# reset
		ag_moved_dist = torch.norm(self.block_states[...,:3]-self.last_step_ag, dim=-1)
		reached_ag = torch.norm(self.block_states[..., :3]-self.goal, dim=-1) < self.cfg.err
		unmoved_ag = ag_moved_dist < self.cfg.ag_moved_threshold
		self.ag_unmoved_steps += unmoved_ag.float()
		self.ag_unmoved_steps[~unmoved_ag | reached_ag] = 0
		early_termin = ((self.progress_buf >= self.cfg.early_termin_step) & \
			(
				# not touch all the object
				torch.all(torch.max(self.init_ag - self.block_states[..., :3], dim=-1)[0] < self.cfg.early_termin_bar, dim=-1) |
				# all ag long time unmoved
				torch.all(self.ag_unmoved_steps > self.cfg.max_ag_unmoved_steps, dim=-1) |
				# hit the ground
				torch.any(self.grip_pos[..., 2] < (self.cfg.block_size/4+self.cfg.table_size[2]), dim=-1) |
				# block droped
				torch.any(self.block_states[..., 2].view(self.cfg.num_envs, self.cfg.num_goals) < self.cfg.table_size[2],dim=-1)
			))
		# print(torch.any(torch.abs(self.init_ag - self.block_states[..., :3])<self.cfg.early_termin_bar, dim=-1))
		success_env = rew > self.cfg.success_bar
		self.success_step_buf[~success_env] = self.progress_buf[~success_env]
		if self.cfg.auto_reset:
			self.reset_buf = ((self.progress_buf >= (self.cfg.max_steps)) |
												(self.progress_buf >= self.success_step_buf + self.cfg.extra_steps) |
												early_termin)
		else:
			self.reset_buf[:] = False
		done = self.reset_buf.clone().type(torch.float)
		# info
		self.ag_normed = ((self.block_states[..., :3]-self.goal_mean)/self.goal_std)
		info = torch.cat((
			success_env.type(torch.float).unsqueeze(-1),
			self.progress_buf.type(torch.float).unsqueeze(-1),
			early_termin.unsqueeze(-1),
			# traj_idx, traj_len, tleft
			torch.empty((self.cfg.num_envs, 3), device=self.device, dtype=torch.float),
			# ag
			self.ag_normed.view(
				self.cfg.num_envs, 3*self.cfg.num_goals).type(torch.float),
			# grip pos
			self.grip_pos.view(self.cfg.num_envs, -1), 
			# ag reached state 0, 1
			reached_ag.float(),
			# ag unmoved steps
			self.ag_unmoved_steps,
		), dim=-1)
		self.last_step_ag = self.block_states[..., :3].clone()

		# debug viz
		if self.viewer and self.cfg.debug_viz:
			self.gym.clear_lines(self.viewer)

			for i in range(self.cfg.num_envs):
				for j in range(self.cfg.num_robots):
					# draw finger mid
					finger_mid = (
						self.grip_lfinger_poses[j][i] + self.grip_rfinger_poses[j][i])/2 + self.finger_shift
					px = ((finger_mid + quat_apply(self.grip_lfinger_rots[j][i],
																				to_torch([1, 0, 0], device=self.device) * 0.2,)).cpu().numpy())
					py = ((finger_mid + quat_apply(self.grip_lfinger_rots[j][i],
																				to_torch([0, 1, 0], device=self.device) * 0.2,)).cpu().numpy())
					pz = ((finger_mid + quat_apply(self.grip_lfinger_rots[j][i],
																				to_torch([0, 0, 1], device=self.device) * 0.2,)).cpu().numpy())
					p0 = finger_mid.cpu().numpy()
					self.gym.add_lines(self.viewer, self.envs[i], 1,
														[p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0],)
					self.gym.add_lines(self.viewer, self.envs[i], 1,
														[p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0],)
					self.gym.add_lines(self.viewer, self.envs[i], 1,
														[p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1],)
					# draw goal space
					low = self.torch_goal_space.low
					high = self.torch_goal_space.high
					mean = (high+low)/2
					pos = gymapi.Transform()
					pos.p.x, pos.p.y, pos.p.z = mean[0]+self.origin_shift[j,0], mean[1]+self.origin_shift[j,1], mean[2]+self.origin_shift[j,2]
					box_geom = gymutil.WireframeBoxGeometry(
						high[0]-low[0], high[1]-low[1], high[2]-low[2], color=(0, 0, 1))
					gymutil.draw_lines(box_geom, self.gym, self.viewer, self.envs[i], pos)
					# draw robot space
					low = self.torch_robot_space.low
					high = self.torch_robot_space.high
					mean = (high+low)/2
					pos = gymapi.Transform()
					pos.p.x, pos.p.y, pos.p.z = mean[0]+self.origin_shift[j,0], mean[1]+self.origin_shift[j,1], mean[2]+self.origin_shift[j,2]
					box_geom = gymutil.WireframeBoxGeometry(
						high[0]-low[0], high[1]-low[1], high[2]-low[2], color=(1, 0, 1))
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
				image = self.gym.get_camera_image(
					self.sim, self.envs[idx], handle, gymapi.IMAGE_COLOR)
				images.append(image.reshape((image.shape[0], -1, 4)))
			return images

	def update_config(self, cfg):
		cfg.update(enable_camera_sensors=cfg.num_cameras > 0)
		cfg.update(
			# dim
			action_dim=cfg.per_action_dim * cfg.num_robots,
			shared_dim=cfg.per_shared_dim * cfg.num_robots, 
			state_dim=cfg.per_shared_dim * cfg.num_robots + cfg.per_seperate_dim * \
			cfg.num_goals + cfg.per_goal_dim*cfg.num_goals,
			info_dim=cfg.info_dim + cfg.num_goals*cfg.per_goal_dim + cfg.num_robots*3 + cfg.num_goals*2,
			seperate_dim=cfg.per_seperate_dim * cfg.num_goals,
			goal_dim=cfg.per_goal_dim * cfg.num_goals,
			# device
			# when not need render, close graph device, else share with sim
			graphics_device_id=-1 \
				if ((not cfg.enable_camera_sensors)
						and cfg.headless) else cfg.sim_device_id,
			sim_device=torch.device(f'cuda:{cfg.sim_device_id}' if cfg.sim_device_id >= 0 else 'cpu'),
			rl_device=torch.device(f'cuda:{cfg.rl_device_id}' if cfg.rl_device_id >= 0 else 'cpu'),
			# isaac
			physics_engine=getattr(gymapi, cfg.physics_engine),
			# steps
			max_steps=cfg.base_steps*cfg.num_goals*cfg.num_robots,
			# judge for success
			success_bar={'sparse':-0.01, 'sparse+':0.95, 'dense': 0.94, 'dense+': 0.95}[cfg.reward_type],
			# block size
			block_length=cfg.block_size if cfg.num_robots <1.5 else cfg.block_size*5,
			# goal distribition
			task_distri = [math.factorial(cfg.num_goals)/(math.factorial(cfg.num_goals-m)*math.factorial(m))*(1-cfg.os_rate)**m*cfg.os_rate**(cfg.num_goals-m) for m in range(cfg.num_goals+1)] if cfg.task_distri is None else cfg.task_distri,
		)
		# robot control
		cfg.action_shift=torch.tensor(cfg.action_shift,device=cfg.sim_device)
		# table size
		cfg.table_size = [cfg.robot_gap-cfg.table_gap, cfg.table_size[1], cfg.table_size[2]]
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
						setattr(sim_params.physx, opt,
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

	def obs_parser(self, obs, name = None):
		assert obs.shape[-1] == self.cfg.shared_dim+self.cfg.seperate_dim+self.cfg.goal_dim, f'obs shape error: {obs.shape}' 
		if name is None:
			return AttrDict(
				shared = obs[..., :self.cfg.shared_dim],
				seperate = obs[..., self.cfg.shared_dim:self.cfg.shared_dim+self.cfg.seperate_dim],
				ag = obs[..., self.cfg.shared_dim+self.cfg.seperate_dim-self.cfg.goal_dim:self.cfg.shared_dim+self.cfg.seperate_dim], 
				g = obs[..., self.cfg.shared_dim+self.cfg.seperate_dim:]
			)
		elif name == 'shared':
			return obs[..., :self.cfg.shared_dim]
		elif name == 'seperate':
			return obs[..., self.cfg.shared_dim:self.cfg.shared_dim+self.cfg.seperate_dim]
		elif name == 'ag':
			return obs[..., self.cfg.shared_dim+self.cfg.seperate_dim-self.cfg.goal_dim:self.cfg.shared_dim+self.cfg.seperate_dim]
		elif name == 'g':
			return obs[..., self.cfg.shared_dim+self.cfg.seperate_dim:] 
		else:
			raise NotImplementedError
	
	def obs_mirror(self, obs):
		return obs * self.obs_rot_mat
	
	def obs_updater(self, old_obs, new_obs:AttrDict):
		if 'shared' in new_obs:
			old_obs[..., :self.cfg.shared_dim] = new_obs.shared	
		if 'seperate' in new_obs:
			old_obs[..., self.cfg.shared_dim:self.cfg.shared_dim+self.cfg.seperate_dim] = new_obs.seperate
		if 'ag' in new_obs:
			old_obs[..., self.cfg.shared_dim+self.cfg.seperate_dim-self.cfg.goal_dim:self.cfg.shared_dim+self.cfg.seperate_dim] = new_obs.ag
		if 'g' in new_obs:
			old_obs[..., self.cfg.shared_dim+self.cfg.seperate_dim:] = new_obs.g
		return old_obs

	def info_parser(self, info, name = None):
		assert info.shape[-1] == self.cfg.info_dim, f'info {self.cfg.info_dim} shape error: {info.shape}' 
		if name is None:
			grip_pos_start = 6+self.cfg.goal_dim
			reached_ag_start = grip_pos_start + self.cfg.num_robots*3
			ag_unmoved_steps_start = reached_ag_start + self.cfg.num_goals
			return AttrDict(
				success = info[..., 0], 
				step= info[...,1],
				early_termin = info[...,2],
				traj_idx = info[..., 3],
				traj_len = info[..., 4],
				tleft = info[..., 5],
				ag = info[..., 6:6+self.cfg.goal_dim], 
				grip_pos = info[..., grip_pos_start:reached_ag_start],
				reached_ag = info[...,reached_ag_start:ag_unmoved_steps_start], 
				ag_unmoved_steps = info[...,ag_unmoved_steps_start:ag_unmoved_steps_start+self.cfg.num_goals], 
			)
		elif name == 'success':
			return info[..., 0]
		elif name == 'step':
			return info[..., 1]
		elif name == 'early_termin':
			return info[..., 2]
		elif name == 'traj_idx':
			return info[..., 3]
		elif name == 'traj_len':
			return info[..., 4]
		elif name == 'tleft':
			return info[..., 5]
		elif name == 'ag':
			return info[..., 6:6+self.cfg.goal_dim]
		elif name == 'grip_pos':
			return info[..., 6+self.cfg.goal_dim:6+self.cfg.goal_dim+self.cfg.num_robot*3]
		elif name == 'reached_ag':
			return info[..., 6+self.cfg.goal_dim+self.cfg.num_robot*3:6+self.cfg.goal_dim+self.cfg.num_robot*3+self.cfg.num_goals]
		elif name == 'ag_unmoved_steps':
			return info[..., 6+self.cfg.goal_dim+self.cfg.num_robot*3+self.cfg.num_goals:6+self.cfg.goal_dim+self.cfg.num_robot*3+self.cfg.num_goals*2]
		else:
			raise NotImplementedError

	def info_updater(self, old_info, new_info:AttrDict):
		if 'success' in new_info:
			old_info[..., 0] = new_info.success
		if 'step' in new_info:
			old_info[..., 1] = new_info.step
		if 'early_termin' in new_info:
			old_info[..., 2] = new_info.early_termin
		if 'traj_idx' in new_info:
			old_info[..., 3] = new_info.traj_idx
		if 'traj_len' in new_info:
			old_info[..., 4] = new_info.traj_len
		if 'tleft' in new_info:
			old_info[..., 5] = new_info.tleft
		if 'ag' in new_info:
			old_info[..., 6:6+self.cfg.num_goals*3] = new_info.ag
		if 'grip_pos' in new_info:
			old_info[..., 6+self.cfg.num_goals*3:6+self.cfg.num_goals*3+self.cfg.num_robot*3] = new_info.grip_pos
		if 'reached_ag' in new_info:
			old_info[..., 6+self.cfg.goal_dim+self.cfg.num_robot*3:6+self.cfg.goal_dim+self.cfg.num_robot*3+self.cfg.num_goals] = new_info.reached_ag
		if 'ag_unmoved_steps' in new_info:
			old_info[..., 6+self.cfg.goal_dim+self.cfg.num_robot*3+self.cfg.num_goals:6+self.cfg.goal_dim+self.cfg.num_robot*3+self.cfg.num_goals*2] = new_info.ag_unmoved_steps
		return old_info

	def sample_goal(self, size, norm = True):
		goal_workspace = torch.randint(self.cfg.num_robots,size=(size, ), device=self.device)
		goal = self.torch_goal_space.sample((size,))+self.origin_shift[goal_workspace]
		if norm:
			return (goal-self.goal_mean)/self.goal_std
		else:
			return goal

	def close(self):
		self.gym.destroy_viewer(self.viewer)
		self.gym.destroy_sim(self.sim)

	def env_params(self):
		return AttrDict(
			# info for relable
			max_ag_unmoved_steps = self.cfg.max_ag_unmoved_steps, 
			ag_moved_threshold = self.cfg.ag_moved_threshold,
			# dims
			action_dim=self.cfg.action_dim,
			state_dim=self.cfg.state_dim,
			shared_dim=self.cfg.shared_dim,
			seperate_dim=self.cfg.seperate_dim,
			goal_dim=self.cfg.goal_dim,
			info_dim=self.cfg.info_dim,  # is_success, step, achieved_goal
			# numbers
			num_goals=self.cfg.num_goals,
			num_envs=self.cfg.num_envs,
			max_env_step=self.cfg.max_steps,
			early_termin_step=self.cfg.early_termin_step,
			# functions
			sample_goal=self.sample_goal, 
			compute_reward=self.compute_reward,
			info_parser=self.info_parser,
			info_updater=self.info_updater,
			obs_parser=self.obs_parser, 
			obs_updater=self.obs_updater,
			obs_mirror=self.obs_mirror,
		)


gym.register(id='gripPNP-v0', entry_point=gripCube)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--random', action='store_true')
	parser.add_argument('-e', '--ezpolicy', action='store_true')
	parser.add_argument('-a', '--action', nargs='+', type=float, default = [0,0,0,0])
	args = parser.parse_args()
	'''
	run policy
	'''
	env = gym.make('gripPNP-v0', num_envs=1, num_robots=2, num_cameras=0, headless=False, bound_robot=True, sim_device_id=0, rl_device_id=0, num_goals=2, inhand_rate=0.0, obj_sample_mode='task_distri', task_distri=[0,0,1], goal_os_rate=1.0, control_freq_inv=1)
	start = time.time()
	# action_list = [
	# 	*([[1,0,0,1]]*4), 
	# 	*([[0,1,0,1]]*4),
	# 	*([[-1,0,0,1]]*4),
	# 	*([[0,-1,0,1]]*4),]
	obs = env.reset()[0]
	for i in range(10):
		for j in range(env.cfg.max_steps):
			if args.random:
				act = torch.randn((env.cfg.num_envs,4*env.cfg.num_robots), device=env.device)
				# act[..., 3] = -1
				# act[..., 0] = 1
				act[..., 7] = -1
				act[..., 4] = -1
			elif args.ezpolicy:
				act = env.ezpolicy(obs)
			else:
				act = torch.tensor([args.action]*env.cfg.num_envs, device=env.device)
				# act = torch.tensor([action_list[j%16]]*env.cfg.num_robots*env.cfg.num_envs, device=env.device)
			obs, rew, done, info = env.step(act)
			# env.render(mode='human')
			# info_dict = env.info_parser(info)
			# print(info_dict.step.item())
		# Image.fromarray(images[0]).save('foo.png')

	print(time.time()-start)
	env.close()