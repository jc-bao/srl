from socket import EAI_OVERFLOW
from this import d
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


class FrankaCube(gym.Env):
	def __init__(self, cfg_file='Franka.yaml', **kwargs):
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
		# self.chain = pk.build_serial_chain_from_urdf(open("./isaac_assets/urdf/franka_description/robots/franka_panda.urdf").read(), "panda_hand").to(device = self.device)

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
			low=torch.tensor([-(self.cfg.robot_gap-self.cfg.block_size)/2, -self.cfg.goal_space[1]/1.5, self.cfg.block_size/2],
											 device=self.device),
			high=torch.tensor([(self.cfg.robot_gap-self.cfg.block_size)/2, self.cfg.goal_space[1]/1.5, self.cfg.block_size/2+self.cfg.goal_space[2]*1.5], device=self.device))
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
			[0.,0,0,-1,0,0],
			[0,0,0,0,-1,0],
			[0,0,0,0,0,1],
			[-1,0,0,0,0,0],
			[0,-1,0,0,0,0],
			[0,0,1,0,0,0],
		], device=self.device)
		pos_rot_mat = torch.tensor([
			[-1.,0,0],
			[0,-1,0],
			[0,0,1]
		], device=self.device)
		quat_rot_mat = torch.tensor([
			[0.,-1,0,0],
			[1,0,0,0],
			[0,0,0,1],
			[0,0,1,0],
		], device=self.device)
		block_other_mat = torch.block_diag(*([quat_rot_mat, pos_rot_mat]))
		# obs=(
		# [p_r0[3],v_r0[3],p_g0[1]]
		# [rot_block_i[4], p_block_i[3]]* block_num
		# [p_goal_i[3]]*num_goals)
		if self.cfg.enable_robot_id:
			self.robot_id = (torch.arange(self.cfg.num_robots, device=self.device)/self.cfg.num_robots).repeat(self.cfg.num_envs, 1)
			id_rot_mat = torch.tensor([
				[0, 1],
				[1, 0]
			], device=self.device)
			self.obs_rot_mat = torch.block_diag(*([id_rot_mat]+[robot_pos_rot_mat]*2+[torch.tensor([[0.,1],[1.,0]],device=self.device)]+[block_other_mat]*self.cfg.num_goals+[pos_rot_mat]*self.cfg.num_goals))
			self.other_robot_obs_mask = torch.tensor([1.,0]+[1]*3+[0]*3+[1]*3+[0]*3+[1,0]+10*self.cfg.num_goals*[1], device=self.device)
		else:
			self.obs_rot_mat = torch.block_diag(*([robot_pos_rot_mat]*2+[torch.tensor([[0.,1],[1.,0]],device=self.device)]+[block_other_mat]*self.cfg.num_goals+[pos_rot_mat]*self.cfg.num_goals))
			self.other_robot_obs_mask = torch.tensor([1]*3+[0]*3+[1]*3+[0]*3+[1,0]+10*self.cfg.num_goals*[1], device=self.device)
		self.single_act_rot_mat = torch.tensor(
			[[-1.,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], device=self.device)
		self.last_act_rot_mat = torch.block_diag(torch.eye(4,dtype=torch.float,device=self.device),self.single_act_rot_mat)
		self.act_rot_mat = torch.tensor(
			[[0,0,0,0,-1.,0,0,0],[0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],
			[-1.,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]
			], device=self.device)
		self.dual_act_rot_mat = torch.block_diag(torch.eye(self.cfg.action_dim, device=self.device, dtype=torch.float), self.act_rot_mat)
		# convert mixture robot obs to seperate robot obs
		self.robot_reshape_mat = torch.tensor(
			[
				[1.,0,0]+[0]*11,
				[0,1,0]+[0]*11,
				[0,0,1]+[0]*11,
				[0]*7+[1,0,0]+[0]*4,
				[0]*7+[0,1,0]+[0]*4,
				[0]*7+[0,0,1]+[0]*4,
				[0]*3+[1,0,0]+[0]*8,
				[0]*3+[0,1,0]+[0]*8,
				[0]*3+[0,0,1]+[0]*8,
				[0]*10+[1,0,0]+[0],
				[0]*10+[0,1,0]+[0],
				[0]*10+[0,0,1]+[0],
				[0]*6+[1]+[0]*7,
				[0]*13+[1],
			],device=self.device
		)
		self.reset()

	def set_viewer(self):
		self.enable_viewer_sync = True
		self.viewer = None
		# if running with a viewer, set up keyboard shortcuts and camera
		if self.cfg.headless == False:
			# subscribe to keyboard shortcuts
			camera_setting = gymapi.CameraProperties()
			self.viewer = self.gym.create_viewer(self.sim, camera_setting)
			cam_pos = gymapi.Vec3(1.5, -1.5, 2)
			look_at = gymapi.Vec3(0.0, 0.0, 0.5)
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
		# self.franka_default_ee_pos = to_torch(
		# 	self.cfg.ee_init_pos,
		# 	device=self.device,) 
		# self.franka_default_dof_pos = to_torch(
		# 	[[0.1840,  0.4244, -0.1571, -2.3733,  0.1884,  2.7877,  2.2164, 0.02, 0.02]],
		# 	device=self.device,)
		self.predefined_dof_pos = torch.load(self.cfg_path/'default_joint_pos.pt').to(self.device)
		random_idx = torch.randint(low=0, high=self.predefined_dof_pos.shape[0], size=(self.cfg.num_envs*self.cfg.num_robots,), device=self.device)
		self.franka_default_dof_pos = torch.empty((self.cfg.num_envs*self.cfg.num_robots, self.predefined_dof_pos.shape[-1]), device=self.device)
		self.franka_default_dof_pos = self.predefined_dof_pos[random_idx]
		# self.franka_default_dof_state = self.franka_default_dof_pos.unsqueeze(-1).repeat(self.cfg.num_envs,self.cfg.num_robots,2)
		self.franka_default_dof_state = self.franka_default_dof_pos.unsqueeze(-1).repeat(1,1,2)
		self.franka_default_dof_state[...,-1] = 0.
		orns = [[0.924, -0.383, 0., 0.],[0.383, 0.924, 0., 0.]]
		# orns = [[1.0, 0., 0., 0.],[-1.0, 0., 0., 0.]]
		self.franka_default_orn = to_torch(
			[[orns[i%2] for i in range(self.cfg.num_robots)]], device=self.device).repeat(self.cfg.num_envs, 1, 1)
		lower = gymapi.Vec3(-self.cfg.env_spacing, -self.cfg.env_spacing, 0.0)
		upper = gymapi.Vec3(*([self.cfg.env_spacing]*3))

		asset_root = os.path.join(
			os.path.dirname(os.path.abspath(__file__)), self.cfg.asset.assetRoot)
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
		box_opts.density = 50
		box_opts.angular_damping = 100
		box_opts.linear_damping = 10
		box_opts.thickness = 0.005
		if self.cfg.lock_block_orn and self.cfg.num_robots > 1:
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


		franka_start_poses = []
		self.franka_roots = []
		self.origin_shift = []
		for franka_id in range(self.cfg.num_robots):
			side = franka_id % 2 * 2 - 1
			x = self.cfg.robot_gap*(franka_id+0.5-self.cfg.num_robots*0.5)
			franka_start_pose = gymapi.Transform()
			pos = [x, side*self.cfg.robot_y, self.cfg.table_size[2]]
			franka_start_pose.p = gymapi.Vec3(*pos)
			pos[1] = 0
			self.origin_shift.append(pos)
			rot = [0.0, 0.0, np.sqrt(2)/2, -side*np.sqrt(2)/2]
			franka_start_pose.r = gymapi.Quat(*rot)
			franka_start_poses.append(franka_start_pose)
			self.franka_roots.append([*pos, *rot])
		self.origin_shift = to_torch(self.origin_shift, device=self.device)
		self.franka_roots = to_torch(self.franka_roots, device=self.device)
		self.default_grip_pos = self.origin_shift.unsqueeze(0).repeat(self.cfg.num_envs,1,1).clone()

		# compute aggregate size
		num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
		num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
		num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
		num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
		num_block_bodies = self.gym.get_asset_rigid_body_count(block_asset)
		num_block_shapes = self.gym.get_asset_rigid_shape_count(block_asset)
		num_goal_bodies = self.gym.get_asset_rigid_body_count(goal_asset)
		num_goal_shapes = self.gym.get_asset_rigid_shape_count(goal_asset)
		max_agg_bodies = (
			(num_franka_bodies+num_table_bodies)*self.cfg.num_robots+ self.cfg.num_goals * (num_block_bodies+num_goal_bodies))
		max_agg_shapes = (
			(num_franka_shapes+num_table_shapes)*self.cfg.num_robots + self.cfg.num_robots + self.cfg.num_goals * (num_block_shapes+num_goal_shapes))

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
			franka_actors = []
			for franka_id in range(self.cfg.num_robots):
				franka_pos = franka_start_poses[franka_id]
				# Key: create Panda
				franka_actor = self.gym.create_actor(
					env_ptr, franka_asset, franka_pos, f"franka{franka_id}", i, 0, i
				)
				self.gym.set_actor_dof_properties(
					env_ptr, franka_actor, franka_dof_props)
				franka_actors.append(franka_actor)
			for franka_id in range(self.cfg.num_robots):
				franka_pos = franka_start_poses[franka_id]
				# create table
				table_state_pose = gymapi.Transform()
				table_state_pose.p = gymapi.Vec3(franka_pos.p.x, 0, self.cfg.table_size[2]/2)
				table_state_pose.r = gymapi.Quat(0, 0, 0, 1)
				table_actor = self.gym.create_actor(
						env_ptr, table_asset, table_state_pose, "table{}".format(franka_id), i, 0, 0,)
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
					block_state_pose.p.x = franka_pos.p.x 
					block_state_pose.p.y = -0.4 - j * self.cfg.block_size * 1.5
					block_state_pose.p.z = self.cfg.table_size[2] + self.cfg.block_size/2
					if np.random.rand() < 0.5:
						block_state_pose.r = gymapi.Quat(0, 0, 1, 0)
					else:
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
					goal_state_pose.p.x = franka_pos.p.x  
					goal_state_pose.p.y = -0.4 - j * self.cfg.block_size * 1.5
					goal_state_pose.p.z = self.cfg.table_size[2] + self.cfg.block_size/2 + 0.2
					goal_state_pose.r = gymapi.Quat(0, 0, 0, 1)
					handle = self.gym.create_actor(
						env_ptr, goal_asset, goal_state_pose, "goal{}".format(j), i+self.cfg.num_envs, 0, 0,)
					self.gym.set_rigid_body_color(
						env_ptr, handle, 0, gymapi.MESH_VISUAL, self.colors[j])
					self.goal_handles.append(handle)
			if self.cfg.aggregate_mode > 0:
				self.gym.end_aggregate(env_ptr)
			self.envs.append(env_ptr)
			self.frankas.append(franka_actors)
		for j in range(self.cfg.num_cameras):
			# create camera
			camera_properties = gymapi.CameraProperties()
			camera_properties.width = self.cfg.render_size[0] # 320
			camera_properties.height = self.cfg.render_size[1] # 200
			h1 = self.gym.create_camera_sensor(self.envs[j], camera_properties)
			# camera_position = gymapi.Vec3(1, -1, 1)
			# camera_target = gymapi.Vec3(0, 0, 0)
			camera_position = gymapi.Vec3(1.5, -1.5, 2)
			camera_target = gymapi.Vec3(0.0, 0.0, 0.5)
			self.gym.set_camera_location(
				h1, self.envs[j], camera_position, camera_target)
			self.cameras.append(h1)
		# set control data
		self.hand_handles = [self.gym.find_actor_rigid_body_handle(
			env_ptr, r, "panda_link7") for r in franka_actors]
		self.lfinger_handles = [self.gym.find_actor_rigid_body_handle(
			env_ptr, r, "panda_leftfinger") for r in franka_actors]
		self.rfinger_handles = [self.gym.find_actor_rigid_body_handle(
			env_ptr, r, "panda_rightfinger") for r in franka_actors]
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

		# hand = self.gym.find_actor_rigid_body_handle(
		# 	self.envs[0], self.frankas[0], "panda_link7")
		# lfinger = self.gym.find_actor_rigid_body_handle(
		# 	self.envs[0], self.frankas[0], "panda_leftfinger")
		# rfinger = self.gym.find_actor_rigid_body_handle(
		# 	self.envs[0], self.frankas[0], "panda_rightfinger")

		# hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
		# lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
		# rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

		# finger_pose = gymapi.Transform()
		# finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
		# finger_pose.r = lfinger_pose.r

		# hand_pose_inv = hand_pose.inverse()
		# grasp_pose_axis = 1
		# franka_local_grasp_pose = hand_pose_inv * finger_pose
		# franka_local_grasp_pose.p += gymapi.Vec3(
		# 	*get_axis_params(0.04, grasp_pose_axis))
		# self.franka_local_grasp_pos = to_torch(
		# 	[
		# 		franka_local_grasp_pose.p.x,
		# 		franka_local_grasp_pose.p.y,
		# 		franka_local_grasp_pose.p.z,
		# 	],
		# 	device=self.device,
		# ).repeat((self.cfg.num_envs, 1))
		# self.franka_local_grasp_rot = to_torch(
		# 	[
		# 		franka_local_grasp_pose.r.x,
		# 		franka_local_grasp_pose.r.y,
		# 		franka_local_grasp_pose.r.z,
		# 		franka_local_grasp_pose.r.w,
		# 	],
		# 	device=self.device,
		# ).repeat((self.cfg.num_envs, 1))

		# self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat(
		# 	(self.cfg.num_envs, 1)
		# )
		# self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat(
		# 	(self.cfg.num_envs, 1)
		# )

		# self.franka_lfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
		# self.franka_rfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
		# self.franka_lfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
		# self.franka_rfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)

		self.j_eefs = []
		for franka_id in range(self.cfg.num_robots):
			# dof
			_jacobian = self.gym.acquire_jacobian_tensor(self.sim, f"franka{franka_id}")
			jacobian = (gymtorch.wrap_tensor(_jacobian))
			self.j_eefs.append(jacobian[:,
																self.hand_handles[0] - 1, :, :self.franka_hand_index])

		# joint pos
		dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
		self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
		self.num_dofs = self.gym.get_sim_dof_count(self.sim) // (self.cfg.num_envs*self.cfg.num_robots)
		self.franka_dof_targets = torch.zeros(
			(self.cfg.num_envs, self.cfg.num_robots, self.num_dofs), dtype=torch.float, device=self.device
		)
		self.franka_dof_states = self.dof_state.view(self.cfg.num_envs, self.cfg.num_robots, -1, 2)[
			:,:, : self.num_franka_dofs
		]
		self.franka_dof_poses = self.franka_dof_states[..., 0]
		self.franka_dof_vels = self.franka_dof_states[..., 1]
		self.finger_widths = self.franka_dof_poses[:,:,self.franka_hand_index]

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
		self.init_ag = self.block_states[..., :3].clone()
		self.init_ag_normed = torch.zeros_like(self.init_ag, device=self.device, dtype=torch.float) 
		self.last_step_ag = torch.zeros_like(self.block_states[..., :3], device=self.device, dtype=torch.float)
		self.ag_unmoved_steps = torch.zeros((self.cfg.num_envs, self.cfg.num_goals,), device=self.device, dtype=torch.float)
		self.goal = self.root_state_tensor[:, self.cfg.num_robots*2 +
																			 self.cfg.num_goals:self.cfg.num_robots*2+self.cfg.num_goals*2, :3]
		self.goal_workspace = torch.zeros((self.cfg.num_envs, self.cfg.num_goals), device=self.device, dtype=torch.long)
		self.block_workspace = torch.zeros((self.cfg.num_envs, self.cfg.num_goals), device=self.device, dtype=torch.long)
		self.num_os_goal = torch.zeros((self.cfg.num_envs,), device=self.device, dtype=torch.long)
		self.goal_mask = torch.zeros((self.cfg.num_envs, self.cfg.num_goals), device=self.device, dtype=torch.float)
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
		self.franka_lfinger_poses = []
		self.franka_rfinger_poses = []
		self.franka_lfinger_rots = []
		self.franka_rfinger_rots = []
		for i in range(self.cfg.num_robots):
			# NOTE do not use self.lfinger_handles to indice as it will copy the tensor
			self.hand_pos.append(self.rigid_body_states[:, self.hand_handles[i]][..., :3])
			self.hand_rot.append(self.rigid_body_states[:, self.hand_handles[i]][..., 3:7])
			self.hand_vel.append(self.rigid_body_states[:, self.hand_handles[i]][..., 7:10])
			self.franka_lfinger_poses.append(self.rigid_body_states[:,
																											self.lfinger_handles[i]][..., 0:3])
			self.franka_rfinger_poses.append(self.rigid_body_states[:,
																											self.rfinger_handles[i]][..., 0:3])
			self.franka_lfinger_rots.append(self.rigid_body_states[:,
																											self.lfinger_handles[i]][..., 3:7])
			self.franka_rfinger_rots.append(self.rigid_body_states[:,
																											self.rfinger_handles[i]][..., 3:7])
		self.grip_pos = (torch.stack(self.franka_lfinger_poses,dim=1) +
										 torch.stack(self.franka_rfinger_poses,dim=1))/2 + self.finger_shift
		self.hand_pos_tensor = torch.stack(self.hand_pos, dim=1)
		self.target_hand_pos = self.hand_pos_tensor.clone()

	def compute_reward(self, ag, dg, info, normed=True):
		ag = ag.view(-1, self.cfg.num_goals, 3)
		dg = dg.view(-1, self.cfg.num_goals, 3)
		if normed:
			ag = ag*self.goal_std + self.goal_mean
			dg = dg*self.goal_std + self.goal_mean
		if self.cfg.reward_type == 'sparse':
			dist = torch.norm(ag-dg, dim=-1)
			far_goal = (dist > self.cfg.err).type(torch.float32)
			if 'goal_mask' in info:
				return -torch.sum(far_goal*info.goal_mask, dim=-1)/torch.sum(info.goal_mask, dim=-1)
			else:
				return -torch.mean(far_goal, dim=-1)
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
				if k == 'num_goals':
					self.cfg.max_steps = self.cfg.base_steps*self.cfg.current_num_goals*self.cfg.num_robots
				# TODO move this to general update
			else:
				print(f'[Curriculum] config has no attribute {k}')
		# resample original dof pos
		random_idx = torch.randint(low=0, high=self.predefined_dof_pos.shape[0], size=(self.cfg.num_envs*self.cfg.num_robots,), device=self.device)
		self.franka_default_dof_state[:,:,0] = self.predefined_dof_pos[random_idx]
		# step first to init params
		act = torch.zeros((self.cfg.num_envs, 4*self.cfg.num_robots), device=self.device, dtype=torch.float)
		obs, _, _, _ = self.step(act) # TODO try to remove this
		# for _ in range(5):
		# change params here TODO more elegant way
		# if 'table_gap' in config and self.cfg.num_robots == 2:
		# 	v = config['table_gap']
		# 	self.table_states[:,:,0] = torch.tensor([-(v+self.cfg.table_size[0])/2,(v+self.cfg.table_size[0])/2], device=self.device)
		self.reset_buf[:] = True
		obs, rew, done, info = self.step(act)
		'''
		reset to any pos (deprecated as it is slow)
		# pos_target = self.torch_goal_space.sample((self.cfg.num_envs,self.cfg.num_robots))+self.origin_shift - self.finger_shift
		pos_target = self.torch_goal_space.sample((self.cfg.num_envs,self.cfg.num_robots))+self.origin_shift - self.finger_shift
		print(pos_target, self.goal)
		filtered_pos_target = self.hand_pos_tensor.clone()
		for _ in range(100): # TODO wrap up as a function
			# setup control params
			orn_errs = self.orientation_error(self.franka_default_orn, torch.stack(self.hand_rot, dim=1))
			# filtered_pos_target = self.cfg.filter_param * pos_target + (1 - self.cfg.filter_param) * filtered_pos_target
			filtered_pos_target = pos_target
			pos_errs = filtered_pos_target - self.hand_pos_tensor 
			# clip with bound
			if self.cfg.bound_robot:
				pos_errs = torch.clip(pos_errs+self.grip_pos, self.torch_robot_space.low,
														self.torch_robot_space.high) - self.grip_pos
			dposes = torch.cat([pos_errs, orn_errs], -1).unsqueeze(-1)
			self.franka_dof_targets[..., :self.franka_hand_index] = self.control_ik_old(dposes)
			# limit
			self.franka_dof_targets[..., :self.num_franka_dofs] = tensor_clamp(
				self.franka_dof_targets[...,
																: self.num_franka_dofs], self.franka_dof_lower_limits, self.franka_dof_upper_limits)
			# Deploy actions
			act_indices = self.global_indices[:, :self.cfg.num_robots].flatten()
			self.gym.set_dof_position_target_tensor_indexed(
				self.sim, 
				gymtorch.unwrap_tensor(self.franka_dof_targets),
				gymtorch.unwrap_tensor(act_indices),
				act_indices.shape[0])
			# simulate
			self.gym.simulate(self.sim)
			# update state data
			self.gym.refresh_actor_root_state_tensor(self.sim)
			self.gym.refresh_dof_state_tensor(self.sim)
			self.gym.refresh_rigid_body_state_tensor(self.sim)
			self.gym.refresh_jacobian_tensors(self.sim)
			self.hand_pos_tensor = torch.stack(self.hand_pos, dim=1)
		print('end:', self.hand_pos_tensor - pos_target)
		'''
		self.progress_buf[:] = 0  # NOTE: make sure step start from 0
		self.default_grip_pos = self.grip_pos.clone()
		return obs, rew, done, info

	def step(self, actions: torch.Tensor):
		# apply actions
		reset_idx = self.reset_buf.clone()
		done_env_num = reset_idx.sum()
		# reset goals
		# self.goal_workspace[reset_idx] = torch.randint(self.cfg.num_robots,size=(done_env_num.item(),self.cfg.num_goals), device=self.device)
		if done_env_num > 0:
			sampled_goal_num = 0
			max_num_goals = min(self.cfg.num_goals, int(self.cfg.current_num_goals))
			new_goal = self.goal[reset_idx, :max_num_goals].clone() 
			new_goal_ws = self.goal_workspace[reset_idx, :max_num_goals].clone() 
			for k in range(self.cfg.max_sample_time):
				if self.cfg.goal_sample_mode == 'uniform':
					extra_goal_ws = torch.randint(self.cfg.num_robots,size=(done_env_num.item(),max_num_goals), device=self.device).repeat(self.cfg.extra_goal_sample,1,1)
				elif self.cfg.goal_sample_mode == 'bernoulli': # TODO extend to multi arm scenario
					extra_goal_ws = torch.randint(self.cfg.num_robots,size=(done_env_num.item(),1), device=self.device).repeat(self.cfg.extra_goal_sample,1,max_num_goals)
					goal_ws_shift = torch.bernoulli(torch.ones((done_env_num.item(),max_num_goals-1), device=self.device, dtype=torch.float)*self.cfg.goal_os_rate).long()
					self.num_os_goal[reset_idx] = goal_ws_shift.sum(dim=-1)
					extra_goal_ws_shift = goal_ws_shift.repeat(self.cfg.extra_goal_sample,1,1)
					extra_goal_ws[...,1:] += extra_goal_ws_shift
					extra_goal_ws %= self.cfg.num_robots
				sampled_goal = self.torch_goal_space.sample((self.cfg.extra_goal_sample, done_env_num.item(),max_num_goals))
				goal_drift = torch.tensor([0,0,self.cfg.block_size/2], device=self.device)
				sampled_goal = (sampled_goal - goal_drift)*self.cfg.goal_scale + goal_drift
				extra_goals = sampled_goal + \
					self.origin_shift[extra_goal_ws.flatten()].view(self.cfg.extra_goal_sample, done_env_num.item(), max_num_goals, 3)
				goal_dist = torch.abs(extra_goals.unsqueeze(-3) - extra_goals.unsqueeze(-2))
				satisfied_idx = ((goal_dist[...,0] > self.cfg.block_length*1.2) | \
					(goal_dist[..., 1] > self.cfg.block_size*2) | \
							torch.eye(max_num_goals, device=self.device, dtype=torch.bool)).all(dim=-1).all(dim=-1)
				new_sampled_goal_num = satisfied_idx.sum().item()
				new_goal_num = min(new_sampled_goal_num, done_env_num.item()-sampled_goal_num)
				if new_goal_num > 0:
					new_goal[sampled_goal_num:sampled_goal_num+new_goal_num] = extra_goals[satisfied_idx][:new_goal_num]
					new_goal_ws[sampled_goal_num:sampled_goal_num+new_goal_num] = extra_goal_ws[satisfied_idx][:new_goal_num]
				sampled_goal_num+=new_goal_num
				if sampled_goal_num >= done_env_num.item():
					break
			self.goal[reset_idx, :max_num_goals] = new_goal 
			self.goal_workspace[reset_idx, :max_num_goals] = new_goal_ws
			if sampled_goal_num < (done_env_num.item()):
				print('[Env] Warning: goal sampling failed')
			num_goals = int(self.cfg.current_num_goals)
			rand_num_rate = self.cfg.current_num_goals - num_goals
			self.goal_mask[reset_idx, :num_goals] = 1.0
			if num_goals < self.cfg.num_goals:
				self.goal_mask[reset_idx, num_goals] = (torch.rand(done_env_num.item(), device=self.device) < rand_num_rate).float()
				self.goal_mask[reset_idx, num_goals+1:] = 0.0
		multi_goal_in_same_ws = torch.zeros((self.cfg.num_envs,), device=self.device, dtype=torch.bool)
		for i in range(self.cfg.num_robots):
			multi_goal_in_same_ws |= ((self.goal_workspace==i).sum(dim=-1) > 1)
		# reset tables
		if self.cfg.num_robots == 2:
			new_gap = self.cfg.table_size[0] + torch.clip(self.cfg.table_gap + torch.rand((done_env_num), device=self.device) * self.cfg.rand_table_gap, min=0, max=self.cfg.max_table_gap)
			self.table_states[reset_idx,0,0] = -new_gap/2
			self.table_states[reset_idx,1,0] = new_gap/2
		# reset blocks
		if done_env_num > 0:
			block_indices = self.global_indices[reset_idx, 1:].flatten()
			# set to default pos
			self.block_states[reset_idx] = self.default_block_states[reset_idx]
			in_hand = torch.rand((self.cfg.num_envs,),
													 device=self.device) < self.cfg.inhand_rate
			self.inhand_idx = reset_idx & in_hand
			sampled_ag_num = 0
			new_ag = self.init_ag[reset_idx, :max_num_goals].clone()
			new_ag_ws = self.block_workspace[reset_idx, :max_num_goals].clone()
			for k in range(self.cfg.max_sample_time):
				if self.cfg.obj_sample_mode == 'uniform':
					extra_block_ws = torch.randint(self.cfg.num_robots,size=(done_env_num.item(),max_num_goals), device=self.device).repeat(self.cfg.extra_goal_sample,1,1)
				elif self.cfg.obj_sample_mode == 'bernoulli': # TODO extend to multi arm scenario
					tiled_goal_ws = self.goal_workspace[reset_idx, :max_num_goals].repeat(self.cfg.extra_goal_sample,1,1)
					if max_num_goals <= self.cfg.max_handover_time:
						extra_block_ws = tiled_goal_ws + torch.bernoulli(torch.ones(tiled_goal_ws.shape[1:], device=self.device, dtype=torch.float)*self.cfg.os_rate).long().repeat(self.cfg.extra_goal_sample,1,1)
					else: # limit the max handover times
						same_side_g_num = max_num_goals - self.cfg.max_handover_time
						rand_side_g_num = self.cfg.max_handover_time
						drifted_ws = torch.bernoulli(torch.ones((done_env_num.item(), rand_side_g_num), device=self.device, dtype=torch.float)*self.cfg.os_rate).long()
						not_drifted_ws = torch.zeros((done_env_num.item(), same_side_g_num), device=self.device).long()
						new_ws = torch.cat((drifted_ws,not_drifted_ws), dim=-1).repeat(self.cfg.extra_goal_sample,1,1) 
						extra_block_ws = tiled_goal_ws + new_ws
					extra_block_ws %= self.cfg.num_robots
				elif self.cfg.obj_sample_mode == 'task_distri':
					rand_number = torch.rand((done_env_num,), device=self.device)
					block_ws = torch.zeros((done_env_num, max_num_goals), device=self.device, dtype=torch.long)
					now_prob = self.cfg.task_distri[0] 
					for i in range(1, max_num_goals+1):
						block_ws[now_prob<=rand_number<now_prob+self.cfg.task_distri[i], :i] = 1
						now_prob += self.cfg.task_distri[i]
					block_ws += self.goal_workspace[reset_idx] 
					extra_block_ws = block_ws.repeat(self.cfg.extra_goal_sample,1,1)
					extra_block_ws %= self.cfg.num_robots
				else:
					raise NotImplementedError
				# TODO fix this
				sampled_ag = self.torch_block_space.sample((self.cfg.extra_goal_sample, done_env_num,max_num_goals))
				goal_dift = torch.tensor([0,0,self.cfg.block_size/2], device=self.device)
				sampled_ag = (sampled_ag - goal_dift)*self.cfg.goal_scale + goal_dift
				extra_ags = sampled_ag + \
					self.origin_shift[extra_block_ws.flatten()].view(self.cfg.extra_goal_sample, done_env_num, max_num_goals, 3)
				ag_dist = torch.abs(extra_ags.unsqueeze(-3) - extra_ags.unsqueeze(-2))
				satisfied_idx = ((ag_dist[...,0] > self.cfg.block_length*1.2) | \
					(ag_dist[..., 1] > self.cfg.block_size*2) | \
							torch.eye(max_num_goals, device=self.device, dtype=torch.bool)).all(dim=-1).all(dim=-1)
				new_sampled_ag_num = satisfied_idx.sum().item()
				new_ag_num = min(new_sampled_ag_num, done_env_num.item()-sampled_ag_num)
				if new_ag_num > 0:
					new_ag[sampled_ag_num:sampled_ag_num+new_ag_num] = extra_ags[satisfied_idx][:new_ag_num]
					new_ag_ws[sampled_ag_num:sampled_ag_num+new_ag_num] = extra_block_ws[satisfied_idx][:new_ag_num]
				sampled_ag_num+=new_ag_num
				if sampled_ag_num >= done_env_num.item():
					break
			self.init_ag[reset_idx, :max_num_goals] = new_ag
			self.block_workspace[reset_idx, :max_num_goals] = new_ag_ws
			if sampled_ag_num < (done_env_num.item()):
				print('[Env] Warning: ag sampling failed')
			if self.inhand_idx.any():
				# choosed_block = torch.randint(max_num_goals, (1,), device=self.device)[0]
				# NOTE can only choose block 0 in hand now TODO fix it
				choosed_block = 0 
				choosed_robot = torch.randint(high=self.cfg.num_robots,size=(self.inhand_idx.sum().item(),), device=self.device)
				self.init_ag[self.inhand_idx, choosed_block] = self.default_grip_pos[self.inhand_idx, choosed_robot] + \
					(torch.rand_like(self.default_grip_pos[self.inhand_idx, choosed_robot], device=self.device) - 0.5) * to_torch([self.cfg.block_length*0.7, 0., 0.], device=self.device)
				self.block_workspace[self.inhand_idx, choosed_block] = choosed_robot 
				if max_num_goals > 1 and self.cfg.num_robots > 1 and torch.rand(1)[0] < 0.5:
					choosed_block = (choosed_block+1)%max_num_goals 
					choosed_robot = (choosed_robot+1)%self.cfg.num_robots
					self.init_ag[self.inhand_idx, choosed_block] = self.default_grip_pos[self.inhand_idx, choosed_robot] + \
						(torch.rand_like(self.default_grip_pos[self.inhand_idx, choosed_robot], device=self.device) - 0.5) * to_torch([self.cfg.block_length*0.7, 0., 0.], device=self.device)
					self.block_workspace[self.inhand_idx, choosed_block] = choosed_robot 
			self.num_handovers = (self.block_workspace != self.goal_workspace).sum(dim=-1)
			self.last_step_ag[reset_idx] = self.init_ag[reset_idx]
			self.ag_unmoved_steps[reset_idx] = 0
			self.block_states[reset_idx,:max_num_goals,:3] = self.init_ag[reset_idx, :max_num_goals]
			self.init_ag_normed[reset_idx] = ((self.init_ag[reset_idx]-self.goal_mean)/self.goal_std)
			# change some goal to the ground
			ground_goal_idx = reset_idx & ((torch.rand((self.cfg.num_envs,),device=self.device) < self.cfg.goal_ground_rate) | multi_goal_in_same_ws | (self.num_handovers > 0))
			self.goal[ground_goal_idx, :, -1] = self.cfg.table_size[2]+self.cfg.block_size/2
			# change to hand or random pos
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
		self.actions = torch.clip(
			actions.clone().view(self.cfg.num_envs,self.cfg.num_robots,self.cfg.per_action_dim).to(self.device)+self.cfg.action_shift, 
			-self.cfg.clip_actions, self.cfg.clip_actions)
		pos_target = self.actions[..., :3] * self.cfg.dt * self.cfg.control_freq_inv * self.cfg.max_vel + self.hand_pos_tensor
		filtered_pos_target = self.hand_pos_tensor
		# step physics and render each frame
		for i in range(self.cfg.control_freq_inv):
			# setup control params
			orn_errs = self.orientation_error(self.franka_default_orn, torch.stack(self.hand_rot, dim=1))
			# self.target_hand_pos = self.actions[..., :3] * self.cfg.dt * self.cfg.max_vel + self.target_hand_pos
			# pos_errs = self.target_hand_pos - self.hand_pos_tensor 
			filtered_pos_target = self.cfg.filter_param * pos_target + (1 - self.cfg.filter_param) * filtered_pos_target
			pos_errs = filtered_pos_target - self.hand_pos_tensor 
			# pos_errs[reset_idx] = self.torch_block_space.sample((done_env_num,self.cfg.num_robots))+self.origin_shift.tile(done_env_num,1,1) - self.grip_pos[reset_idx]
			# clip with bound
			if self.cfg.bound_robot:
				pos_errs = torch.clip(pos_errs+self.grip_pos-self.origin_shift, self.torch_robot_space.low,
														self.torch_robot_space.high) - self.grip_pos + self.origin_shift
			dposes = torch.cat([pos_errs, orn_errs], -1).unsqueeze(-1)
			self.franka_dof_targets[..., :self.franka_hand_index] = self.control_ik_old(dposes)
			# grip
			grip_acts = self.franka_dof_poses[..., [self.franka_hand_index]] + self.actions[..., [3]] * self.cfg.dt * self.cfg.max_grip_vel
			# reset gripper
			self.franka_dof_targets[..., self.franka_hand_index:
															self.franka_hand_index+2] = grip_acts.repeat(1, 1, 2)
			# limit
			self.franka_dof_targets[..., :self.num_franka_dofs] = tensor_clamp(
				self.franka_dof_targets[...,
																: self.num_franka_dofs], self.franka_dof_lower_limits, self.franka_dof_upper_limits)
			# Deploy actions
			if done_env_num < self.cfg.num_envs:
				# set action
				act_indices = self.global_indices[~reset_idx, :self.cfg.num_robots].flatten()
				self.gym.set_dof_position_target_tensor_indexed(
					self.sim, 
					gymtorch.unwrap_tensor(self.franka_dof_targets),
					gymtorch.unwrap_tensor(act_indices),
					act_indices.shape[0])
			
			if done_env_num > 0:
				reset_indices = self.global_indices[reset_idx, :self.cfg.num_robots].flatten()
				self.gym.set_dof_state_tensor_indexed(
					self.sim, 
					gymtorch.unwrap_tensor(self.franka_default_dof_state),
					gymtorch.unwrap_tensor(reset_indices),
					reset_indices.shape[0])
			# simulate
			self.gym.simulate(self.sim)
			# update state data
			self.gym.refresh_actor_root_state_tensor(self.sim)
			self.gym.refresh_dof_state_tensor(self.sim)
			self.gym.refresh_rigid_body_state_tensor(self.sim)
			self.gym.refresh_jacobian_tensors(self.sim)
			self.hand_pos_tensor = torch.stack(self.hand_pos, dim=1)
			self.target_hand_pos[reset_idx] = self.hand_pos_tensor[reset_idx] 
		if not self.cfg.headless:
			self.render(mode='human')
		if self.device == "cpu":
			self.gym.fetch_results(self.sim, True)
		# compute observations, rewards, resets, ...
		# update state buffer
		self.progress_buf += 1

		# update obs, rew, done, info
		self.grip_pos = (torch.stack(self.franka_lfinger_poses,dim=1) +
										 torch.stack(self.franka_rfinger_poses,dim=1))/2 + self.finger_shift
		grip_pos_normed = (self.grip_pos-self.goal_mean)/self.goal_std
		hand_vel_normed = (torch.stack(self.hand_vel,dim=1)-self.hand_vel_mean)/self.hand_vel_std
		finger_widths_normed = (self.finger_widths.unsqueeze(-1)-self.finger_width_mean) / self.finger_width_std
		block_pos_normed = (self.block_states[..., :3]-self.goal_mean) / self.goal_std # CHECK multi robot
		# NOTE make sure achieved goal is close to end
		block_obs = torch.cat((self.block_states[..., 3:7], block_pos_normed), dim=-1)
		goal_normed = (self.goal-self.goal_mean)/self.goal_std
		obs = torch.cat((
			grip_pos_normed.view(self.cfg.num_envs, self.cfg.num_robots*3),  # mid finger
			hand_vel_normed.view(self.cfg.num_envs, self.cfg.num_robots*3),
			finger_widths_normed.view(self.cfg.num_envs, self.cfg.num_robots),  # robot
			block_obs.view(self.cfg.num_envs, -1),  # objects
			goal_normed.view(self.cfg.num_envs, self.cfg.num_goals*3),
		), dim=-1)
		if self.cfg.enable_robot_id:
			obs = torch.cat((self.robot_id, obs), dim=-1)

		# rew
		rew = self.compute_reward(
			self.block_states[..., :3], self.goal, AttrDict(grip_pos=self.grip_pos, goal_mask=self.goal_mask.bool()), normed=False)
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
			# goal masks
			self.goal_mask.view(self.cfg.num_envs, -1),
		), dim=-1)
		self.last_step_ag = self.block_states[..., :3].clone()

		# debug viz
		if self.viewer and self.cfg.debug_viz:
			self.gym.clear_lines(self.viewer)

			for i in range(self.cfg.num_envs):
				for j in range(self.cfg.num_robots):
					# draw finger mid
					finger_mid = (
						self.franka_lfinger_poses[j][i] + self.franka_rfinger_poses[j][i])/2 + self.finger_shift
					px = ((finger_mid + quat_apply(self.franka_lfinger_rots[j][i],
																				to_torch([1, 0, 0], device=self.device) * 0.2,)).cpu().numpy())
					py = ((finger_mid + quat_apply(self.franka_lfinger_rots[j][i],
																				to_torch([0, 1, 0], device=self.device) * 0.2,)).cpu().numpy())
					pz = ((finger_mid + quat_apply(self.franka_lfinger_rots[j][i],
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

	def control_ik(self, dpose):
		# solve damped least squares
		num_it = 0
		lmbda = torch.eye(6, device=self.device) * (self.cfg.damping ** 2)
		# init_joint = self.franka_dof_poses[...,:self.franka_hand_index] 
		current_joint = self.franka_dof_poses[...,:self.franka_hand_index].clone()
		target_ee_pos = self.chain.forward_kinematics(
			current_joint.view(-1,self.franka_hand_index),
			world=pk.Transform3d(
				pos=[0,-0.5,0.4],
				rot=[0,0,np.pi/2],device=self.device)
		).get_matrix()[:,:3,3].view(self.cfg.num_envs, self.cfg.num_robots, 3)+dpose[:,:,:3,0]
		rot = torch.tensor(
			[
				[0,-1,0,0,0,0],
				[1,0,0,0,0,0],
				[0,0,1,0,0,0],
				[0,0,0,0,-1,0],
				[0,0,0,1,0,0],
				[0,0,0,0,0,1],
			],
			dtype=torch.float,
			device=self.device)
		while True:
			j_eef = rot@self.chain.jacobian(
				current_joint.view(-1,self.franka_hand_index),
				).view(self.cfg.num_envs,self.cfg.num_robots,6,self.franka_hand_index) 
			j_eef_T = j_eef.transpose(-2,-1)
			current_joint = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda)@ dpose).view(self.cfg.num_envs, self.cfg.num_robots, self.franka_hand_index) + current_joint
			current_ee_trans = self.chain.forward_kinematics(
				current_joint.view(-1,self.franka_hand_index),
				world=pk.Transform3d(
					pos=[0,-0.5,0.4],
					rot=[0,0,np.pi/2],device=self.device)
			)
			current_ee_pos = current_ee_trans.get_matrix()[:, :3, 3]
			current_ee_rot = pk.matrix_to_quaternion(current_ee_trans.get_matrix()[:, :3, :3])
			current_ee_rot = torch.cat((current_ee_rot[...,1:4], current_ee_rot[...,:1]),dim=-1)
			orn_errs = self.orientation_error(self.franka_default_orn, current_ee_rot.view(self.cfg.num_envs, self.cfg.num_robots, 4))
			pos_errs = target_ee_pos - current_ee_pos 
			dpose = torch.cat([pos_errs, orn_errs], -1).unsqueeze(-1)
			err = torch.norm(dpose.squeeze(-1),dim=-1)
			num_it += 1
			# print(num_it, current_ee_pos, target_ee_pos)
			if err < self.cfg.ik_err or num_it > self.cfg.max_ik_iter:
				break
		return current_joint

	def control_ik_old(self, dpose):
		# solve damped least squares
		j_eefs = torch.stack(self.j_eefs).transpose(0,1)
		j_eef_T = torch.transpose(j_eefs, -2, -1)
		lmbda = torch.eye(6, device=self.device) * (self.cfg.damping ** 2)
		u = (j_eef_T @ torch.inverse(j_eefs @ j_eef_T + lmbda)
				 @ dpose).view(self.cfg.num_envs, self.cfg.num_robots, self.franka_hand_index)
		return u+self.franka_dof_poses[...,:self.franka_hand_index]

	def orientation_error(self, desired, current):
		cc = quat_conjugate(current)
		q_r = quat_mul(desired, cc)
		return q_r[..., 0:3] * torch.sign(q_r[..., 3]).unsqueeze(-1)

	def ezpolicy(self, obs):
		assert self.cfg.num_goals == 1, "ezpolicy only works for 1 goal"
		pos = obs[..., :3]*self.single_goal_std+self.single_goal_mean + self.origin_shift
		obj = obs[..., 17:20].view(
			self.cfg.num_envs, self.cfg.num_goals, 3)*self.goal_std+self.goal_mean
		goal = obs[..., 20:23].view(
			self.cfg.num_envs, self.cfg.num_goals, 3)*self.goal_std+self.goal_mean
		if self.cfg.num_robots == 1:
			up_step = 6
			reach_step = 26
			grasp_step = 30
			end_step = 80
			action = torch.zeros((self.cfg.num_envs, 4),
													device=self.device, dtype=torch.float32)
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
						action[env_id, 2:4] = 1
					elif up_step <= self.progress_buf[env_id] < reach_step:
						# action[env_id, :3] = (torch.tensor([-0.15,0,0.05],device=self.device) - pos_now)
						action[env_id, :3] = (obj_now - pos_now)*16
						action[env_id, 3] = 1
					elif reach_step <= self.progress_buf[env_id] < grasp_step:
						action[env_id, 3] = -1
					elif grasp_step <= self.progress_buf[env_id] < end_step:
						action[env_id, :3] = (goal_now - obj_now)*16
						action[env_id, 3] = -1
		elif self.cfg.num_robots == 2:
			up_step = 6
			reach_step = 26
			grasp_step = 30
			move_to_other_step =60
			close_both_step=65
			open_grip_step=70
			move_to_goal_step=90
			obs_dict = self.obs_parser(obs[0])
			ag = obs_dict.ag[0,0]*self.goal_std+self.goal_mean
			g = obs_dict.g*self.goal_std+self.goal_mean
			robot0 = obs_dict.shared[:3]*self.goal_std+self.goal_mean
			robot1 = obs_dict.shared[3:6]*self.goal_std+self.goal_mean
			g_side= g[0] > 0
			if self.progress_buf[0] < 6:
				self.ag_side= ag[0] > 0
			action = torch.zeros((2, 4),
													device=self.device, dtype=torch.float32)
			if g_side and self.ag_side:
				if self.progress_buf[0] < up_step:
					action[1, 2] = 1
					action[1, 3] = 1
				elif self.progress_buf[0] < reach_step:
					delta = ag - robot1
					action[1, :3] = delta*20
				elif self.progress_buf[0] < grasp_step:
					action[1, 3] = -1
				elif self.progress_buf[0] < move_to_other_step:
					delta =  g - ag
					action[1, :3] = delta*20
					action[1, 3] = -1
			elif (not g_side) and (not self.ag_side):
				if self.progress_buf[0] < up_step:
					action[0, 2] = 1
					action[0, 3] = 1
				elif self.progress_buf[0] < reach_step:
					delta = ag - robot0
					action[0, :3] = delta*20
				elif self.progress_buf[0] < grasp_step:
					action[0, 3] = -1
				elif self.progress_buf[0] < move_to_other_step:
					delta =  g - ag
					action[0, :3] = delta*20
					action[0, 3] = -1
			elif g_side and (not self.ag_side):
				if self.progress_buf[0] < up_step: # move up
					action[0, 2] = 1
					action[0, 3] = 1
				elif self.progress_buf[0] < reach_step: # reach obj
					delta = ag - robot0 + torch.tensor([-0.05,0,0], device=self.device)
					action[0, :3] = delta*20
				elif self.progress_buf[0] < grasp_step:
					action[0, 3] = -1
				elif self.progress_buf[0] < move_to_other_step: # move to gap
					delta0 =  torch.tensor([-0.05,0,0.5], device=self.device)- robot0
					action[0, :3] = delta0*20
					action[0, 3] = -1
					delta1 =  torch.tensor([0.05,0,0.5], device=self.device)- robot1
					action[1, :3] = delta1*20
					action[1, 3] = 1
				elif self.progress_buf[0] < close_both_step: # close both
					action[:,3]=-1
				elif self.progress_buf[0] < open_grip_step: # open left
					action[0,3]=1
					action[1,3]=-1
				elif self.progress_buf[0] < move_to_goal_step: # move to goal
					delta = g - ag
					action[1, :3] = delta*20
					action[1,3]=-1
			elif (not g_side) and self.ag_side:
				if self.progress_buf[0] < up_step: # move up
					action[1, 2] = 1
					action[1, 3] = 1
				elif self.progress_buf[0] < reach_step: # reach obj
					delta = ag - robot1 + torch.tensor([0.05,0,0], device=self.device)
					action[1, :3] = delta*20
				elif self.progress_buf[0] < grasp_step:
					action[1, 3] = -1
				elif self.progress_buf[0] < move_to_other_step: # move to gap
					delta0 =  torch.tensor([-0.05,0,0.3], device=self.device)- robot0
					action[0, :3] = delta0*20
					action[0, 3] = 1
					delta1 =  torch.tensor([0.05,0,0.3], device=self.device)- robot1
					action[1, :3] = delta1*20
					action[1, 3] = -1
				elif self.progress_buf[0] < close_both_step: # close both
					action[:,3]=-1
				elif self.progress_buf[0] < open_grip_step: # open left
					action[0,3]=-1
					action[1,3]=1
				elif self.progress_buf[0] < move_to_goal_step: # move to goal
					delta = g - ag
					action[0, :3] = delta*20
					action[0,3]=-1
		else:
			raise NotImplementedError("ezpolicy only works for 1 or 2 robots")
		return action - self.cfg.action_shift

	def update_config(self, cfg):
		cfg.update(enable_camera_sensors=cfg.num_cameras > 0)
		if cfg.enable_robot_id:
			cfg.per_shared_dim += 1
		cfg.update(
			# dim
			action_dim=cfg.per_action_dim * cfg.num_robots,
			shared_dim=cfg.per_shared_dim * cfg.num_robots, 
			state_dim=cfg.per_shared_dim * cfg.num_robots + cfg.per_seperate_dim * \
			cfg.num_goals + cfg.per_goal_dim*cfg.num_goals,
			info_dim=cfg.info_dim + cfg.num_goals*cfg.per_goal_dim + cfg.num_robots*3 + cfg.num_goals*3,
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
				ag = obs[..., self.cfg.shared_dim:self.cfg.shared_dim+self.cfg.seperate_dim].view(-1,self.cfg.num_goals, self.cfg.per_seperate_dim)[:,:,-self.cfg.goal_dim:], 
				g = obs[..., self.cfg.shared_dim+self.cfg.seperate_dim:]
			)
		elif name == 'shared':
			return obs[..., :self.cfg.shared_dim]
		elif name == 'seperate':
			return obs[..., self.cfg.shared_dim:self.cfg.shared_dim+self.cfg.seperate_dim]
		elif name == 'ag':
			return obs[..., self.cfg.shared_dim:self.cfg.shared_dim+self.cfg.seperate_dim].view(-1,self.cfg.num_goals, self.cfg.per_seperate_dim)[:,:,-self.cfg.goal_dim:]
		elif name == 'g':
			return obs[..., self.cfg.shared_dim+self.cfg.seperate_dim:] 
		else:
			raise NotImplementedError
	
	def obs_mirror(self, obs):
		return obs @ self.obs_rot_mat

	def act_mirror(self, act):
		return act @ self.act_rot_mat
	
	def obs_updater(self, old_obs, new_obs:AttrDict):
		if 'shared' in new_obs:
			old_obs[..., :self.cfg.shared_dim] = new_obs.shared	
		if 'seperate' in new_obs:
			old_obs[..., self.cfg.shared_dim:self.cfg.shared_dim+self.cfg.seperate_dim] = new_obs.seperate
		if 'ag' in new_obs:
			old_obs[..., self.cfg.shared_dim:self.cfg.shared_dim+self.cfg.seperate_dim].view(-1,self.cfg.num_goals, self.cfg.single_seperate_dim)[:,:,-self.cfg.goal_dim:] = new_obs.ag
		if 'g' in new_obs:
			old_obs[..., self.cfg.shared_dim+self.cfg.seperate_dim:] = new_obs.g
		return old_obs

	def info_parser(self, info, name = None):
		assert info.shape[-1] == self.cfg.info_dim, f'info {self.cfg.info_dim} shape error: {info.shape}' 
		if name is None:
			grip_pos_start = 6+self.cfg.goal_dim
			reached_ag_start = grip_pos_start + self.cfg.num_robots*3
			ag_unmoved_steps_start = reached_ag_start + self.cfg.num_goals
			goal_mask_start = ag_unmoved_steps_start + self.cfg.num_goals
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
				ag_unmoved_steps = info[...,ag_unmoved_steps_start:goal_mask_start], 
				goal_mask = info[...,goal_mask_start:goal_mask_start+self.cfg.num_goals], 
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
			return info[..., 6+self.cfg.goal_dim:6+self.cfg.goal_dim+self.cfg.num_robots*3]
		elif name == 'reached_ag':
			return info[..., 6+self.cfg.goal_dim+self.cfg.num_robots*3:6+self.cfg.goal_dim+self.cfg.num_robots*3+self.cfg.num_goals]
		elif name == 'ag_unmoved_steps':
			return info[..., 6+self.cfg.goal_dim+self.cfg.num_robots*3+self.cfg.num_goals:6+self.cfg.goal_dim+self.cfg.num_robots*3+self.cfg.num_goals*2]
		elif name == 'goal_mask':
			return info[..., 6+self.cfg.goal_dim+self.cfg.num_robots*3+self.cfg.num_goals*2:6+self.cfg.goal_dim+self.cfg.num_robots*3+self.cfg.num_goals*3].bool()
		else:
			return None

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
			old_info[..., 6+self.cfg.num_goals*3:6+self.cfg.num_goals*3+self.cfg.num_robots*3] = new_info.grip_pos
		if 'reached_ag' in new_info:
			old_info[..., 6+self.cfg.goal_dim+self.cfg.num_robots*3:6+self.cfg.goal_dim+self.cfg.num_robots*3+self.cfg.num_goals] = new_info.reached_ag
		if 'ag_unmoved_steps' in new_info:
			old_info[..., 6+self.cfg.goal_dim+self.cfg.num_robots*3+self.cfg.num_goals:6+self.cfg.goal_dim+self.cfg.num_robots*3+self.cfg.num_goals*2] = new_info.ag_unmoved_steps
		if 'goal_mask' in new_info:
			old_info[..., 6+self.cfg.goal_dim+self.cfg.num_robots*3+self.cfg.num_goals*2:6+self.cfg.goal_dim+self.cfg.num_robots*3+self.cfg.num_goals*3] = new_info.goal_mask
		return old_info

	def sample_goal(self, size, norm = True, change_ws=False, g_origin=None):
		if change_ws:
			init_ws = (g_origin[..., 0] > 0).long()
			goal_workspace = (init_ws+1)%2
		else:
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
			num_robots = self.cfg.num_robots, 
			per_action_dim=self.cfg.per_action_dim,
			action_dim=self.cfg.action_dim,
			state_dim=self.cfg.state_dim,
			per_shared_dim=self.cfg.per_shared_dim, 
			shared_dim=self.cfg.shared_dim,
			seperate_dim=self.cfg.seperate_dim,
			goal_dim=self.cfg.goal_dim,
			info_dim=self.cfg.info_dim,  # is_success, step, achieved_goal
			# numbers
			num_goals=self.cfg.num_goals,
			num_envs=self.cfg.num_envs,
			max_env_step=self.cfg.max_steps,
			early_termin_step=self.cfg.early_termin_step,
			# rot matrixs
			obs_rot_mat = self.obs_rot_mat, 
			act_rot_mat = self.act_rot_mat, 
			dual_act_rot_mat = self.dual_act_rot_mat, 
			single_act_rot_mat = self.single_act_rot_mat, 
			last_act_rot_mat = self.last_act_rot_mat,
			robot_reshape_mat = self.robot_reshape_mat,
			other_robot_obs_mask = self.other_robot_obs_mask,
			# functions
			sample_goal=self.sample_goal, 
			compute_reward=self.compute_reward,
			info_parser=self.info_parser,
			info_updater=self.info_updater,
			obs_parser=self.obs_parser, 
			obs_updater=self.obs_updater,
			obs_mirror=self.obs_mirror,
		)


gym.register(id='FrankaPNP-v0', entry_point=FrankaCube)

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
	env = gym.make('FrankaPNP-v0', num_envs=1, num_robots=2, num_cameras=0, headless=False, bound_robot=True, sim_device_id=0, rl_device_id=0, num_goals=8, current_num_goals=3, os_rate=1.0, max_handover_time=2, inhand_rate=0, table_gap=0.0, base_step=1, early_termin_step=10, extra_goal_sample=100, max_sample_time=200)
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
				# act[..., 7] = -1
				# act[..., 4] = -1
			elif args.ezpolicy:
				obs = env.obs_mirror(obs)
				act = env.ezpolicy(obs)
				act = act.flatten()@env.act_rot_mat
			else:
				act = torch.tensor([args.action]*env.cfg.num_envs, device=env.device)
				# act = torch.tensor([action_list[j%16]]*env.cfg.num_robots*env.cfg.num_envs, device=env.device)
			obs, rew, done, info = env.step(act)
			# env.render(mode='human')
			# print(info_dict.step.item())
		# Image.fromarray(images[0]).save('foo.png')

	print(time.time()-start)
	env.close()