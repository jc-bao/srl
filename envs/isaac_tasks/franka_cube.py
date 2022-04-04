import numpy as np
import os
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
import torch
from attrdict import AttrDict
import torchgeometry as tgm

from base.vec_task import VecTask

'''
TODO
1. init pos
		[x] get hand pos and ori (rigid body)
		[ ] 
2. observation
		[ ] goal setup (sphere)
3. action
		[ ] control eef
4. pick and place policy test
		[ ] naive policy
'''


class FrankaCube(VecTask):
	def __init__(self, cfg, sim_device, graphics_device_id, headless):
		self.cfg = cfg

		# gym params
		self.max_episode_length = self.cfg["env"]["episodeLength"]
		self.action_scale = self.cfg["env"]["actionScale"]

		# isaac params
		self.aggregate_mode = self.cfg["env"]["aggregateMode"]  # create group

		# env params
		# noise
		self.start_position_noise = self.cfg["env"]["startPositionNoise"]
		self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
		self.num_blocks = self.cfg["env"]["numblocks"]
		# block size
		self.block_width = 0.08
		self.block_height = 0.08
		self.block_length = 0.08
		self.block_spacing = 0.09
		# reward
		self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
		self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
		# axis
		self.up_axis = "z"
		self.up_axis_idx = 2
		# step
		self.dt = 1 / 60.0
		# action
		self.cfg["env"]["numActions"] = 4
		# observation
		self.cfg["env"]["numObservations"] = 23

		# robot params
		self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
		self.open_reward_scale = self.cfg["env"]["openRewardScale"]
		self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
		self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
		# IK
		self.ikconfig = AttrDict(
			damping=0.05,
		)

		# camera params
		self.debug_viz = self.cfg["env"]["enableDebugVis"]
		self.distX_offset = 0.04

		# Note: setup tensor after call the father class
		super().__init__(
			config=self.cfg,
			sim_device=sim_device,
			graphics_device_id=graphics_device_id,
			headless=headless,
		)

		# get basic params
		actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(
			self.sim)  # base pos
		self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
			self.num_envs, -1, 13)
		dof_state_tensor = self.gym.acquire_dof_state_tensor(
			self.sim)  # joint pos
		rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
		self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
			self.num_envs, -1, 13
		)  # object pos
		self.num_bodies = self.rigid_body_states.shape[1]
		self.franka_default_dof_pos = to_torch(
			[0, 0, 0, -1.57, 0, 1.82, 0, 0.035, 0.035],
			device=self.device,
		)
		self.franka_default_orn = to_torch(
			[[0., 1., 0., 0.]], 
			device = self.device
		).repeat(self.num_envs,1)
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

		# robot observation
		self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
		self.hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
		self.hand_vel = self.rigid_body_states[:, self.hand_handle][:, 7:10]
		self.finger_width = self.rigid_body_states[:, self.lfinger_handle][:,
																																	0:3]+self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
		# object observation
		self.block_states = self.rigid_body_states[:, self.block_handles]

		# store for visualization
		self.franka_lfinger_pos = self.rigid_body_states[:,
																										 self.lfinger_handle][:, 0:3]
		self.franka_rfinger_pos = self.rigid_body_states[:,
																										 self.rfinger_handle][:, 0:3]
		self.franka_lfinger_rot = self.rigid_body_states[:,
																										 self.lfinger_handle][:, 3:7]
		self.franka_rfinger_rot = self.rigid_body_states[:,
																										 self.rfinger_handle][:, 3:7]

		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_rigid_body_state_tensor(self.sim)
		self.gym.refresh_jacobian_tensors(self.sim)

		# block base pos
		if self.num_blocks > 0:
			self.block_states = self.root_state_tensor[:, 1:]

		# block
		self.global_indices = torch.arange(
			self.num_envs * (1 + self.num_blocks), dtype=torch.int32, device=self.device
		).view(self.num_envs, -1)

		self.reset_idx(torch.arange(self.num_envs, device=self.device))

	def create_sim(self):
		self.sim_params.up_axis = gymapi.UP_AXIS_Z
		self.sim_params.gravity.x = 0
		self.sim_params.gravity.y = 0
		self.sim_params.gravity.z = -9.81
		self.sim = super().create_sim(
			self.device_id,
			self.graphics_device_id,
			self.physics_engine,
			self.sim_params,
		)
		self._create_ground_plane()
		self._create_envs(
			self.num_envs, self.cfg["env"]["envSpacing"], int(
				np.sqrt(self.num_envs))
		)

	def _create_ground_plane(self):
		plane_params = gymapi.PlaneParams()
		plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
		self.gym.add_ground(self.sim, plane_params)

	def _create_envs(self, num_envs, spacing, num_per_row):
		lower = gymapi.Vec3(-spacing, -spacing, 0.0)
		upper = gymapi.Vec3(spacing, spacing, spacing)

		asset_root = os.path.join(
			os.path.dirname(os.path.abspath(__file__)), "../isaac_assets"
		)
		franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

		if "asset" in self.cfg["env"]:
			asset_root = os.path.join(
				os.path.dirname(os.path.abspath(__file__)),
				self.cfg["env"]["asset"].get("assetRoot", asset_root),
			)
			franka_asset_file = self.cfg["env"]["asset"].get(
				"assetFileNameFranka", franka_asset_file
			)

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
			if self.physics_engine == gymapi.SIM_PHYSX:
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
			self.sim, self.block_width, self.block_height, self.block_width, box_opts
		)

		franka_start_pose = gymapi.Transform()
		franka_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
		franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

		# compute aggregate size
		num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
		num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
		num_block_bodies = self.gym.get_asset_rigid_body_count(block_asset)
		num_block_shapes = self.gym.get_asset_rigid_shape_count(block_asset)
		max_agg_bodies = (
			num_franka_bodies + self.num_blocks * num_block_bodies
		)
		max_agg_shapes = (
			num_franka_shapes + self.num_blocks * num_block_shapes
		)

		self.frankas = []
		self.default_block_states = []
		self.prop_start = []
		self.envs = []

		for i in range(self.num_envs):
			# create env instance
			env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

			if self.aggregate_mode >= 3:
				self.gym.begin_aggregate(
					env_ptr, max_agg_bodies, max_agg_shapes, True)

			# Key: create Panda
			franka_actor = self.gym.create_actor(
				env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0
			)
			self.gym.set_actor_dof_properties(
				env_ptr, franka_actor, franka_dof_props)

			if self.aggregate_mode == 2:
				self.gym.begin_aggregate(
					env_ptr, max_agg_bodies, max_agg_shapes, True)

			if self.aggregate_mode == 1:
				self.gym.begin_aggregate(
					env_ptr, max_agg_bodies, max_agg_shapes, True)

			if self.num_blocks > 0:
				self.prop_start.append(self.gym.get_sim_actor_count(self.sim))

				xmin = -0.5 * self.block_spacing * (self.num_blocks - 1)
				self.block_handles = []
				for j in range(self.num_blocks):
					block_state_pose = gymapi.Transform()
					block_state_pose.p.x = xmin + j * self.block_spacing
					block_state_pose.p.y = 0
					block_state_pose.p.z = 0
					block_state_pose.r = gymapi.Quat(0, 0, 0, 1)
					self.block_handles.append(self.gym.create_actor(
						env_ptr, block_asset, block_state_pose, "block{}".format(
							j), i, 0, 0,
					))
					self.default_block_states.append(
						[block_state_pose.p.x, block_state_pose.p.y, block_state_pose.p.z,
						 block_state_pose.r.x, block_state_pose.r.y, block_state_pose.r.z,
						 block_state_pose.r.w, ] + [0]*6
					)
			if self.aggregate_mode > 0:
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
		).view(self.num_envs, self.num_blocks, 13)
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

		self.franka_grasp_pos = torch.zeros_like(self.franka_local_grasp_pos)
		self.franka_grasp_rot = torch.zeros_like(self.franka_local_grasp_rot)
		self.franka_grasp_rot[..., -1] = 1  # xyzw
		self.franka_lfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
		self.franka_rfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
		self.franka_lfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
		self.franka_rfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)

		# dof
		self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
		_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
		self.jacobian = gymtorch.wrap_tensor(_jacobian)
		self.j_eef = self.jacobian[:,
															 self.hand_handle - 1, :, :self.franka_hand_index]

	def compute_reward(self, actions):
		self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
			self.reset_buf,
			self.progress_buf,
			self.num_envs,
			self.max_episode_length,
		)

	def compute_observations(self):
		self.obs_buf = torch.cat(
			(
				self.hand_pos, self.hand_vel, self.finger_width, # robot
				self.block_states.view(self.num_blocks, -1), # object
				# goal TODO
			),
			dim=-1
		)

		return self.obs_buf

	def reset_idx(self, env_ids):
		# reset franka
		pos = self.franka_default_dof_pos
		self.franka_dof_pos[env_ids, :] = pos
		self.franka_dof_vel[env_ids, :] = torch.zeros_like(
			self.franka_dof_vel[env_ids])
		self.franka_dof_targets[env_ids, : self.num_franka_dofs] = pos

		# reset blocks
		if self.num_blocks > 0:
			block_indices = self.global_indices[env_ids, 1:].flatten()
			self.block_states[env_ids] = self.default_block_states[env_ids]
			self.gym.set_actor_root_state_tensor_indexed(
				self.sim,
				gymtorch.unwrap_tensor(self.root_state_tensor),
				gymtorch.unwrap_tensor(block_indices),
				len(block_indices),
			)

		multi_env_ids_int32 = self.global_indices[env_ids, :1].flatten()
		self.gym.set_dof_position_target_tensor_indexed(
			self.sim,
			gymtorch.unwrap_tensor(self.franka_dof_targets),
			gymtorch.unwrap_tensor(multi_env_ids_int32),
			len(multi_env_ids_int32),
		)

		self.gym.set_dof_state_tensor_indexed(
			self.sim,
			gymtorch.unwrap_tensor(self.dof_state),
			gymtorch.unwrap_tensor(multi_env_ids_int32),
			len(multi_env_ids_int32),
		)

		self.reset_buf[env_ids] = 0
		self.progress_buf[env_ids] = 0
		self.actions = torch.zeros((self.num_envs, self.cfg['env']['numActions']))
		return self.step(torch.zeros_like(self.actions, device = self.device))

	def pre_physics_step(self, actions):
		# update data
		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_rigid_body_state_tensor(self.sim)
		self.gym.refresh_jacobian_tensors(self.sim)
		# set action here
		self.actions = actions.clone().to(self.device)
		# eef
		# orn_err = self.hand_rot - self.franka_default_orn
		# orn_err = tgm.quaternion_to_angle_axis(orn_err)
		orn_err = self.orientation_error(self.franka_default_orn, self.hand_rot)
		pos_err = self.actions[..., :3] * self.dt * self.action_scale
		dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
		self.franka_dof_targets[:, :self.franka_hand_index] = self.franka_dof_pos.squeeze(
			-1)[:, :self.franka_hand_index] + self.control_ik(dpose)
		# grip
		grip_acts = (self.actions[..., 3] + 1) * 0.02
		self.franka_dof_targets[:, self.franka_hand_index:
														self.franka_hand_index+2] = grip_acts.unsqueeze(0).repeat(2, 1)
		# limit
		self.franka_dof_targets[:, : self.num_franka_dofs] = tensor_clamp(
			self.franka_dof_targets[:,
															: self.num_franka_dofs], self.franka_dof_lower_limits, self.franka_dof_upper_limits
		)
		# Deploy actions
		self.gym.set_dof_position_target_tensor(
			self.sim, gymtorch.unwrap_tensor(self.franka_dof_targets))

	def post_physics_step(self):
		# update tensor and get observation here
		self.progress_buf += 1

		env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
		if len(env_ids) > 0:
			self.reset_idx(env_ids)

		self.compute_observations()
		self.compute_reward(self.actions)

		# debug viz
		if self.viewer and self.debug_viz:
			self.gym.clear_lines(self.viewer)

			for i in range(self.num_envs):
				px = (
					(
						self.franka_grasp_pos[i]
						+ quat_apply(
							self.franka_grasp_rot[i],
							to_torch([1, 0, 0], device=self.device) * 0.2,
						)
					)
					.cpu()
					.numpy()
				)
				py = (
					(
						self.franka_grasp_pos[i]
						+ quat_apply(
							self.franka_grasp_rot[i],
							to_torch([0, 1, 0], device=self.device) * 0.2,
						)
					)
					.cpu()
					.numpy()
				)
				pz = (
					(
						self.franka_grasp_pos[i]
						+ quat_apply(
							self.franka_grasp_rot[i],
							to_torch([0, 0, 1], device=self.device) * 0.2,
						)
					)
					.cpu()
					.numpy()
				)

				p0 = self.franka_grasp_pos[i].cpu().numpy()
				self.gym.add_lines(
					self.viewer,
					self.envs[i],
					1,
					[p0[0], p0[1], p0[2], px[0], px[1], px[2]],
					[0.85, 0.1, 0.1],
				)
				self.gym.add_lines(
					self.viewer,
					self.envs[i],
					1,
					[p0[0], p0[1], p0[2], py[0], py[1], py[2]],
					[0.1, 0.85, 0.1],
				)
				self.gym.add_lines(
					self.viewer,
					self.envs[i],
					1,
					[p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
					[0.1, 0.1, 0.85],
				)

				self.gym.add_lines(
					self.viewer,
					self.envs[i],
					1,
					[p0[0], p0[1], p0[2], px[0], px[1], px[2]],
					[1, 0, 0],
				)
				self.gym.add_lines(
					self.viewer,
					self.envs[i],
					1,
					[p0[0], p0[1], p0[2], py[0], py[1], py[2]],
					[0, 1, 0],
				)
				self.gym.add_lines(
					self.viewer,
					self.envs[i],
					1,
					[p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
					[0, 0, 1],
				)

				px = (
					(
						self.franka_lfinger_pos[i]
						+ quat_apply(
							self.franka_lfinger_rot[i],
							to_torch([1, 0, 0], device=self.device) * 0.2,
						)
					)
					.cpu()
					.numpy()
				)
				py = (
					(
						self.franka_lfinger_pos[i]
						+ quat_apply(
							self.franka_lfinger_rot[i],
							to_torch([0, 1, 0], device=self.device) * 0.2,
						)
					)
					.cpu()
					.numpy()
				)
				pz = (
					(
						self.franka_lfinger_pos[i]
						+ quat_apply(
							self.franka_lfinger_rot[i],
							to_torch([0, 0, 1], device=self.device) * 0.2,
						)
					)
					.cpu()
					.numpy()
				)

				p0 = self.franka_lfinger_pos[i].cpu().numpy()
				self.gym.add_lines(
					self.viewer,
					self.envs[i],
					1,
					[p0[0], p0[1], p0[2], px[0], px[1], px[2]],
					[1, 0, 0],
				)
				self.gym.add_lines(
					self.viewer,
					self.envs[i],
					1,
					[p0[0], p0[1], p0[2], py[0], py[1], py[2]],
					[0, 1, 0],
				)
				self.gym.add_lines(
					self.viewer,
					self.envs[i],
					1,
					[p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
					[0, 0, 1],
				)

				px = (
					(
						self.franka_rfinger_pos[i]
						+ quat_apply(
							self.franka_rfinger_rot[i],
							to_torch([1, 0, 0], device=self.device) * 0.2,
						)
					)
					.cpu()
					.numpy()
				)
				py = (
					(
						self.franka_rfinger_pos[i]
						+ quat_apply(
							self.franka_rfinger_rot[i],
							to_torch([0, 1, 0], device=self.device) * 0.2,
						)
					)
					.cpu()
					.numpy()
				)
				pz = (
					(
						self.franka_rfinger_pos[i]
						+ quat_apply(
							self.franka_rfinger_rot[i],
							to_torch([0, 0, 1], device=self.device) * 0.2,
						)
					)
					.cpu()
					.numpy()
				)

				p0 = self.franka_rfinger_pos[i].cpu().numpy()
				self.gym.add_lines(
					self.viewer,
					self.envs[i],
					1,
					[p0[0], p0[1], p0[2], px[0], px[1], px[2]],
					[1, 0, 0],
				)
				self.gym.add_lines(
					self.viewer,
					self.envs[i],
					1,
					[p0[0], p0[1], p0[2], py[0], py[1], py[2]],
					[0, 1, 0],
				)
				self.gym.add_lines(
					self.viewer,
					self.envs[i],
					1,
					[p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
					[0, 0, 1],
				)

	def control_ik(self, dpose):
		# solve damped least squares
		j_eef_T = torch.transpose(self.j_eef, 1, 2)
		lmbda = torch.eye(6, device=self.device) * (self.ikconfig.damping ** 2)
		u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda)
				 @ dpose).view(self.num_envs, self.franka_hand_index)
		return u

	def orientation_error(self, desired, current):
		cc = quat_conjugate(current)
		q_r = quat_mul(desired, cc)
		return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
		reset_buf,
		progress_buf,
		num_envs,
		max_episode_length
):
	# type: (Tensor, Tensor, int, int) -> Tuple[Tensor, Tensor]

	rewards = torch.zeros(num_envs)
	reset_buf = torch.where(
		progress_buf >= max_episode_length -
		1, torch.ones_like(reset_buf), reset_buf
	)
	return rewards, reset_buf


if __name__ == '__main__':
	'''
	run random policy
	'''
	import yaml
	with open('/home/reed/rl/srl/envs/isaac_configs/FrankaCube.yaml') as config_file:
		task_config = yaml.load(config_file, Loader=yaml.SafeLoader)
	task_config['env']['numEnvs'] = 2
	env = FrankaCube(cfg=task_config, sim_device='cuda:0',
									 graphics_device_id=0, headless=False)
	while True:
		# act = (torch.tensor([[-1,0,0,0],[-1,0,0,0]]))
		act = torch.rand((2,4))
		env.step(act)