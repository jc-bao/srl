import isaacgym
import torch
import numpy as np

from config import build_env, Arguments
from agent import AgentSAC, AgentModSAC, AgentREDqSAC, AgentDDPG
from replay_buffer import ReplayBuffer, ReplayBufferList
from envs import ReachToyEnv, PNPToyEnv, HandoverToyEnv

def train(args):
	torch.set_grad_enabled(False)
	args.init_before_training()
	gpu_id = args.learner_gpus

	'''init'''
	env = build_env(args.env, args.env_func, args.env_args)
	args.reward_fn = env.compute_reward

	agent = init_agent(args, gpu_id, env)
	buffer = init_buffer(args, gpu_id)

	agent.state = env.reset()
	if args.if_off_policy:
		print('explore...')
		total_steps, useless_steps = agent.explore_env(env, args.target_steps_per_env, buffer = buffer)
		print(useless_steps/total_steps)
		# buffer.update_buffer((trajectory,))

	'''start training'''
	target_steps_per_env = args.target_steps_per_env
	del args

	for _ in range(10000):  # TODO fix it
		# print('explore...')
		total_steps, useless_steps = agent.explore_env(env, target_steps_per_env, buffer = buffer)
		# print(useless_steps/total_steps)

		# print('update...')
		torch.set_grad_enabled(True)
		logging_tuple = agent.update_net(buffer)
		torch.set_grad_enabled(False)

		print(logging_tuple)

		# print('eval...')
		print(agent.evaluate_save(env))


def init_agent(args, gpu_id, env=None):
	agent = args.agent(args.net_dim, args.state_dim,
										 args.action_dim, max_env_step = args.max_env_step, goal_dim = args.goal_dim,
										 info_dim=args.info_dim, gpu_id=gpu_id, args=args)
	agent.save_or_load_agent(args.cwd, if_save=False)

	if env is not None:
		'''assign `agent.states` for exploration'''
		if args.env_num == 1:
			states = [env.reset(), ]
			assert isinstance(states[0], np.ndarray)
			assert states[0].shape in {(args.state_dim,), args.state_dim}
		else:
			states = env.reset()
			assert isinstance(states, torch.Tensor)
			assert states.shape == (args.env_num, args.state_dim)
		agent.states = states
	return agent


def init_buffer(args, gpu_id):
	if args.if_off_policy:
		buffer = ReplayBuffer(gpu_id=gpu_id,
													max_len=args.max_memo,
													state_dim=args.state_dim,
													action_dim=1 if args.if_discrete else args.action_dim, 
													goal_dim = args.goal_dim, 
													info_dim = args.info_dim, 
													reward_fn = args.reward_fn)
		buffer.save_or_load_history(args.cwd, if_save=False)

	else:
		buffer = ReplayBufferList()
	return buffer


if __name__ == '__main__':
	env_func = HandoverToyEnv 
	env_args = {
		# 'env_num': 2**8, 
		'env_num': 2**10, 
		'max_step': 100, 
		'env_name': 'PNPToy-v0',
		'state_dim': 10, # obs+goal
		'goal_dim': 2, 
		'info_dim': 4+4,
		'action_dim': 6
		,
		'if_discrete': False,
		'target_return': 0, 
		'err': 0.2,
		'vel': 0.2,
		'gpu_id': 0,
	}
	args = Arguments(agent=AgentREDqSAC, env_func=env_func, env_args = env_args)
	train(args)
