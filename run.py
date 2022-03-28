import isaacgym
import torch
import numpy as np

from config import build_env, Arguments
from agent import AgentSAC
from replay_buffer import ReplayBuffer, ReplayBufferList
from envs import ReachToyEnv

def train(args):
	torch.set_grad_enabled(False)
	args.init_before_training()
	gpu_id = args.learner_gpus

	'''init'''
	env = build_env(args.env, args.env_func, args.env_args)

	agent = init_agent(args, gpu_id, env)
	buffer = init_buffer(args, gpu_id)

	agent.state = env.reset()
	if args.if_off_policy:
		trajectory = agent.explore_env(env, args.target_step)
		buffer.update_buffer((trajectory,))

	'''start training'''
	target_step = args.target_step
	del args

	for _ in range(100):  # TODO fix it
		trajectory = agent.explore_env(env, target_step)
		steps, r_exp = buffer.update_buffer((trajectory,))

		torch.set_grad_enabled(True)
		logging_tuple = agent.update_net(buffer)
		torch.set_grad_enabled(False)
		# TODO add eval here
		print(steps, r_exp, logging_tuple)


def init_agent(args, gpu_id, env=None):
	agent = args.agent(args.net_dim, args.state_dim,
										 args.action_dim, gpu_id=gpu_id, args=args)
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
													action_dim=1 if args.if_discrete else args.action_dim, )
		buffer.save_or_load_history(args.cwd, if_save=False)

	else:
		buffer = ReplayBufferList()
	return buffer


if __name__ == '__main__':
	env_func = ReachToyEnv 
	env_args = {
		'env_num': 4, 
		'max_step': 20, 
		'env_name': 'ReachToy-v0',
		'state_dim': 4, # obs+goal
		'goal_dim': 2, 
		'action_dim': 2,
		'if_discrete': False,
		'target_return': 0, 
	}
	args = Arguments(agent=AgentSAC, env_func=env_func, env_args = env_args)
	train(args)
