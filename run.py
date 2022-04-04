import isaacgym
import torch
import numpy as np
from attrdict import AttrDict
import yaml
import json
import argparse
import wandb

import agent
import envs


def train(config):
	torch.set_grad_enabled(False)

	'''init'''
	exp_agent:agent.AgentBase = getattr(agent, config.agent_name)(config)
	print('explore...')
	result = exp_agent.explore_vec_env()
	print(result)

	'''start training'''
	num_rollouts = config.max_collect_steps//config.steps_per_rollout
	for _ in range(num_rollouts):  # TODO fix it
		print('explore...')
		result = exp_agent.explore_vec_env()
		print(result)

		print('update...')
		torch.set_grad_enabled(True)
		result = exp_agent.update_net()
		torch.set_grad_enabled(False)
		print(result)

		print('eval...')
		result = exp_agent.eval_vec_env()
		print(result)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config-file', type=str,
											default='sac', help='config file')
	parser.add_argument('--kwargs', type=json.loads, default={})
	args = parser.parse_args()
	with open(f'configs/{args.config_file}.yaml', "r") as stream:
		try:
			config = AttrDict(yaml.safe_load(stream))
			config.update(args.kwargs)
		except yaml.YAMLError as exc:
			print(exc)

	# start run
	# wandb.init(project='debug', config=config)
	train(config)
	# wandb.join()
