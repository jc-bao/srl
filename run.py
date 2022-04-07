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

	'''init'''
	if config.wandb:
		wandb.init(name=config.name, project=config.project, config=config)
	exp_agent:agent.AgentBase = getattr(agent, config.agent_name)(config)
	def log(msg):
		print(msg)
		if config.wandb:
			wandb.log(msg, step=exp_agent.total_step.item())
			if msg.get('video') is not None:
				wandb.log({"video": wandb.Video(msg.video, fps=30, format="mp4")})
	torch.set_grad_enabled(False)

	# warmup
	print('explore...')
	result = exp_agent.explore_vec_env()
	log(result)

	'''start training'''
	best_rew = -1000
	num_rollouts = int(config.max_collect_steps//config.steps_per_rollout)
	for i in range(num_rollouts):
		print('========explore...')
		result = exp_agent.explore_vec_env()
		log(result)

		print('========update...')
		torch.set_grad_enabled(True)
		result = exp_agent.update_net()
		torch.set_grad_enabled(False)
		log(result)

		if i % int(1/config.eval_per_rollout) == 0:
			print('========eval...')
			result = exp_agent.eval_vec_env()
			log(result)

			if result.final_rew > best_rew and (i%config.rollout_per_save)==0:
				best_rew = result.final_rew 
				exp_agent.save_or_load_agent(file_tag = f'rew{best_rew:.2f}', if_save=True)
				exp_agent.save_or_load_agent(file_tag = 'best', if_save=True)
				print('=========saved!')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', type=str,
											default='sac', help='config file')
	parser.add_argument('-k', '--kwargs', type=json.loads, default={})
	args = parser.parse_args()
	with open(f'configs/{args.file}.yaml', "r") as stream:
		try:
			config = AttrDict(yaml.safe_load(stream))
			config.update(args.kwargs)
		except yaml.YAMLError as exc:
			print(exc)

	# start run
	train(config)