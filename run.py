import isaacgym
import torch
import wandb
import hydra
from attrdict import AttrDict

import agent
from envs.franka_cube import FrankaCube
from envs.toy import ReachToyEnv


@hydra.main(config_name='main', config_path='configs')
def train(config):
	config = AttrDict(config)
	'''init'''
	if config.wandb:
		if config.resume_mode is None or config.resume_mode == 'restart':
			print('[Wandb] start new run...')
			wandb.init(name=config.name, project=config.project, config=config)
		elif config.resume_mode == 'load_only':
			print('[Wandb] start load only mode... set wandb to false')
			config.wandb = False
		elif config.resume_mode == 'continue':
			print('[Wandb] resume old run...')
			wandb.init(project=config.project, id=config.wid,
								 resume="allow", config=config)
		else:
			raise NotImplementedError(
				'[Wandb] resume mode {} not implemented'.format(config.resume_mode))
	exp_agent: agent.AgentBase = getattr(agent, config.agent_name)(config)
	if config.load_path is not None:
		print('[Load] resume from local')
		exp_agent.save_or_load_agent(
			file_tag='best', cwd=config.load_path, if_save=False)
	elif (config.wid is not None):
		print('[Load] resume from cloud')
		exp_agent.save_or_load_agent(file_tag='best', if_save=False)
		# exp_agent.save_or_load_agent(file_tag = 'rew-0.08', if_save=False)

	def log(msg):
		print(msg)
		if config.wandb:
			wandb.log(msg, step=exp_agent.total_step)
			if msg.get('video') is not None:
				wandb.log({"Media/video": wandb.Video(msg.video, fps=30, format="mp4")})
	torch.set_grad_enabled(False)

	# warmup
	print('warm up...')
	result = exp_agent.explore_vec_env()
	log(result)

	'''start training'''
	best_rew = -1000
	num_rollouts = int(config.max_collect_steps//config.steps_per_rollout)
	for i in range(num_rollouts):
		print(f'========explore{i}==========')
		result = exp_agent.explore_vec_env()
		log(result)

		print(f'========update{i}...========')
		torch.set_grad_enabled(True)
		result = exp_agent.update_net()
		torch.set_grad_enabled(False)
		log(result)

		if i % int(1/config.eval_per_rollout) == 0:
			num_eval = i // int(1/config.eval_per_rollout)
			print(f'========eval{num_eval}...===========')
			result = exp_agent.eval_vec_env(
				render=(num_eval % int(1/config.render_per_eval) == 0))
			log(result)

			if result.final_rew > best_rew and (i % int(1/config.rollout_per_save)) == 0:
				best_rew = result.final_rew
				exp_agent.save_or_load_agent(
					file_tag=f'rew{best_rew:.2f}', if_save=True)
				exp_agent.save_or_load_agent(file_tag='best', if_save=True)
				print('=========saved!')


if __name__ == '__main__':
	train()

'''
import numpy as np
import yaml
import json
import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', type=str,
											default='sac', help='config file')
	parser.add_argument('-k', '--kwargs', type=json.loads, default={})
	parser.add_argument('-e', '--envargs', type=json.loads, default={})
	args = parser.parse_args()
	with open(f'configs/{args.file}.yaml', "r") as stream:
		try:
			config = AttrDict(yaml.safe_load(stream))
			config.update(args.kwargs)
			config['env_kwargs'].update(args.envargs)
		except yaml.YAMLError as exc:
			print(exc)

	# start run
	train(config)
'''
