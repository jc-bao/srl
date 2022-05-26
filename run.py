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
			file_tag=config.load_tag, cwd=config.load_path, if_save=False)
	elif (config.wid is not None):
		print('[Load] resume from cloud')
		exp_agent.save_or_load_agent(file_tag=config.load_tag, if_save=False)

	def log(msg, prefix=''):
		print(msg)
		if config.wandb:
			wandb.log({f'{prefix}/{k}': v for k, v in msg.items()}, step=exp_agent.total_step)
			if 'curri' in msg:
				curri_params = msg.curri
				wandb.log({f'curri/{k}': v for k, v in curri_params.items()}, step=exp_agent.total_step)
			if msg.get('video') is not None:
				wandb.log({"Media/video": wandb.Video(msg.video, fps=10, format="mp4")})
	torch.set_grad_enabled(False)

	# warmup
	print('warm up...')
	result = exp_agent.explore_vec_env()
	log(result, prefix='explore')

	'''start training'''
	best_rew = -1000
	num_rollouts = int(config.max_collect_steps//config.steps_per_rollout)
	for i in range(num_rollouts):
		print(f'========explore{i}==========')
		result = exp_agent.explore_vec_env()
		log(result, prefix='explore')

		print(f'========update{i}...========')
		torch.set_grad_enabled(True)
		result = exp_agent.update_net()
		torch.set_grad_enabled(False)
		log(result, prefix='update')

		if i % int(1/config.eval_per_rollout) == 0:
			num_eval = i // int(1/config.eval_per_rollout)
			print(f'========eval{num_eval}...===========')
			result = exp_agent.eval_vec_env(
				render=(num_eval % int(1/config.render_per_eval) == 0))
			log(result, prefix=f'eval_{exp_agent.EP.num_goals}')

			if (i % int(1/config.eval_per_save)) == 0:
				# if result.final_rew > best_rew: 
				# 	best_rew = result.final_rew
				# 	tag = 'best'
				# 	if exp_agent.cfg.curri is not None:
				# 		for k,v in exp_agent.cfg.curri.items():
				# 			tag += f'_{k}={v.now}'
				# 	exp_agent.save_or_load_agent(file_tag=tag, if_save=True)
				# else:
				exp_agent.save_or_load_agent(file_tag='latest', if_save=True)
				print('=========saved!')
			


if __name__ == '__main__':
	train()
