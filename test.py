import logging
import argparse
import os
import sys
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

from rl.networks.envs import make_vec_envs
from rl.evaluation import evaluate
from rl.networks.model import Policy

from crowd_sim import *


def main():
	"""
	The main function for testing a trained model
	"""
	# the following parameters will be determined for each test run
	parser = argparse.ArgumentParser('Parse configuration file')
	# the model directory that we are testing
	parser.add_argument('--model_dir', type=str, default='trained_models/GST_predictor_rand')
	# render the environment or not
	parser.add_argument('--visualize', default=True, action='store_true')
	# if -1, it will run 500 different cases; if >=0, it will run the specified test case repeatedly
	parser.add_argument('--test_case', type=int, default=-1)
	# model weight file you want to test
	parser.add_argument('--test_model', type=str, default='41665.pt')
	# whether to save trajectories of episodes
	parser.add_argument('--render_traj', default=False, action='store_true')
	# whether to save slide show of episodes
	parser.add_argument('--save_slides', default=False, action='store_true')
	test_args = parser.parse_args()
	if test_args.save_slides:
		test_args.visualize = True

	from importlib import import_module
	model_dir_temp = test_args.model_dir
	if model_dir_temp.endswith('/'):
		model_dir_temp = model_dir_temp[:-1]
	# import arguments.py from saved directory
	# if not found, import from the default directory
	try:
		model_dir_string = model_dir_temp.replace('/', '.') + '.arguments'
		model_arguments = import_module(model_dir_string)
		get_args = getattr(model_arguments, 'get_args')
	except:
		print('Failed to get get_args function from ', test_args.model_dir, '/arguments.py')
		from arguments import get_args

	# get_args() (whichever arguments.py it came from) parses sys.argv with its own
	# PPO-specific parser, which doesn't recognize this script's own --model_dir /
	# --test_model / --test_case / --visualize / --render_traj / --save_slides --
	# already parsed above into test_args -- so it would crash on them unchanged.
	# This is pre-existing (reproduces on an unmodified checkout, for every policy
	# including this repo's own ORCA/social_force baselines), not specific to the
	# legacy_rl policies added below.
	_real_argv = sys.argv
	sys.argv = [_real_argv[0]]
	algo_args = get_args()
	sys.argv = _real_argv

	# import config class from saved directory
	# if not found, import from the default directory

	# crowd_nav/configs/config.py also calls arguments.get_args() at class-body
	# level (same reason as above) the first time it's imported anywhere in this
	# process -- which is here, since nothing above imports it.
	sys.argv = [_real_argv[0]]
	try:
		model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
		model_arguments = import_module(model_dir_string)
		Config = getattr(model_arguments, 'Config')

	except:
		print('Failed to get Config function from ', test_args.model_dir)
		from crowd_nav.configs.config import Config
	sys.argv = _real_argv
	env_config = config = Config()


	# configure logging and device
	# print test result in log file
	log_file = os.path.join(test_args.model_dir,'test')
	if not os.path.exists(log_file):
		os.mkdir(log_file)
	if test_args.visualize:
		log_file = os.path.join(test_args.model_dir, 'test', 'test_visual.log')
	else:
		log_file = os.path.join(test_args.model_dir, 'test', 'test_' + test_args.test_model + '.log')



	file_handler = logging.FileHandler(log_file, mode='w')
	stdout_handler = logging.StreamHandler(sys.stdout)
	level = logging.INFO
	logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

	logging.info('robot FOV %f', config.robot.FOV)
	logging.info('humans FOV %f', config.humans.FOV)

	torch.manual_seed(algo_args.seed)
	torch.cuda.manual_seed_all(algo_args.seed)
	if algo_args.cuda:
		if algo_args.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False


	torch.set_num_threads(1)
	device = torch.device("cuda" if algo_args.cuda else "cpu")

	logging.info('Create other envs with new settings')

	# set up visualization
	if test_args.visualize:
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.set_xlim(-6.5, 6.5) # 6
		ax.set_ylim(-6.5, 6.5)
		ax.axes.xaxis.set_visible(False)
		ax.axes.yaxis.set_visible(False)
		# ax.set_xlabel('x(m)', fontsize=16)
		# ax.set_ylabel('y(m)', fontsize=16)
		plt.ion()
		plt.show()
	else:
		ax = None


	load_path=os.path.join(test_args.model_dir,'checkpoints', test_args.test_model)
	print(load_path)


	# create an environment
	env_name = algo_args.env_name

	eval_dir = os.path.join(test_args.model_dir,'eval')
	if not os.path.exists(eval_dir):
		os.mkdir(eval_dir)

	env_config.render_traj = test_args.render_traj
	env_config.save_slides = test_args.save_slides
	env_config.save_path = os.path.join(test_args.model_dir, 'social_eval', test_args.test_model[:-3])
	envs = make_vec_envs(env_name, algo_args.seed, 1,
						 algo_args.gamma, eval_dir, device, allow_early_resets=True,
						 config=env_config, ax=ax, test_case=test_args.test_case, pretext_wrapper=config.env.use_wrapper)

	# legacy value-network RL policies ported from ChanganVR/RelationalGraphLearning
	# (crowd_nav/policy/legacy_rl/) -- like 'orca'/'social_force', they don't use the
	# PPO actor_critic at all; instead their own weights (saved via policy.save_model())
	# are loaded directly into the policy object the env already built.
	LEGACY_RL_POLICIES = ['cadrl', 'lstm_rl', 'sarl', 'model_predictive_rl']

	if config.robot.policy not in ['orca', 'social_force'] + LEGACY_RL_POLICIES:
		# load the policy weights
		actor_critic = Policy(
			envs.observation_space.spaces,
			envs.action_space,
			base_kwargs=algo_args,
			base=config.robot.policy)
		actor_critic.load_state_dict(torch.load(load_path, map_location=device))
		actor_critic.base.nenv = 1

		# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
		nn.DataParallel(actor_critic).to(device)
	else:
		actor_critic = None
		if config.robot.policy in LEGACY_RL_POLICIES:
			# reach the already-instantiated, already-configured policy object
			# inside the (possibly vectorized) env, and load its trained weights
			if hasattr(envs.venv, 'envs'):
				base_env = envs.venv.envs[0].env
			else:
				base_env = envs.venv.unwrapped.envs[0].env
			base_env.robot.policy.set_phase('test')
			base_env.robot.policy.set_device(device)
			base_env.robot.policy.set_env(base_env)
			base_env.robot.policy.load_model(load_path)

	test_size = config.env.test_size

	# call the evaluation function
	evaluate(actor_critic, envs, 1, device, test_size, logging, config, algo_args, test_args.visualize)


if __name__ == '__main__':
	main()