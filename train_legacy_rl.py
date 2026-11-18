"""
Train one of the legacy value-network RL policies (CADRL, LSTM-RL, SARL, RGL --
'model_predictive_rl'), ported from ChanganVR/RelationalGraphLearning, inside this
repo's own CrowdSimVarNum-v0 environment using this repo's own
crowd_nav/configs/config.py -- the same environment and config the attention/GST
policies in train.py are trained and tested with, so results are directly
comparable (apples-to-apples): same human_num, time_limit, discomfort_dist, etc.

Mirrors RelationalGraphLearning's own crowd_nav/train.py: imitation-learning warm
start from ORCA demonstrations, then reinforcement learning with experience replay.
Adapted to this repo's single Config object/environment instead of
RelationalGraphLearning's own EnvConfig/PolicyConfig/TrainConfig and CrowdSim-v0.

Usage:
    python train_legacy_rl.py --policy sarl --output_dir trained_models/sarl
    python train_legacy_rl.py --policy sarl --output_dir trained_models/sarl --resume
"""
import argparse
import copy
import logging
import os
import shutil
import sys

import gym
import torch
from tensorboardX import SummaryWriter

import crowd_sim  # noqa: F401 -- registers the CrowdSim*-v0 gym environments
# NOTE: crowd_sim.envs.utils.robot must be imported before crowd_nav.policy.* --
# crowd_nav/policy/orca.py imports crowd_sim.envs.utils.action, which (via
# crowd_sim/envs/__init__.py) re-enters crowd_sim.envs.crowd_sim, which imports
# crowd_nav.policy.orca back -- a pre-existing circular import in this repo (also
# reproduces on an unmodified checkout) that every existing entry point (train.py,
# test.py) avoids by happening to import something under crowd_sim.envs first.
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.legacy_rl.explorer import Explorer
from crowd_nav.utils.legacy_rl.memory import ReplayMemory
from crowd_nav.utils.legacy_rl.trainer import MPRLTrainer, VNRLTrainer
# crowd_nav.configs.config.Config is NOT imported at module level: its class body
# calls arguments.get_args(), which parses sys.argv with train.py/test.py's PPO
# argument parser -- it would choke on this script's own --policy/--output_dir
# flags. Parse this script's own args first, then import Config with sys.argv
# sanitized (see main() below).

LEGACY_RL_POLICIES = ['cadrl', 'lstm_rl', 'sarl', 'model_predictive_rl']


def main():
    parser = argparse.ArgumentParser("Train a legacy_rl policy inside this repo's own environment/config")
    parser.add_argument('--policy', type=str, required=True, choices=LEGACY_RL_POLICIES)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=17)
    args = parser.parse_args()

    # arguments.get_args() (invoked by importing Config below) parses sys.argv with
    # its own, unrelated PPO argument parser -- hide this script's args from it.
    _real_argv = sys.argv
    sys.argv = [_real_argv[0]]
    from crowd_nav.configs.config import Config
    sys.argv = _real_argv

    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        elif not args.resume:
            raise RuntimeError('{} already exists; pass --overwrite or --resume'.format(args.output_dir))
    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # test.py tries to import <model_dir>.configs.config and <model_dir>.arguments
    # first, falling back to the repo-root defaults only if those don't exist --
    # train.py's own PPO runs rely on this to pin each trained model to the config
    # it was trained with. Do the same here: snapshot config.py with robot.policy
    # set to this run's policy, and arguments.py with --env-name set to
    # CrowdSimVarNum-v0 (the env these policies were actually trained in -- the
    # repo-wide default, CrowdSimPredRealGST-v0, is for the GST predictor).
    save_config_dir = os.path.join(args.output_dir, 'configs')
    os.makedirs(save_config_dir, exist_ok=True)
    with open('crowd_nav/configs/config.py') as f:
        config_src = f.read()
    patched_config_src = config_src.replace(
        "robot.policy = 'selfAttn_merge_srnn'", "robot.policy = '{}'".format(args.policy))
    # also pin sim.predict_method to 'none' -- these policies don't consume
    # predicted trajectories, and leaving the default 'inferred' would also flip
    # env.use_wrapper to True (computed from predict_method a few lines down in
    # the same file), wrapping the env in the GST-specific VecPretextNormalize
    # wrapper at test time.
    patched_config_src = patched_config_src.replace(
        "sim.predict_method = 'inferred'", "sim.predict_method = 'none'")
    if patched_config_src == config_src:
        raise RuntimeError("Couldn't find robot.policy/sim.predict_method defaults to patch in "
                           "crowd_nav/configs/config.py -- has it been edited since this script was written?")
    with open(os.path.join(save_config_dir, 'config.py'), 'w') as f:
        f.write(patched_config_src)
    shutil.copy('crowd_nav/configs/__init__.py', save_config_dir)

    with open('arguments.py') as f:
        args_src = f.read()
    patched_args_src = args_src.replace(
        "default='CrowdSimPredRealGST-v0'", "default='CrowdSimVarNum-v0'")
    if patched_args_src == args_src:
        raise RuntimeError("Couldn't find --env-name default to patch in arguments.py "
                           "-- has it been edited since this script was written?")
    with open(os.path.join(args.output_dir, 'arguments.py'), 'w') as f:
        f.write(patched_args_src)

    log_file = os.path.join(args.output_dir, 'output.log')
    # saved under checkpoints/ to match the path test.py already builds for any
    # policy: os.path.join(model_dir, 'checkpoints', test_model)
    il_weight_file = os.path.join(checkpoints_dir, 'il_model.pth')
    rl_weight_file = os.path.join(checkpoints_dir, 'rl_model.pth')

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout),
                 logging.FileHandler(log_file, mode='a' if args.resume else 'w')],
        format='%(asctime)s, %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    torch.manual_seed(args.randomseed)
    torch.cuda.manual_seed_all(args.randomseed)
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    logging.info('Using device: %s', device)
    writer = SummaryWriter(log_dir=args.output_dir)

    # this repo's own Config -- same env/sim/reward/action_space etc. used by the
    # attention/GST policies in train.py, so the comparison is apples-to-apples
    config = Config()
    config.robot.policy = args.policy
    # inert for CrowdSimVarNum-v0 (only CrowdSimPred*/GST envs read it), kept
    # consistent with the configs/config.py snapshot above
    config.sim.predict_method = 'none'
    if args.debug:
        config.env.val_size = 1
        config.env.test_size = 1
        config.legacy_rl_training.il_episodes = 10
        config.legacy_rl_training.il_epochs = 5
        config.legacy_rl_training.train_episodes = 1
        config.legacy_rl_training.checkpoint_interval = 1
        config.legacy_rl_training.evaluation_interval = 1
        config.legacy_rl_training.target_update_interval = 1
    tc = config.legacy_rl_training

    env = gym.make('CrowdSimVarNum-v0')
    env.configure(config)
    env.nenv = 1
    env.thisSeed = args.randomseed

    # constructing Robot instantiates+configures config.robot.policy via
    # policy_factory (see crowd_sim/envs/utils/agent.py)
    robot = Robot(config, 'robot')
    env.set_robot(robot)
    policy = robot.policy
    if not policy.trainable:
        raise ValueError('Policy has to be trainable')
    policy.set_device(device)

    memory = ReplayMemory(tc.capacity)
    model = policy.get_model()
    if args.policy == 'model_predictive_rl':
        trainer = MPRLTrainer(model, policy.state_predictor, memory, device, policy, writer,
                              tc.trainer_batch_size, tc.trainer_optimizer, config.sim.human_num,
                              reduce_sp_update_frequency=tc.reduce_sp_update_frequency,
                              freeze_state_predictor=tc.freeze_state_predictor,
                              detach_state_predictor=tc.detach_state_predictor,
                              share_graph_model=config.model_predictive_rl.share_graph_model)
    else:
        trainer = VNRLTrainer(model, memory, device, policy, tc.trainer_batch_size, tc.trainer_optimizer, writer)
    explorer = Explorer(env, robot, device, writer, memory, policy.gamma, target_policy=policy)

    # imitation learning: warm-start from ORCA demonstrations, as in
    # RelationalGraphLearning's own train.py
    if args.resume:
        if not os.path.exists(rl_weight_file):
            raise FileNotFoundError('RL weights do not exist: {}'.format(rl_weight_file))
        model.load_state_dict(torch.load(rl_weight_file))
        rl_weight_file = os.path.join(args.output_dir, 'resumed_rl_model.pth')
        logging.info('Loaded RL weights, resuming training')
    elif os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file))
        logging.info('Loaded IL weights')
    else:
        trainer.set_learning_rate(tc.il_learning_rate)
        safety_space = 0 if robot.visible else tc.safety_space
        il_policy = policy_factory[tc.il_policy](config)
        il_policy.time_step = config.env.time_step
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        robot.policy = il_policy
        explorer.run_k_episodes(tc.il_episodes, 'train', update_memory=True, imitation_learning=True)
        trainer.optimize_epoch(tc.il_epochs)
        policy.save_model(il_weight_file)
        logging.info('Finished imitation learning. Experience set size: %d/%d', len(memory), memory.capacity)

    trainer.update_target_model(model)

    # reinforcement learning
    policy.set_env(env)
    robot.policy = policy
    trainer.set_learning_rate(tc.rl_learning_rate)
    if args.resume:
        robot.policy.set_epsilon(tc.epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)

    episode = 0
    best_val_reward = -1
    best_val_model = None
    explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
    explorer.log('val', episode)

    while episode < tc.train_episodes:
        if args.resume:
            epsilon = tc.epsilon_end
        elif episode < tc.epsilon_decay:
            epsilon = tc.epsilon_start + (tc.epsilon_end - tc.epsilon_start) / tc.epsilon_decay * episode
        else:
            epsilon = tc.epsilon_end
        robot.policy.set_epsilon(epsilon)

        explorer.run_k_episodes(tc.sample_episodes, 'train', update_memory=True, episode=episode)
        explorer.log('train', episode)
        trainer.optimize_batch(tc.train_batches, episode)
        episode += 1

        if episode % tc.target_update_interval == 0:
            trainer.update_target_model(model)

        if episode % tc.evaluation_interval == 0:
            _, _, _, reward, _ = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
            explorer.log('val', episode)
            if episode % tc.checkpoint_interval == 0 and reward > best_val_reward:
                best_val_reward = reward
                best_val_model = copy.deepcopy(policy.get_state_dict())

        if episode != 0 and episode % tc.checkpoint_interval == 0:
            checkpoint_idx = episode // tc.checkpoint_interval - 1
            policy.save_model(rl_weight_file.replace('.pth', '_{}.pth'.format(checkpoint_idx)))

    if best_val_model is not None:
        policy.load_state_dict(best_val_model)
        torch.save(best_val_model, os.path.join(args.output_dir, 'best_val.pth'))
        logging.info('Saved best val model, reward=%.4f', best_val_reward)
    policy.save_model(rl_weight_file)
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)


if __name__ == '__main__':
    main()
