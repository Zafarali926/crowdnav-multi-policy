import logging
import torch
from tqdm import tqdm
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.info import *


class Explorer(object):
    """
    Ported from ChanganVR/RelationalGraphLearning's crowd_nav/utils/explorer.py.

    Differences from upstream, required to run inside this repo's CrowdSimVarNum env:
    - the action is NOT computed here via `self.robot.act(ob)` before stepping, because
      this repo's env always returns a dict observation (built for the attention/GST
      pipeline) rather than the plain per-human state list RelationalGraphLearning's
      `Robot.act()` expects. Instead, a harmless placeholder action is passed into
      `env.step()`, which (for legacy_rl/legacy_orca policies, see
      crowd_sim/envs/crowd_sim_var_num.py's bypass branch and
      crowd_sim/envs/utils/robot.py's patched `act()`) recomputes the real action
      internally from the env's own last_human_states -- the same mechanism already
      used for this repo's ORCA/social_force baselines.
    - this repo wraps episode-end info as {'info': <Info object>} rather than handing it
      back directly, and names the "too close" info class `Danger` rather than
      `Discomfort`.
    """

    def __init__(self, env, robot, device, writer, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None

    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, epoch=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        # this repo's own env.step() only special-cases env.phase == 'train' (cheap
        # "circle" danger zone) vs anything else (future-trajectory-based danger
        # zone, see crowd_sim_var_num.py's calc_reward) -- there's no third 'val'
        # env.phase in this repo's own pipeline (make_vec_envs only ever sets
        # 'train' or 'test'), so map RelationalGraphLearning's 'val' phase onto the
        # same 'test' semantics Repo A's own evaluate() uses.
        self.env.phase = 'train' if phase == 'train' else 'test'
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        discomfort = 0
        min_dist = []
        cumulative_rewards = []
        average_returns = []
        collision_cases = []
        timeout_cases = []

        placeholder_action = ActionXY(0, 0) if self.robot.kinematics == 'holonomic' else ActionRot(0, 0)

        pbar = tqdm(total=k) if k != 1 else None

        for i in range(k):
            self.env.reset(phase)
            done = False
            states = []
            rewards = []
            episode_info = None
            while not done:
                ob, reward, done, info = self.env.step(placeholder_action)
                episode_info = info['info']
                states.append(self.robot.policy.last_state)
                rewards.append(reward)

                if isinstance(episode_info, Danger):
                    discomfort += 1
                    min_dist.append(episode_info.min_dist)

            if isinstance(episode_info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(episode_info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(episode_info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(episode_info, ReachGoal) or isinstance(episode_info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            average_returns.append(average(returns))

            if pbar:
                pbar.update(1)
        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f},'
                     ' average return: {:.4f}'. format(phase.upper(), extra_info, success_rate, collision_rate,
                                                       avg_nav_time, average(cumulative_rewards),
                                                       average(average_returns)))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times)
            if total_time > 0 and min_dist:
                logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                             discomfort / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        self.statistics = success_rate, collision_rate, avg_nav_time, average(cumulative_rewards), average(average_returns)

        return self.statistics

    def update_memory(self, states, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states[:-1]):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                next_state = self.target_policy.transform(states[i + 1])
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.robot.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i + 1]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)

            if self.target_policy.name == 'ModelPredictiveRL':
                self.memory.push((state[0], state[1], value, reward, next_state[0], next_state[1]))
            else:
                self.memory.push((state, value, reward, next_state))

    def log(self, tag_prefix, global_step):
        sr, cr, time, reward, avg_return = self.statistics
        self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        self.writer.add_scalar(tag_prefix + '/collision_rate', cr, global_step)
        self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
