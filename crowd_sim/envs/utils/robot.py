from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_nav.policy.legacy_rl.policy import Policy as LegacyPolicy
from crowd_nav.policy.legacy_rl.state import FullState as LegacyFullState, \
    ObservableState as LegacyObservableState, JointState as LegacyJointState


class Robot(Agent):
    def __init__(self, config,section):
        super().__init__(config,section)
        self.sensor_range = config.robot.sensor_range

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        # CADRL/LSTM-RL/SARL/RGL (ported from ChanganVR/RelationalGraphLearning, see
        # crowd_nav/policy/legacy_rl/) depend on RelationalGraphLearning's own
        # class-based JointState (attribute `robot_state`, not `self_state`), not this
        # repo's namedtuple-based one -- build that instead for those policies.
        if isinstance(self.policy, LegacyPolicy):
            robot_state = LegacyFullState(*self.get_full_state())
            human_states = [LegacyObservableState(*human_state) for human_state in ob]
            state = LegacyJointState(robot_state, human_states)
        else:
            state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action


    def actWithJointState(self,ob):
        action = self.policy.predict(ob)
        return action
