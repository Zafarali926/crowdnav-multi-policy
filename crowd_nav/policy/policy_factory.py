policy_factory = dict()
def none_policy():
    return None

from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.social_force import SOCIAL_FORCE
from crowd_nav.policy.srnn import SRNN, selfAttn_merge_SRNN
from crowd_nav.policy.legacy_rl.cadrl import CADRL
from crowd_nav.policy.legacy_rl.lstm_rl import LstmRL
from crowd_nav.policy.legacy_rl.sarl import SARL
from crowd_nav.policy.legacy_rl.model_predictive_rl import ModelPredictiveRL
from crowd_nav.policy.legacy_rl.orca import ORCA as LegacyORCA

policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
policy_factory['social_force'] = SOCIAL_FORCE
policy_factory['srnn'] = SRNN
policy_factory['selfAttn_merge_srnn'] = selfAttn_merge_SRNN


def _legacy_rl_policy(policy_class):
    """
    crowd_sim/envs/utils/agent.py calls policy_factory[name](config) expecting a
    ready-to-use policy back. The legacy_rl policies (ported from
    ChanganVR/RelationalGraphLearning) instead take no constructor args and need a
    separate .configure(config) call -- wrap that difference here so Agent.__init__
    doesn't need to know about it, and so the vendored files need no changes.

    Also calls set_time_step() here rather than relying on Agent.__init__'s later
    `self.policy.time_step = config.env.time_step` direct attribute assignment:
    ModelPredictiveRL overrides set_time_step() to also propagate time_step to its
    nested state_predictor (which otherwise keeps a None time_step and crashes).
    """
    def factory(config):
        policy = policy_class()
        policy.configure(config)
        policy.set_time_step(config.env.time_step)
        return policy
    return factory


# CADRL/LSTM-RL/SARL/RGL, ported from ChanganVR/RelationalGraphLearning
policy_factory['cadrl'] = _legacy_rl_policy(CADRL)
policy_factory['lstm_rl'] = _legacy_rl_policy(LstmRL)
policy_factory['sarl'] = _legacy_rl_policy(SARL)
policy_factory['model_predictive_rl'] = _legacy_rl_policy(ModelPredictiveRL)
# imitation-learning demonstrator used to warm-start the four policies above
# (see train_legacy_rl.py) -- distinct from this repo's own 'orca' baseline above,
# since it needs to build RelationalGraphLearning's own JointState/FullState and
# record policy.last_state the way its training loop expects.
policy_factory['legacy_orca'] = _legacy_rl_policy(LegacyORCA)