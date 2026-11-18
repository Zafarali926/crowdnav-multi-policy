# legacy_rl

CADRL, LSTM-RL, SARL, and RGL (`model_predictive_rl.py`), ported from
[ChanganVR/RelationalGraphLearning](https://github.com/ChanganVR/RelationalGraphLearning)
(`master` branch), vendored here unchanged except for import paths.

## Why a separate package

This repo's own `crowd_nav/policy/` and `crowd_sim/envs/utils/state.py` use a different,
newer interface (namedtuple-based `FullState`/`ObservableState`, attribute `self_state`)
built for the attention/GST-prediction pipeline. The files in this package depend on
RelationalGraphLearning's own class-based `state.py` (attribute `robot_state`, methods like
`to_tuple()`/`to_tensor()`, `isinstance()` checks in `cadrl.py`), so that original `state.py`
and `policy.py` base class are vendored here too (`state.py`, `policy.py`) rather than reusing
this repo's versions, to avoid breaking either interface.

`crowd_sim/envs/utils/action.py` (`ActionXY`/`ActionRot`) is identical in both repos and is
**not** duplicated here — these files import it directly from this repo's own
`crowd_sim.envs.utils.action`.

## What changed from upstream

Only import paths, to resolve under this package (e.g.
`from crowd_sim.envs.utils.state import ObservableState` ->
`from crowd_nav.policy.legacy_rl.state import ObservableState`). No logic was modified.
All files pass `python -m py_compile`.

## What's still needed before these policies can run inside this repo's simulator (not done yet)

- Register each policy (`cadrl`, `lstm_rl`, `sarl`, `model_predictive_rl`) in
  `crowd_nav/policy/policy_factory.py`.
- Add the config sections these policies read (`config.cadrl`, `config.lstm_rl`,
  `config.sarl`, `config.gcn`, `config.om`, `config.action_space`, etc.) to
  `crowd_nav/configs/config.py`.
- Add these policy names to the existing bypass branch in
  `crowd_sim/envs/crowd_sim_var_num.py::step()` (and `rl/evaluation.py::evaluate()`) that
  currently special-cases `['ORCA', 'social_force']`, so these policies compute their action
  via `Robot.act(ob)` (which builds a `JointState` through this repo's own `state.py`) —
  that legacy `JointState` then needs converting into this package's `JointState`
  (`robot_state`/`human_states` of `FullState`/`ObservableState`) before calling `.predict()`.
