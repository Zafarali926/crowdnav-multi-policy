# legacy_rl

CADRL, LSTM-RL, SARL, and RGL (`model_predictive_rl.py`), ported from
[ChanganVR/RelationalGraphLearning](https://github.com/ChanganVR/RelationalGraphLearning)
(`master` branch). Fully wired into this repo's own environment, config, training,
and testing -- see `train_legacy_rl.py` at the repo root to train one, and `test.py`
to evaluate a trained checkpoint with this repo's own metrics (same as the
attention/GST policies: success/collision/timeout rate, nav time, path length,
intrusion ratio, min intrusion distance).

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

Import paths, to resolve under this package (e.g.
`from crowd_sim.envs.utils.state import ObservableState` ->
`from crowd_nav.policy.legacy_rl.state import ObservableState`). All files pass
`python -m py_compile` and were exercised end-to-end (see below).

Three genuine bugs in the **upstream** files (reproduce against an unmodified
checkout of ChanganVR/RelationalGraphLearning too — not introduced by vendoring)
were also fixed here, since otherwise these policies crash immediately:

- `cadrl.py`: `CADRL.predict()` (used when `multiagent_training=False`) referenced
  `state.self_state`, an attribute that doesn't exist on this repo's own `JointState`
  (it's `robot_state`) -- fixed to `state.robot_state` (5 occurrences).
- `cadrl.py`: `CADRL.predict()` also called `self.compute_reward(...)`, a method only
  defined on the `MultiHumanRL` subclass -- added the same implementation to `CADRL`
  itself (`MultiHumanRL.compute_reward` is unchanged and still shadows it).
- `lstm_rl.py`: the interaction-module human-sorting helper had the same
  `state.self_state` -> should be `state.robot_state` bug (1 occurrence).
- `graph_model.py`: `RGL.forward()`'s skip-connection branch did `next_H += H`, an
  in-place add on a tensor that is also the direct output of `relu()` just above and
  needed unmodified for that `relu`'s own backward pass. PyTorch's autograd version
  check rejects this (`RuntimeError: ... modified by an inplace operation`). Changed
  to the equivalent non-in-place `next_H = next_H + H`.

`gcn.py` (an alternate/duplicate graph-model implementation vendored alongside
`graph_model.py`) is not imported by any of the four policies -- left untouched.

## Wiring (done)

- **`crowd_nav/policy/policy_factory.py`**: registers `cadrl`, `lstm_rl`, `sarl`,
  `model_predictive_rl`, and `legacy_orca` (the imitation-learning demonstrator used
  to warm-start the other four, see below) via a small adapter (`_legacy_rl_policy`)
  that calls `.configure(config)` and `.set_time_step(config.env.time_step)` -- these
  policies take no constructor args, unlike this repo's own `Policy(config)` classes.
- **`crowd_nav/configs/config.py`**: added the config sections these policies read
  (`rl.gamma` mirroring this repo's own `reward.gamma`, `om`, `cadrl`, `lstm_rl`,
  `sarl`, `gcn`, `model_predictive_rl`, plus extra `action_space.*` fields and a new
  `legacy_rl_training` section for IL/RL hyperparameters) -- field names/defaults
  follow RelationalGraphLearning's own `icra_benchmark/config.py` so the policies'
  `configure()` methods needed no changes.
- **`crowd_sim/envs/utils/robot.py`**: `Robot.act()` now builds this package's own
  `JointState` (via `isinstance(self.policy, legacy_rl.policy.Policy)`) instead of
  this repo's namedtuple one, for these policies only.
- **`crowd_sim/envs/crowd_sim_var_num.py`** / **`rl/evaluation.py`**: the existing
  bypass branch that special-cases `['ORCA', 'social_force']` (these policies don't
  consume the attention-network dict observation, so their action is recomputed from
  the env's own human states via `Robot.act()`) now also covers `'CADRL'`, `'LSTM-RL'`,
  `'SARL'`/`'OM-SARL'`, `'ModelPredictiveRL'` (env) and `cadrl`/`lstm_rl`/`sarl`/
  `model_predictive_rl` (evaluation.py's config-string check).
- **`test.py`**: extended the same bypass so these policies skip the PPO
  `actor_critic` entirely, loading their own saved weights via `policy.load_model()`
  into the policy object the env already built. Also fixed a pre-existing bug (see
  below) where `get_args()`/`Config()` crash on this script's own CLI flags.

## Training (`crowd_nav/utils/legacy_rl/`, `train_legacy_rl.py`)

`memory.py` and `trainer.py` are vendored unchanged (no RelationalGraphLearning-specific
imports). `explorer.py` is adapted from RelationalGraphLearning's own
`crowd_nav/utils/explorer.py`:

- it no longer calls `self.robot.act(ob)` itself before stepping -- this repo's env
  always returns a dict observation (built for the attention pipeline), not the
  plain per-human state list RelationalGraphLearning's `Robot.act()` expects. A
  placeholder action is passed into `env.step()` instead, which (via the bypass
  branch above) recomputes the real action internally, the same mechanism already
  used for this repo's ORCA/social_force baselines.
- it reads episode-end info as `info['info']` (this repo wraps it in a dict) and
  checks for `Danger` rather than `Discomfort` (this repo's name for the same thing).
- it sets `self.env.phase` explicitly (`'train'` vs `'test'`) before each batch of
  episodes -- this repo's own env only ever gets `'train'`/`'test'` from
  `make_vec_envs`, with no third `'val'` phase, so RelationalGraphLearning's `'val'`
  phase is mapped onto this repo's `'test'` semantics (future-trajectory-based danger
  zone) rather than crashing on a never-computed `human_future_traj`.

`train_legacy_rl.py` mirrors RelationalGraphLearning's own `crowd_nav/train.py`
(imitation-learning warm start from ORCA demonstrations, then RL with experience
replay) but built on this repo's own `CrowdSimVarNum-v0` environment and
`crowd_nav/configs/config.py` -- the same environment and config the attention/GST
policies in `train.py` are trained and tested with, so results are directly
comparable. It also snapshots a patched `configs/config.py` (`robot.policy` set to
the trained policy, `sim.predict_method` forced to `'none'`) and `arguments.py`
(`--env-name` forced to `CrowdSimVarNum-v0`, since the repo-wide default is
`CrowdSimPredRealGST-v0`) into its output directory, matching the convention
`train.py` already uses so `test.py` picks up the right settings automatically.

All four policies were run end-to-end (IL + RL phases, checkpointing, val/test
evaluation) with `--debug` (tiny episode counts) to verify there are no further
crashes; `test.py` was verified to load a trained checkpoint and produce the full
metric suite via a direct call to `rl.evaluation.evaluate()` (bypassing a separate,
pre-existing `sys.argv`-parsing bug in `test.py`/`get_args()`, fixed above, which
reproduces for this repo's own ORCA/social_force baselines too, not just these
policies -- `test.py`'s own `--model_dir` etc. flags and `arguments.py`'s PPO parser
were both reading the same real `sys.argv`).

## Not done

- Training only covers these four policies against this repo's attention/GST
  policies on the *same environment/metrics*. It does not reproduce
  RelationalGraphLearning's own published hyperparameter search or multi-seed
  results, and `config.legacy_rl_training` is a direct carry-over of
  RelationalGraphLearning's `icra_benchmark` defaults, not retuned for this repo's
  larger default scenario (20 humans vs. RelationalGraphLearning's own 5).
- `gcn.py` is unused dead code (see above) -- left as-is rather than removed, to
  keep the vendored set a faithful copy of upstream's `crowd_nav/policy/` directory.
