import numpy as np
from arguments import get_args

class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    # for now, import all args from arguments.py
    args = get_args()

    training = BaseConfig()
    training.device = "cuda:0" if args.cuda else "cpu"

    # general configs for OpenAI gym env
    env = BaseConfig()
    env.time_limit = 50
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 500
    # if randomize human behaviors, set to True, else set to False
    env.randomize_attributes = True
    env.num_processes = args.num_processes
    # record robot states and actions an episode for system identification in sim2real
    env.record = False
    env.load_act = False

    # config for reward function
    reward = BaseConfig()
    reward.success_reward = 10
    reward.collision_penalty = -20
    # discomfort distance
    reward.discomfort_dist = 0.25
    reward.discomfort_penalty_factor = 10
    reward.gamma = 0.99

    # config for simulation
    sim = BaseConfig()
    sim.circle_radius = 6 * np.sqrt(2)
    sim.arena_size = 6
    sim.human_num = 20
    # actual human num in each timestep, in [human_num-human_num_range, human_num+human_num_range]
    sim.human_num_range = 0
    sim.predict_steps = 5
    # 'const_vel': constant velocity model,
    # 'truth': ground truth future traj (with info in robot's fov)
    # 'inferred': inferred future traj from GST network
    # 'none': no prediction
    sim.predict_method = 'inferred'
    # render the simulation during training or not
    sim.render = False

    # for save_traj only
    render_traj = False
    save_slides = False
    save_path = None

    # whether wrap the vec env with VecPretextNormalize class
    # = True only if we are using a network for human trajectory prediction (sim.predict_method = 'inferred')
    if sim.predict_method == 'inferred':
        env.use_wrapper = True
    else:
        env.use_wrapper = False

    # human config
    humans = BaseConfig()
    humans.visible = True
    # orca or social_force for now
    humans.policy = "orca"
    humans.radius = 0.3
    humans.v_pref = 1
    humans.sensor = "coordinates"
    # FOV = this values * PI
    humans.FOV = 2.

    # a human may change its goal before it reaches its old goal
    # if randomize human behaviors, set to True, else set to False
    humans.random_goal_changing = True
    humans.goal_change_chance = 0.5

    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = True
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3

    humans.random_policy_changing = False

    # robot config
    robot = BaseConfig()
    # whether robot is visible to humans (whether humans respond to the robot's motion)
    robot.visible = False
    # For baseline: srnn; our method: selfAttn_merge_srnn
    robot.policy = 'selfAttn_merge_srnn'
    robot.radius = 0.3
    robot.v_pref = 1
    robot.sensor = "coordinates"
    # FOV = this values * PI
    robot.FOV = 2
    # radius of perception range
    robot.sensor_range = 5

    # action space of the robot
    action_space = BaseConfig()
    # holonomic or unicycle
    action_space.kinematics = "holonomic"

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.15
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # config for social force
    sf = BaseConfig()
    sf.A = 2.
    sf.B = 1
    sf.KI = 1

    # config for data collection for training the GST predictor
    data = BaseConfig()
    data.tot_steps = 40000
    data.render = False
    data.collect_train_data = False
    data.num_processes = 5
    data.data_save_dir = 'gst_updated/datasets/orca_20humans_no_rand'
    # number of seconds between each position in traj pred model
    data.pred_timestep = 0.25

    # config for the GST predictor
    pred = BaseConfig()
    # see 'gst_updated/results/README.md' for how to set this variable
    # If randomized humans: gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj
    # else: gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj
    pred.model_dir = 'gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj'

    # LIDAR config
    lidar = BaseConfig()
    # angular resolution (offset angle between neighboring rays) in degrees
    lidar.angular_res = 5
    # range in meters
    lidar.range = 10

    # config for sim2real
    sim2real = BaseConfig()
    # use dummy robot and human states or not
    sim2real.use_dummy_detect = True
    sim2real.record = False
    sim2real.load_act = False
    sim2real.ROSStepInterval = 0.03
    sim2real.fixed_time_interval = 0.1
    sim2real.use_fixed_time_interval = True

    if sim.predict_method == 'inferred' and env.use_wrapper == False:
        raise ValueError("If using inferred prediction, you must wrap the envs!")
    if sim.predict_method != 'inferred' and env.use_wrapper:
        raise ValueError("If not using inferred prediction, you must NOT wrap the envs!")

    # --- config needed by the legacy value-network RL policies ported from
    # ChanganVR/RelationalGraphLearning (crowd_nav/policy/legacy_rl/): CADRL,
    # LSTM-RL, SARL, RGL (model_predictive_rl). Field names/defaults follow that
    # repo's crowd_nav/configs/icra_benchmark/config.py so the policies' own
    # configure() methods need no changes. ---

    # these policies read config.rl.gamma (RelationalGraphLearning's own name for
    # this repo's reward.gamma)
    rl = BaseConfig()
    rl.gamma = reward.gamma

    # occupancy map, shared by SARL/LSTM-RL(with_om)/CADRL(with_om)
    om = BaseConfig()
    om.cell_num = 4
    om.cell_size = 1
    om.om_channel_size = 3

    # extra action-space fields these policies need beyond kinematics
    action_space.speed_samples = 5
    action_space.rotation_samples = 16
    action_space.sampling = 'exponential'
    action_space.query_env = False
    action_space.rotation_constraint = np.pi / 3

    cadrl = BaseConfig()
    cadrl.mlp_dims = [150, 100, 100, 1]
    cadrl.multiagent_training = False

    lstm_rl = BaseConfig()
    lstm_rl.global_state_dim = 50
    lstm_rl.mlp1_dims = [150, 100, 100, 50]
    lstm_rl.mlp2_dims = [150, 100, 100, 1]
    lstm_rl.multiagent_training = True
    lstm_rl.with_om = False
    lstm_rl.with_interaction_module = True

    sarl = BaseConfig()
    sarl.mlp1_dims = [150, 100]
    sarl.mlp2_dims = [100, 50]
    sarl.attention_dims = [100, 100, 1]
    sarl.mlp3_dims = [150, 100, 100, 1]
    sarl.multiagent_training = True
    sarl.with_om = True
    sarl.with_global_state = True

    # graph model used by RGL (model_predictive_rl)
    gcn = BaseConfig()
    gcn.multiagent_training = True
    gcn.num_layer = 2
    gcn.X_dim = 32
    gcn.wr_dims = [64, gcn.X_dim]
    gcn.wh_dims = [64, gcn.X_dim]
    gcn.final_state_dim = gcn.X_dim
    gcn.gcn2_w1_dim = gcn.X_dim
    gcn.planning_dims = [150, 100, 100, 1]
    gcn.similarity_function = 'embedded_gaussian'
    gcn.layerwise_graph = False
    gcn.skip_connection = True

    model_predictive_rl = BaseConfig()
    model_predictive_rl.linear_state_predictor = False
    model_predictive_rl.planning_depth = 1
    model_predictive_rl.planning_width = 1
    model_predictive_rl.do_action_clip = False
    model_predictive_rl.motion_predictor_dims = [64, 5]
    model_predictive_rl.value_network_dims = [32, 100, 100, 1]
    model_predictive_rl.share_graph_model = False

    # imitation-learning warm start + RL training hyperparameters for the four
    # legacy policies above, used only by train_legacy_rl.py -- train.py's PPO
    # policies do not read this section
    legacy_rl_training = BaseConfig()
    legacy_rl_training.trainer_batch_size = 100
    legacy_rl_training.trainer_optimizer = 'Adam'
    legacy_rl_training.il_episodes = 2000
    legacy_rl_training.il_policy = 'legacy_orca'
    legacy_rl_training.il_epochs = 50
    legacy_rl_training.il_learning_rate = 0.001
    legacy_rl_training.safety_space = 0.15
    legacy_rl_training.rl_learning_rate = 0.001
    legacy_rl_training.train_batches = 100
    legacy_rl_training.train_episodes = 10000
    legacy_rl_training.sample_episodes = 1
    legacy_rl_training.target_update_interval = 1000
    legacy_rl_training.evaluation_interval = 1000
    legacy_rl_training.capacity = 100000
    legacy_rl_training.epsilon_start = 0.5
    legacy_rl_training.epsilon_end = 0.1
    legacy_rl_training.epsilon_decay = 4000
    legacy_rl_training.checkpoint_interval = 1000
    legacy_rl_training.freeze_state_predictor = False
    legacy_rl_training.detach_state_predictor = False
    legacy_rl_training.reduce_sp_update_frequency = False
