""" Basic model configs for Unitree Go2 """
import numpy as np
import os.path as osp

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

go2_action_scale = 0.5
go2_const_dof_range = dict(
    Hip_max= 1.0472,
    Hip_min= -1.0472,
    Front_Thigh_max= 3.4907,
    Front_Thigh_min= -1.5708,
    Rear_Thingh_max= 4.5379,
    Rear_Thingh_min= -0.5236,
    Calf_max= -0.83776,
    Calf_min= -2.7227,
)

class Go2BaseCfg( LeggedRobotCfg ):
    class env:
        num_envs = 4096
        num_observations = None # No use, use obs_components
        num_privileged_obs = None # No use, use privileged_obs_components

        use_lin_vel = False # to be decided
        num_actions = 12
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        obs_components = [
            "lin_vel",
            "ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "last_actions",
            "height_measurements",
            "engaging_block",
            "sidewall_distance"
        ]

    class sensor:
        class proprioception:
            obs_components = ["ang_vel", "projected_gravity", "commands", "dof_pos", "dof_vel"]
            latency_range = [0.005, 0.045] # [s]
            latency_resampling_time = 5.0 # [s]

    class terrain:
        # selected = "TerrainPerlin"
        mesh_type = None
        curriculum = False

        measure_heights = True
        # x: [-0.5, 1.5], y: [-0.5, 0.5] range for go2
        measured_points_x = [i for i in np.arange(-0.5, 1.51, 0.1)]
        measured_points_y = [i for i in np.arange(-0.5, 0.51, 0.1)]

        horizontal_scale = 0.025 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 4.
        terrain_width = 4.
        slope_treshold = 1.
    
        num_rows= 16 # number of terrain rows (levels)
        num_cols = 16 # number of terrain cols (types)

        selected = "BarrierTrack"
        BarrierTrack_kwargs = dict(
            options= [
                # "jump",
                # "crawl",
                # "tilt",
                # "leap",
            ], # each race track will permute all the options
            track_width= 1.6,
            track_block_length= 2., # the x-axis distance from the env origin point
            wall_thickness= (0.04, 0.2), # [m]
            wall_height= -0.05,
            jump= dict(
                height= (0.2, 0.6),
                depth= (0.1, 0.8), # size along the forward axis
                fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
                jump_down_prob= 0., # probability of jumping down use it in non-virtual terrain
            ),
            crawl= dict(
                height= (0.25, 0.5),
                depth= (0.1, 0.6), # size along the forward axis
                wall_height= 0.6,
                no_perlin_at_obstacle= False,
            ),
            tilt= dict(
                width= (0.24, 0.32),
                depth= (0.4, 1.), # size along the forward axis
                opening_angle= 0.0, # [rad] an opening that make the robot easier to get into the obstacle
                wall_height= 0.5,
            ),
            leap= dict(
                length= (0.2, 1.0),
                depth= (0.4, 0.8),
                height= 0.2,
            ),
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 1.2,
            engaging_finish_threshold= 0.,
            curriculum_perlin= False,
            no_perlin_threshold= 0.1,
        )

        TerrainPerlin_kwargs = dict(
            zScale= 0.07,
            frequency= 10,
        )
    
    class commands( LeggedRobotCfg.commands ):
        heading_command = False
        resampling_time = 5 # [s]
        lin_cmd_cutoff = 0.2
        ang_cmd_cutoff = 0.2
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.5]
            lin_vel_y = [-1., 1.]
            ang_vel_yaw = [-2., 2.]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0., 0., 0.5] # [m]
        default_joint_angles = { # 12 joints in the order of simulation
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.7,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.7,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        }

    class control( LeggedRobotCfg.control ):
        stiffness = {'joint': 40.}
        damping = {'joint': 1.}
        action_scale = go2_action_scale
        computer_clip_torque = False
        motor_clip_torque = True

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2_qrc/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        front_hip_names = ["FL_hip_joint", "FR_hip_joint"]
        rear_hip_names = ["RL_hip_joint", "RR_hip_joint"]
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        sdk_dof_range = go2_const_dof_range
        dof_velocity_override = 35.

    class termination:
        termination_terms = [
            "roll",
            "pitch",
        ]

        roll_kwargs = dict(
            threshold= 3.0, # [rad]
        )
        pitch_kwargs = dict(
            threshold= 3.0, # [rad] # for leap, jump
        )

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_com = True
        class com_range:
            x = [-0.2, 0.2]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]

        randomize_motor = True
        leg_motor_strength_range = [0.8, 1.2]

        randomize_base_mass = True
        added_mass_range = [1.0, 3.0]

        randomize_friction = True
        friction_range = [0., 2.]

        init_base_pos_range = dict(
            x= [0.05, 0.6],
            y= [-0.25, 0.25],
        )
        init_base_rot_range = dict(
            roll= [-0.75, 0.75],
            pitch= [-0.75, 0.75],
        )
        init_base_vel_range = dict(
            x= [-0.2, 1.5],
            y= [-0.2, 0.2],
            z= [-0.2, 0.2],
            roll= [-1., 1.],
            pitch= [-1., 1.],
            yaw= [-1., 1.],
        )
        init_dof_vel_range = [-5, 5]

        push_robots = True 
        max_push_vel_xy = 0.5 # [m/s]
        push_interval_s = 2

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            tracking_lin_vel = 1.
            tracking_ang_vel = 1.
            energy_substeps = -2e-5
            stand_still = -2.
            dof_error_named = -1.
            dof_error = -0.01
            # penalty for hardware safety
            exceed_dof_pos_limits = -0.4
            exceed_torque_limits_l1norm = -0.4
            dof_vel_limits = -0.4
        dof_error_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
        only_positive_rewards = False
        soft_dof_vel_limit = 0.9
        soft_dof_pos_limit = 0.9
        soft_torque_limit = 0.9

    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 1.
        height_measurements_offset = -0.2
        clip_actions_method = None # let the policy learns not to exceed the limits

    class noise( LeggedRobotCfg.noise ):
        add_noise = False

    class viewer( LeggedRobotCfg.viewer ):
        pos = [-1., -1., 0.4]
        lookat = [0., 0., 0.3]

    class sim( LeggedRobotCfg.sim ):
        body_measure_points = { # transform are related to body frame
            "base": dict(
                x= [i for i in np.arange(-0.24, 0.41, 0.03)],
                y= [-0.08, -0.04, 0.0, 0.04, 0.08],
                z= [i for i in np.arange(-0.061, 0.071, 0.03)],
                transform= [0., 0., 0.005, 0., 0., 0.],
            ),
            "thigh": dict(
                x= [
                    -0.16, -0.158, -0.156, -0.154, -0.152,
                    -0.15, -0.145, -0.14, -0.135, -0.13, -0.125, -0.12, -0.115, -0.11, -0.105, -0.1, -0.095, -0.09, -0.085, -0.08, -0.075, -0.07, -0.065, -0.05,
                    0.0, 0.05, 0.1,
                ],
                y= [-0.015, -0.01, 0.0, -0.01, 0.015],
                z= [-0.03, -0.015, 0.0, 0.015],
                transform= [0., 0., -0.1,   0., 1.57079632679, 0.],
            ),
            "calf": dict(
                x= [i for i in np.arange(-0.13, 0.111, 0.03)],
                y= [-0.015, 0.0, 0.015],
                z= [-0.015, 0.0, 0.015],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
        }

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2BaseCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 0.2
        learning_rate = 1e-4
        optimizer_class_name = "AdamW"

    class policy( LeggedRobotCfgPPO.policy ):
        # configs for estimator module
        estimator_obs_components = [
            "ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "last_actions",
        ]
        estimator_target_components = ["lin_vel"]
        replace_state_prob = 1.0
        class estimator_kwargs:
            hidden_sizes = [128, 64]
            nonlinearity = "CELU"
        # configs for (critic) encoder
        encoder_component_names = ["height_measurements", "engaging_block"]
        encoder_class_name = "MlpModel"
        class encoder_kwargs:
            hidden_sizes = [128, 64]
            nonlinearity = "CELU"
        encoder_output_size = 32
        critic_encoder_component_names = ["height_measurements", "engaging_block"]
        init_noise_std = 0.5
        # configs for policy: using recurrent policy with GRU
        rnn_type = 'gru'
        mu_activation = None

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "EncoderStateAcRecurrent"
        algorithm_class_name = "EstimatorPPO"
        experiment_name = "base_go2"
        
        resume = False
        load_run = None

        # Compute urdf_tag: take the basename of the parent directory of asset.file's parent
        # (e.g. .../go2_qrc/urdf/go2.urdf -> go2_qrc)
        try:
            asset_file = getattr(Go2BaseCfg.asset, "file", "")
            # Remove template variables to obtain an operable path fragment
            asset_file_clean = asset_file.replace("{LEGGED_GYM_ROOT_DIR}", "").replace("$LEGGED_GYM_ROOT_DIR", "")
            urdf_tag = osp.basename(osp.dirname(osp.dirname(asset_file_clean))) or "unknown"
        except Exception:
            urdf_tag = "unknown"

        run_name = "".join(["Go2Base",
            ("_pEnergy" + np.format_float_scientific(Go2BaseCfg.rewards.scales.energy_substeps, precision= 1, trim= "-") if Go2BaseCfg.rewards.scales.energy_substeps != 0 else ""),
            ("_pDofErr" + np.format_float_scientific(Go2BaseCfg.rewards.scales.dof_error, precision= 1, trim= "-") if Go2BaseCfg.rewards.scales.dof_error != 0 else ""),
            ("_pDofErrN" + np.format_float_scientific(Go2BaseCfg.rewards.scales.dof_error_named, precision= 1, trim= "-") if Go2BaseCfg.rewards.scales.dof_error_named != 0 else ""),
            ("_pStand" + np.format_float_scientific(Go2BaseCfg.rewards.scales.stand_still, precision= 1, trim= "-") if Go2BaseCfg.rewards.scales.stand_still != 0 else ""),
            ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
            ("_urdf" + urdf_tag),
        ])

        max_iterations = 2000
        save_interval = 1000
        log_interval = 100
