""" Config to train the whole parkour oracle policy """
import numpy as np
from os import path as osp
from collections import OrderedDict
from datetime import datetime

from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO, Go2BaseCfgPPO
from legged_gym.envs.go2.go2_crawl_config import Go2CrawlCfg, Go2CrawlCfgPPO

multi_process_ = True
class Go2DistillCrawlCfg( Go2CrawlCfg ):
    class env( Go2CrawlCfg.env ):
        num_envs = 256
        obs_components = [
            "lin_vel",
            "ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "last_actions",
            "forward_depth",
        ]

        privileged_obs_components = [
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

    class terrain( Go2CrawlCfg.terrain ):
        if multi_process_:
            num_rows = 4
            num_cols = 1
        else:
            num_rows = 10
            num_cols = 20
        curriculum = False

        BarrierTrack_kwargs = merge_dict(Go2CrawlCfg.terrain.BarrierTrack_kwargs, dict(
            leap= dict(
                length= [0.05, 0.8],
                depth= [0.5, 0.8],
                height= 0.15, # expected leap height over the gap
                fake_offset= 0.1,
            ),
        ))

    class sensor( Go2CrawlCfg.sensor ):
        class forward_camera:
            obs_components = ["forward_depth"]
            resolution = [int(480/4), int(640/4)]
            position = dict(
                mean= [0.32, 0.0, 0.035],
                std= [0.01, 0.01, 0.01],
            )
            rotation = dict(
                lower= [-0.1, -0.1, -0.1],
                upper= [0.1, 0.1, 0.1],
            )
            resized_resolution = [48, 64]
            output_resolution = [48, 64]
            horizontal_fov = [86, 90]
            crop_top_bottom = [int(48/4), 0]
            crop_left_right = [int(28/4), int(36/4)]
            near_plane = 0.05
            depth_range = [0.0, 3.0]

            latency_range = [0.08, 0.142]
            latency_resampling_time = 5.0
            refresh_duration = 1/10 # [s]

    class commands( Go2CrawlCfg.commands ):
        # a mixture of command sampling and goal_based command update allows only high speed range
        # in x-axis but no limits on y-axis and yaw-axis
        lin_cmd_cutoff = 0.2
        class ranges( Go2CrawlCfg.commands.ranges ):
            # lin_vel_x = [0.6, 1.8]
            lin_vel_x = [-0.6, 2.0]
        
        is_goal_based = True
        class goal_based:
            # the ratios are related to the goal position in robot frame
            x_ratio = None # sample from lin_vel_x range
            y_ratio = 1.2
            yaw_ratio = 0.8
            follow_cmd_cutoff = True
            x_stop_by_yaw_threshold = 1. # stop when yaw is over this threshold [rad]

    class normalization( Go2CrawlCfg.normalization ):
        class obs_scales( Go2CrawlCfg.normalization.obs_scales ):
            forward_depth = 1.0

    class noise( Go2CrawlCfg.noise ):
        add_noise = False
        class noise_scales( Go2CrawlCfg.noise.noise_scales ):
            forward_depth = 0.0
            ### noise for simulating sensors
            commands = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            projected_gravity = 0.02
            dof_pos = 0.02
            dof_vel = 0.2
            last_actions = 0.
            ### noise for simulating sensors
        class forward_depth:
            stereo_min_distance = 0.175 # when using (480, 640) resolution
            stereo_far_distance = 1.2
            stereo_far_noise_std = 0.08 
            stereo_near_noise_std = 0.02
            stereo_full_block_artifacts_prob = 0.008
            stereo_full_block_values = [0.0, 0.25, 0.5, 1., 3.]
            stereo_full_block_height_mean_std = [62, 1.5]
            stereo_full_block_width_mean_std = [3, 0.01]
            stereo_half_block_spark_prob = 0.02
            stereo_half_block_value = 3000
            sky_artifacts_prob = 0.0001
            sky_artifacts_far_distance = 2.
            sky_artifacts_values = [0.6, 1., 1.2, 1.5, 1.8]
            sky_artifacts_height_mean_std = [2, 3.2]
            sky_artifacts_width_mean_std = [2, 3.2]

    class curriculum:
        no_moveup_when_fall = False

    class sim( Go2CrawlCfg.sim ):
        no_camera = False
    
logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2DistillCrawlCfgPPO( Go2CrawlCfgPPO ):
    class algorithm( Go2CrawlCfgPPO.algorithm ):
        entropy_coef = 0.0
        using_ppo = False
        num_learning_epochs = 8
        num_mini_batches = 2
        distill_target = "l1"
        learning_rate = 3e-4
        optimizer_class_name = "AdamW"
        teacher_act_prob = 0.
        distillation_loss_coef = 1.0
        # update_times_scale = 100
        action_labels_from_sample = False

        teacher_policy_class_name = "EncoderStateAcRecurrent"
        teacher_ac_path = osp.join(logs_root, "field_go2_crawl",
            "/home/yjh/parkour/legged_gym/logs/field_go2_crawl/Sep28_20-23-15_Skills_crawl_comXRange-0.2-0.2_noLinVel_pDof1e-01_pTorque1e-7_pTorqueL11e-01_noDelayActObs_noTanh_fromSep25_21-19-36",
            "model_22000.pt"
        )

        class teacher_policy( Go2CrawlCfgPPO.policy ):
            num_actor_obs = 48 + 21 * 11 + 203 + 2
            num_critic_obs = 48 + 21 * 11 + 203 + 2
            num_actions = 12
            obs_segments = OrderedDict([
                ("lin_vel", (3,)),
                ("ang_vel", (3,)),
                ("projected_gravity", (3,)),
                ("commands", (3,)),
                ("dof_pos", (12,)),
                ("dof_vel", (12,)),
                ("last_actions", (12,)), # till here: 3+3+3+3+12+12+12 = 48
                ("height_measurements", (1, 21, 11)),
                ("engaging_block", (203,)),
                ("sidewall_distance", (2,)),
            ])

    class policy( Go2BaseCfgPPO.policy ):
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
        # configs for visual encoder
        encoder_component_names = ["forward_depth"]
        encoder_class_name = "Conv2dHeadModel"
        class encoder_kwargs:
            channels = [16, 32, 32]
            kernel_sizes = [5, 4, 3]
            strides = [2, 2, 1]
            hidden_sizes = [128]
            use_maxpool = True
            nonlinearity = "LeakyReLU"
        # configs for critic encoder
        critic_encoder_component_names = ["height_measurements", "engaging_block"]
        critic_encoder_class_name = "MlpModel"
        class critic_encoder_kwargs:
            hidden_sizes = [128, 64]
            nonlinearity = "CELU"
        encoder_output_size = 32

        init_noise_std = 0.1

    if multi_process_:
        runner_class_name = "TwoStageRunner"
    class runner( Go2CrawlCfgPPO.runner ):
        policy_class_name = "EncoderStateAcRecurrent"
        algorithm_class_name = "EstimatorTPPO"
        experiment_name = "distill_go2"
        num_steps_per_env = 32

        if multi_process_:
            pretrain_iterations = -1
            class pretrain_dataset:
                data_dir = "/home/yjh/parkour/legged_gym/logs/tmp"
                dataset_loops = -1
                random_shuffle_traj_order = True
                keep_latest_n_trajs = 1500
                starting_frame_range = [0, 50]

        resume = False
        load_run = osp.join(logs_root, "field_go2_crawl",
            "/home/yjh/parkour/legged_gym/logs/field_go2_crawl/Sep28_20-23-15_Skills_crawl_comXRange-0.2-0.2_noLinVel_pDof1e-01_pTorque1e-7_pTorqueL11e-01_noDelayActObs_noTanh_fromSep25_21-19-36",
        )
        ckpt_manipulator = "replace_encoder0" if "field_go2_crawl" in load_run else None

        run_name = "".join(["Go2_",
            ("{:d}skills".format(len(Go2DistillCrawlCfg.terrain.BarrierTrack_kwargs["options"]))),
            ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])

        max_iterations = 60000
        log_interval = 50
        