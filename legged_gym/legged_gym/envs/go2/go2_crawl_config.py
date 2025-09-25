import numpy as np
import os.path as osp
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class Go2CrawlCfg( Go2FieldCfg ):

    class env( Go2FieldCfg.env ):
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
        ]

    #### uncomment this to train non-virtual terrain
    class sensor( Go2FieldCfg.sensor ):
        class proprioception( Go2FieldCfg.sensor.proprioception ):
            latency_range = [0.04-0.0025, 0.04+0.0075]
            delay_action_obs = False
    #### uncomment the above to train non-virtual terrain
    
    class terrain( Go2FieldCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = True

        BarrierTrack_kwargs = merge_dict(Go2FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "crawl",
            ],
            track_block_length= 1.6,
            crawl= dict(
                height= (0.25, 0.5),
                depth= (0.1, 0.6), # size along the forward axis
                wall_height= 0.6,
                no_perlin_at_obstacle= False,
            ),
            virtual_terrain= False, # Change this to False for real terrain
        ))

        TerrainPerlin_kwargs = merge_dict(Go2FieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= 0.12,
        ))
    
    class commands( Go2FieldCfg.commands ):
        class ranges( Go2FieldCfg.commands.ranges ):
            lin_vel_x = [0.3, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class asset( Go2FieldCfg.asset ):
        terminate_after_contacts_on = ["base"]

    class termination( Go2FieldCfg.termination ):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]
        roll_kwargs = dict(
            threshold= 1.4, # [rad]
        )
        pitch_kwargs = dict(
            threshold= 1.6, # [rad]
        )
        z_low_kwargs = dict(
            threshold= 0.15, # [m]
        )
        z_high_kwargs = dict(
            threshold= 1.5, # [m]
        )
        out_of_track_kwargs = dict(
            threshold= 1., # [m]
        )
        timeout_at_border = True
        timeout_at_finished = False

    class domain_rand( Go2FieldCfg.domain_rand ):
        init_base_rot_range = dict(
            roll= [-0.1, 0.1],
            pitch= [-0.1, 0.1],
        )

    class rewards( Go2FieldCfg.rewards ):
        # class scales:
        #     tracking_lin_vel = 1.0
        #     tracking_ang_vel = 0.05
        #     # world_vel_l2norm = -1.
        #     # legs_energy_substeps = -1e-5
        #     alive = 0.1
        #     # penetrate_depth = -6e-2 # comment this out if trianing non-virtual terrain
        #     # penetrate_volume = -6e-2 # comment this out if trianing non-virtual terrain
        #     exceed_dof_pos_limits = -8e-1
        #     # exceed_torque_limits_i = -2e-1
        #     exceed_torque_limits_l1norm = -4e-1
        #     # collision = -0.05
        #     # tilt_cond = 0.1
        #     torques = -1e-5
        #     yaw_abs = -0.1
        #     lin_pos_y = -0.1

        soft_dof_pos_limit = 0.9

    # class curriculum( Go2FieldCfg.curriculum ):
    #     penetrate_volume_threshold_harder = 1500
    #     penetrate_volume_threshold_easier = 10000
    #     penetrate_depth_threshold_harder = 10
    #     penetrate_depth_threshold_easier = 400


logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2CrawlCfgPPO( Go2FieldCfgPPO ):
    class algorithm( Go2FieldCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.1
    
    class runner( Go2FieldCfgPPO.runner ):
        experiment_name = "field_go2_crawl"
        resume = True
        load_run = "{Your traind walking model directory}"
        # load_run = "{Your virtually trained crawling model directory}"

        run_name = "".join(["Skills_",
        ("Multi" if len(Go2CrawlCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else (Go2CrawlCfg.terrain.BarrierTrack_kwargs["options"][0] if Go2CrawlCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        ("_comXRange{:.1f}-{:.1f}".format(Go2CrawlCfg.domain_rand.com_range.x[0], Go2CrawlCfg.domain_rand.com_range.x[1])),
        ("_noLinVel" if not Go2CrawlCfg.env.use_lin_vel else ""),
        ("_propDelay{:.2f}-{:.2f}".format(
                Go2CrawlCfg.sensor.proprioception.latency_range[0],
                Go2CrawlCfg.sensor.proprioception.latency_range[1],
            ) if Go2CrawlCfg.sensor.proprioception.delay_action_obs else ""
        ),
        # ("_pPenD{:.0e}".format(Go2CrawlCfg.rewards.scales.penetrate_depth) if getattr(Go2CrawlCfg.rewards.scales, "penetrate_depth", 0.) != 0. else ""),
        ("_pEnergySubsteps" + np.format_float_scientific(Go2CrawlCfg.rewards.scales.legs_energy_substeps, precision= 1, exp_digits= 1, trim= "-") if getattr(Go2CrawlCfg.rewards.scales, "legs_energy_substeps", 0.) != 0. else ""),
        ("_pDof{:.0e}".format(-Go2CrawlCfg.rewards.scales.exceed_dof_pos_limits) if getattr(Go2CrawlCfg.rewards.scales, "exceed_dof_pos_limits", 0.) != 0 else ""),
        ("_pTorque" + np.format_float_scientific(-Go2CrawlCfg.rewards.scales.torques, precision= 1, exp_digits= 1, trim= "-") if getattr(Go2CrawlCfg.rewards.scales, "torques", 0.) != 0 else ""),
        ("_pTorqueL1{:.0e}".format(-Go2CrawlCfg.rewards.scales.exceed_torque_limits_l1norm) if getattr(Go2CrawlCfg.rewards.scales, "exceed_torque_limits_l1norm", 0.) != 0 else ""),
        # ("_rTilt{:.0e}".format(Go2CrawlCfg.rewards.scales.tilt_cond) if getattr(Go2CrawlCfg.rewards.scales, "tilt_cond", 0.) != 0 else ""),
        # ("_pYaw{:.1f}".format(-Go2CrawlCfg.rewards.scales.yaw_abs) if getattr(Go2CrawlCfg.rewards.scales, "yaw_abs", 0.) != 0 else ""),
        # ("_pPosY{:.1f}".format(-Go2CrawlCfg.rewards.scales.lin_pos_y) if getattr(Go2CrawlCfg.rewards.scales, "lin_pos_y", 0.) != 0 else ""),
        # ("_pCollision{:.1f}".format(-Go2CrawlCfg.rewards.scales.collision) if getattr(Go2CrawlCfg.rewards.scales, "collision", 0.) != 0 else ""),
        # ("_kp{:d}".format(int(Go2CrawlCfg.control.stiffness["joint"])) if Go2CrawlCfg.control.stiffness["joint"] != 50 else ""),
        ("_noDelayActObs" if not Go2CrawlCfg.sensor.proprioception.delay_action_obs else ""),
        ("_noTanh"),
        ("_virtual" if Go2CrawlCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])
        max_iterations = 20000
        save_interval = 500
        log_interval = 100
    