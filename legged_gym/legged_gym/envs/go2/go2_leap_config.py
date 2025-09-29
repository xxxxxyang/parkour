import numpy as np
import os.path as osp
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class Go2LeapCfg( Go2FieldCfg ):

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
            "sidewall_distance"
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
                "leap",
            ],
            leap= dict(
                length= [0.05, 0.8],
                depth= [0.5, 0.8],
                height= 0.2, # expected leap height over the gap
                # fake_offset= 0.1,
            ),
            track_block_length= 1.6,
            virtual_terrain= False, # Change this to False for real terrain
        ))

        TerrainPerlin_kwargs = merge_dict(Go2FieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= 0.12,
        ))
    

    class domain_rand( Go2FieldCfg.domain_rand ):
        init_base_rot_range = dict(
            roll= [-0.1, 0.1],
            pitch= [-0.1, 0.1],
        )

    class rewards( Go2FieldCfg.rewards ):
        soft_dof_pos_limit = 0.9



logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2LeapCfgPPO( Go2FieldCfgPPO ):
    class algorithm( Go2FieldCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.2
    
    class runner( Go2FieldCfgPPO.runner ):
        experiment_name = "field_go2_leap"
        resume = True
        load_run = "{Your trained walking model directory}"
        # load_run = "{Your virtually trained crawling model directory}"

        run_name = "".join(["Skills_",
        ("Multi" if len(Go2LeapCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else (Go2LeapCfg.terrain.BarrierTrack_kwargs["options"][0] if Go2LeapCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        ("_comXRange{:.1f}-{:.1f}".format(Go2LeapCfg.domain_rand.com_range.x[0], Go2LeapCfg.domain_rand.com_range.x[1])),
        ("_noLinVel" if not Go2LeapCfg.env.use_lin_vel else ""),
        ("_propDelay{:.2f}-{:.2f}".format(
                Go2LeapCfg.sensor.proprioception.latency_range[0],
                Go2LeapCfg.sensor.proprioception.latency_range[1],
            ) if Go2LeapCfg.sensor.proprioception.delay_action_obs else ""
        ),
        # ("_pPenD{:.0e}".format(Go2LeapCfg.rewards.scales.penetrate_depth) if getattr(Go2LeapCfg.rewards.scales, "penetrate_depth", 0.) != 0. else ""),
        ("_pEnergySubsteps" + np.format_float_scientific(Go2LeapCfg.rewards.scales.legs_energy_substeps, precision= 1, exp_digits= 1, trim= "-") if getattr(Go2LeapCfg.rewards.scales, "legs_energy_substeps", 0.) != 0. else ""),
        ("_pDof{:.0e}".format(-Go2LeapCfg.rewards.scales.exceed_dof_pos_limits) if getattr(Go2LeapCfg.rewards.scales, "exceed_dof_pos_limits", 0.) != 0 else ""),
        ("_pTorque" + np.format_float_scientific(-Go2LeapCfg.rewards.scales.torques, precision= 1, exp_digits= 1, trim= "-") if getattr(Go2LeapCfg.rewards.scales, "torques", 0.) != 0 else ""),
        ("_pTorqueL1{:.0e}".format(-Go2LeapCfg.rewards.scales.exceed_torque_limits_l1norm) if getattr(Go2LeapCfg.rewards.scales, "exceed_torque_limits_l1norm", 0.) != 0 else ""),
        # ("_rTilt{:.0e}".format(Go2LeapCfg.rewards.scales.tilt_cond) if getattr(Go2LeapCfg.rewards.scales, "tilt_cond", 0.) != 0 else ""),
        # ("_pYaw{:.1f}".format(-Go2LeapCfg.rewards.scales.yaw_abs) if getattr(Go2LeapCfg.rewards.scales, "yaw_abs", 0.) != 0 else ""),
        # ("_pPosY{:.1f}".format(-Go2LeapCfg.rewards.scales.lin_pos_y) if getattr(Go2LeapCfg.rewards.scales, "lin_pos_y", 0.) != 0 else ""),
        # ("_pCollision{:.1f}".format(-Go2LeapCfg.rewards.scales.collision) if getattr(Go2LeapCfg.rewards.scales, "collision", 0.) != 0 else ""),
        # ("_kp{:d}".format(int(Go2LeapCfg.control.stiffness["joint"])) if Go2LeapCfg.control.stiffness["joint"] != 50 else ""),
        ("_noDelayActObs" if not Go2LeapCfg.sensor.proprioception.delay_action_obs else ""),
        ("_noTanh"),
        ("_virtual" if Go2LeapCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])
        max_iterations = 20000
        save_interval = 1000
        log_interval = 50
    