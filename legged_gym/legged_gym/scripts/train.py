# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
np.float = np.float32
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import class_to_dict
import torch

from legged_gym.debugger import break_into_debugger
import faulthandler
faulthandler.enable()

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # DEBUG: 打印 obs layout 与维度，帮助定位 RNN/模型输入不匹配问题
    print("CONFIG task:", args.task)
    print("env_cfg.env.obs_components:", getattr(env_cfg.env, "obs_components", None))
    print("env.obs_segments:", getattr(env, "obs_segments", None))
    print("env.num_obs:", getattr(env, "num_obs", None))
    print("env.num_privileged_obs:", getattr(env, "num_privileged_obs", None))
    reward_dict = class_to_dict(getattr(env_cfg, "rewards", None))
    commands_dict = class_to_dict(getattr(env_cfg, "commands", None))
    print("reward:", reward_dict)
    print("commands:", commands_dict)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    if args.headless:
        print("Running in headless mode")
        os.environ["MPLBACKEND"] = "Agg"
    args.task = "go2_distillleap"
    # args.task = "go2_leap"
    args.headless = False
    train(args)
