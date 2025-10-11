"""
# A python module that manipulates torch checkpoint file in a hacky way.
Each function should be used with caution and should be used only when thoughtfully considered.
---
Args:
    source_state_dict: the state_dict loaded using torch.load
    algo_state_dict: the algorithm state_dict summarized from algorithm as an example
---
Returns:
    new_state_dict: the state_dict that has been manipulated or directly saved as a checkpoint file.
"""
import torch
from collections import OrderedDict
#! replace encoder0 with Conv2d, 其他的 actor, critic, memory_a, memory_c, critiic_encoders, memory_s, state_estimator都用的teacher
def replace_encoder0(source_state_dict, algo_state_dict):
    print("\033[1;36m Replacing encoder.0 weights with untrained weights and avoid critic_encoder.0 \033[0m")
    new_model_state_dict = OrderedDict()
    print("source_state_dict keys:", source_state_dict["model_state_dict"].keys())

    for key in algo_state_dict["model_state_dict"].keys():
        if "critic_encoders.0" in key:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
        elif "encoders.0" in key:   #! 把encoder0 从 MLP 改成 Conv2dHeadModel 
            print(
                "key:", key,
                "shape:", algo_state_dict["model_state_dict"][key].shape,
                "using untrained module weights.")
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict= new_model_state_dict,
        # No optimizer_state_dict
        iter= source_state_dict["iter"],
        infos= source_state_dict["infos"],
    )
    print("new_model_state_dict keys:", new_model_state_dict.keys())

    return new_state_dict
