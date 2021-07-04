import collections
import numpy as np
import torch

AgentOutput = collections.namedtuple(
    "AgentOutput", ["action", "policy_logits", "baseline"])
AgentState = collections.namedtuple(
    "AgentState", ["lstm_state", "prev_action"])


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict