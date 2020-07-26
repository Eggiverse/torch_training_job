import importlib.util
from pathlib import Path
from typing import Any, Dict, Union

import torch
import yaml


def get_device(config: Union[str, int]) -> torch.device:
    if config == "cpu":
        return torch.device('cpu')

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count == 1:
            return torch.device('cuda')

        return torch.device(f'cuda:{config}')

    return torch.device('cpu')

def load_mod(mod_path: Union[str, Path]):
    mod_path = Path(mod_path)
    mymod_spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
    my_mod = importlib.util.module_from_spec(mymod_spec)
    mymod_spec.loader.exec_module(my_mod)
    return my_mod

def load_config(config_file: Union[str, Path] = 'boltz_config.yml') -> Dict[str, Any]:
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return config