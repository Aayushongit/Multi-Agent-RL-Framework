import sys
import os
import importlib.util
from typing import List, Dict, Any

def make_env(env_name: str, **kwargs):
    env_dir = f"env_{env_name}"
    env_file = f"env_{env_name}.py"
    env_path = os.path.join(env_dir, env_file)
    
    if not os.path.exists(env_path):
        raise ValueError(f"Environment {env_name} not found at {env_path}")
    
    spec = importlib.util.spec_from_file_location(f"env_{env_name}", env_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    class_name = f"Env{env_name}"
    if hasattr(module, class_name):
        env_class = getattr(module, class_name)
        return env_class(**kwargs)
    else:
        available_classes = [name for name in dir(module) if name.startswith('Env')]
        if available_classes:
            env_class = getattr(module, available_classes[0])
            return env_class(**kwargs)
        else:
            raise ValueError(f"No environment class found in {env_path}")

def get_env_list() -> List[str]:
    env_dirs = [d for d in os.listdir('.') if d.startswith('env_')]
    env_names = [d.replace('env_', '') for d in env_dirs]
    return sorted(env_names)

def validate_action_list(action_list: List[int], n_agents: int, action_space_size: int) -> bool:
    if len(action_list) != n_agents:
        return False
    return all(0 <= action < action_space_size for action in action_list)

def get_random_actions(n_agents: int, action_space_size: int) -> List[int]:
    import random
    return [random.randint(0, action_space_size - 1) for _ in range(n_agents)]