import sys
import os
import importlib.util
from typing import List, Dict, Any
from env_adapter import EnvAdapter

def make_env(env_name: str, **kwargs):
    try:
        env_module_name = f"env_{env_name}"
        module = importlib.import_module(env_module_name)
        
        class_name = f"Env{env_name}"
        if hasattr(module, class_name):
            env_class = getattr(module, class_name)
            original_env = env_class(**kwargs)
            return EnvAdapter(original_env)
        else:
            available_classes = [name for name in dir(module) if name.startswith('Env')]
            if available_classes:
                env_class = getattr(module, available_classes[0])
                original_env = env_class(**kwargs)
                return EnvAdapter(original_env)
            else:
                raise ValueError(f"No environment class found in {env_module_name}")
                
    except ImportError as e:
        env_dir = f"env_{env_name}"
        if not os.path.exists(env_dir):
            raise ValueError(f"Environment directory {env_dir} not found")
        
        sys.path.insert(0, os.path.abspath('.'))
        try:
            module = importlib.import_module(env_module_name)
            class_name = f"Env{env_name}"
            if hasattr(module, class_name):
                env_class = getattr(module, class_name)
                original_env = env_class(**kwargs)
                return EnvAdapter(original_env)
            else:
                available_classes = [name for name in dir(module) if name.startswith('Env')]
                if available_classes:
                    env_class = getattr(module, available_classes[0])
                    original_env = env_class(**kwargs)
                    return EnvAdapter(original_env)
                else:
                    raise ValueError(f"No environment class found in {env_module_name}")
        finally:
            sys.path.pop(0)

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