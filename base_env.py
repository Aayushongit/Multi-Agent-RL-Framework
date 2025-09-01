from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Any, Dict, Optional

class BaseMultiAgentEnv(ABC):
    
    def __init__(self, n_agents: int, seed: Optional[int] = None):
        self.n_agents = n_agents
        self.seed = seed
        self._episode_step = 0
        self._max_episode_steps = 1000
        
        if seed is not None:
            np.random.seed(seed)
    
    @abstractmethod
    def reset(self) -> None:
        self._episode_step = 0
    
    @abstractmethod
    def step(self, action_list: List[int]) -> Tuple[List[float], bool]:
        self._episode_step += 1
        
    @abstractmethod
    def get_obs(self) -> List[Any]:
        pass
    
    @abstractmethod
    def get_action_space_size(self) -> int:
        pass
    
    @abstractmethod
    def get_obs_space_size(self) -> int:
        pass
    
    def get_num_agents(self) -> int:
        return self.n_agents
    
    def get_episode_step(self) -> int:
        return self._episode_step
    
    def is_episode_done(self) -> bool:
        return self._episode_step >= self._max_episode_steps
    
    def set_max_episode_steps(self, max_steps: int) -> None:
        self._max_episode_steps = max_steps
    
    def get_env_info(self) -> Dict[str, Any]:
        return {
            'n_agents': self.n_agents,
            'action_space_size': self.get_action_space_size(),
            'obs_space_size': self.get_obs_space_size(),
            'max_episode_steps': self._max_episode_steps,
            'episode_step': self._episode_step
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        pass
    
    def close(self) -> None:
        pass