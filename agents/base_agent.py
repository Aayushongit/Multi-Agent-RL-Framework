from abc import ABC, abstractmethod
import torch
from typing import Any, List

class BaseAgent(ABC):
    
    def __init__(self, obs_size: int, action_size: int, agent_id: int = 0):
        self.obs_size = obs_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def get_action(self, observation: Any) -> int:
        pass
    
    @abstractmethod
    def update(self, experience: dict) -> None:
        pass
    
    def save(self, filepath: str) -> None:
        pass
    
    def load(self, filepath: str) -> None:
        pass
    
    def set_training_mode(self, training: bool = True) -> None:
        self.training = training
    
    def reset(self) -> None:
        pass