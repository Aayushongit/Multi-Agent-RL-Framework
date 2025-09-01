import random
from typing import Any
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    
    def __init__(self, obs_size: int, action_size: int, agent_id: int = 0):
        super().__init__(obs_size, action_size, agent_id)
    
    def get_action(self, observation: Any) -> int:
        return random.randint(0, self.action_size - 1)
    
    def update(self, experience: dict) -> None:
        pass