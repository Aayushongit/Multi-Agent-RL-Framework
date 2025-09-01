from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from utils.logging_utils import TrainingLogger

class BaseTrainer(ABC):
    
    def __init__(self, env, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = TrainingLogger(config.get('log_dir', 'logs'))
        self.episode_rewards = []
        self.episode_steps = []
        
    @abstractmethod
    def train_episode(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_actions(self, observations: List[Any]) -> List[int]:
        pass
    
    def train(self, num_episodes: int) -> Dict[str, List[float]]:
        for episode in range(num_episodes):
            episode_info = self.train_episode()
            
            self.logger.log_episode(
                episode=episode,
                rewards=episode_info['rewards'],
                steps=episode_info['steps'],
                **episode_info.get('extra_info', {})
            )
            
            self.episode_rewards.append(sum(episode_info['rewards']))
            self.episode_steps.append(episode_info['steps'])
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        total_rewards = []
        total_steps = []
        
        for _ in range(num_episodes):
            self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                obs = self.env.get_obs()
                actions = self.get_actions(obs)
                rewards, done = self.env.step(actions)
                episode_reward += sum(rewards)
                episode_steps += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)
        
        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps)
        }
    
    def save_model(self, filepath: str) -> None:
        pass
    
    def load_model(self, filepath: str) -> None:
        pass