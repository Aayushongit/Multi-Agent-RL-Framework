import random
from typing import Dict, Any, List
from .base_trainer import BaseTrainer

class RandomTrainer(BaseTrainer):
    
    def __init__(self, env, config: Dict[str, Any]):
        super().__init__(env, config)
        self.action_space_size = env.get_action_space_size()
    
    def get_actions(self, observations: List[Any]) -> List[int]:
        return [random.randint(0, self.action_space_size - 1) for _ in range(self.env.get_num_agents())]
    
    def train_episode(self) -> Dict[str, Any]:
        self.env.reset()
        episode_rewards = []
        episode_steps = 0
        done = False
        
        while not done and episode_steps < 1000:
            obs = self.env.get_obs()
            actions = self.get_actions(obs)
            rewards, done = self.env.step(actions)
            episode_rewards.append(rewards)
            episode_steps += 1
        
        total_rewards = [sum(agent_rewards) for agent_rewards in zip(*episode_rewards)]
        
        return {
            'rewards': total_rewards,
            'steps': episode_steps,
            'extra_info': {
                'trainer_type': 'random'
            }
        }