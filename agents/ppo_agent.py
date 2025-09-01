import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from typing import Any, List
from .base_agent import BaseAgent

class ActorCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

class PPOAgent(BaseAgent):
    
    def __init__(self, obs_size: int, action_size: int, agent_id: int = 0,
                 hidden_size: int = 128, learning_rate: float = 0.0003,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, update_epochs: int = 4):
        super().__init__(obs_size, action_size, agent_id)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        
        self.network = ActorCritic(obs_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.trajectory = []
        self.training = True
    
    def get_action(self, observation: Any) -> int:
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        action_probs, state_value = self.network(obs_tensor)
        
        if self.training:
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            
            self.last_action_log_prob = action_log_prob
            self.last_state_value = state_value
            
            return action.item()
        else:
            return action_probs.argmax().item()
    
    def update(self, experience: dict) -> None:
        if hasattr(self, 'last_action_log_prob') and hasattr(self, 'last_state_value'):
            experience['action_log_prob'] = self.last_action_log_prob
            experience['state_value'] = self.last_state_value
        
        self.trajectory.append(experience)
        
        if experience.get('done', False):
            self._update_network()
            self.trajectory = []
    
    def _compute_gae(self, rewards: List[float], values: List[torch.Tensor], dones: List[bool]):
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[i])
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def _update_network(self):
        if not self.trajectory:
            return
        
        observations = [t['state'] for t in self.trajectory]
        actions = [t['action'] for t in self.trajectory]
        rewards = [t['reward'] for t in self.trajectory]
        dones = [t['done'] for t in self.trajectory]
        
        if 'action_log_prob' in self.trajectory[0] and 'state_value' in self.trajectory[0]:
            old_log_probs = [t['action_log_prob'] for t in self.trajectory]
            values = [t['state_value'] for t in self.trajectory]
        else:
            return
        
        advantages, returns = self._compute_gae(rewards, values, dones)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        observations = torch.FloatTensor(observations).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.update_epochs):
            action_probs, state_values = self.network(observations)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = F.mse_loss(state_values.squeeze(), returns)
            
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
    
    def save(self, filepath: str) -> None:
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])