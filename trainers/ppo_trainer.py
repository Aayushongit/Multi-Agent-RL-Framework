import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from typing import Dict, Any, List
from .base_trainer import BaseTrainer

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

class PPOTrainer(BaseTrainer):
    
    def __init__(self, env, config: Dict[str, Any]):
        super().__init__(env, config)
        self.obs_size = env.get_obs_space_size()
        self.action_size = env.get_action_space_size()
        self.n_agents = env.get_num_agents()
        
        self.hidden_size = config.get('hidden_size', 128)
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.update_epochs = config.get('update_epochs', 4)
        self.batch_size = config.get('batch_size', 64)
        
        self.networks = [ActorCritic(self.obs_size, self.hidden_size, self.action_size).to(self.device) 
                        for _ in range(self.n_agents)]
        self.optimizers = [optim.Adam(net.parameters(), lr=self.learning_rate) 
                          for net in self.networks]
        
        self.trajectories = [[] for _ in range(self.n_agents)]
    
    def get_actions(self, observations: List[Any]) -> List[int]:
        actions = []
        action_log_probs = []
        state_values = []
        
        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_probs, state_value = self.networks[i](obs_tensor)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            
            actions.append(action.item())
            action_log_probs.append(action_log_prob)
            state_values.append(state_value)
        
        return actions, action_log_probs, state_values
    
    def compute_gae(self, rewards: List[float], values: List[torch.Tensor], dones: List[bool]):
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
    
    def update_networks(self):
        for agent_id in range(self.n_agents):
            if not self.trajectories[agent_id]:
                continue
            
            observations = [t[0] for t in self.trajectories[agent_id]]
            actions = [t[1] for t in self.trajectories[agent_id]]
            old_log_probs = [t[2] for t in self.trajectories[agent_id]]
            rewards = [t[3] for t in self.trajectories[agent_id]]
            values = [t[4] for t in self.trajectories[agent_id]]
            dones = [t[5] for t in self.trajectories[agent_id]]
            
            advantages, returns = self.compute_gae(rewards, values, dones)
            
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            old_log_probs = torch.stack(old_log_probs).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            observations = torch.FloatTensor(observations).to(self.device)
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for _ in range(self.update_epochs):
                action_probs, state_values = self.networks[agent_id](observations)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                value_loss = F.mse_loss(state_values.squeeze(), returns)
                
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizers[agent_id].zero_grad()
                total_loss.backward()
                self.optimizers[agent_id].step()
            
            self.trajectories[agent_id] = []
    
    def train_episode(self) -> Dict[str, Any]:
        self.env.reset()
        episode_rewards = []
        episode_steps = 0
        done = False
        
        while not done and episode_steps < 1000:
            obs = self.env.get_obs()
            actions, action_log_probs, state_values = self.get_actions(obs)
            rewards, done = self.env.step(actions)
            
            for i in range(self.n_agents):
                self.trajectories[i].append((
                    obs[i], actions[i], action_log_probs[i], 
                    rewards[i], state_values[i], done
                ))
            
            episode_rewards.append(rewards)
            episode_steps += 1
        
        self.update_networks()
        
        total_rewards = [sum(agent_rewards) for agent_rewards in zip(*episode_rewards)]
        
        return {
            'rewards': total_rewards,
            'steps': episode_steps,
            'extra_info': {
                'trainer_type': 'ppo'
            }
        }