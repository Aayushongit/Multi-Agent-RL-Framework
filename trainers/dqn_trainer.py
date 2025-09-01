import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, Any, List
from .base_trainer import BaseTrainer

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNTrainer(BaseTrainer):
    
    def __init__(self, env, config: Dict[str, Any]):
        super().__init__(env, config)
        self.obs_size = env.get_obs_space_size()
        self.action_size = env.get_action_space_size()
        self.n_agents = env.get_num_agents()
        
        self.hidden_size = config.get('hidden_size', 128)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.target_update = config.get('target_update', 10)
        self.memory_size = config.get('memory_size', 10000)
        
        self.q_networks = [DQN(self.obs_size, self.hidden_size, self.action_size).to(self.device) 
                          for _ in range(self.n_agents)]
        self.target_networks = [DQN(self.obs_size, self.hidden_size, self.action_size).to(self.device) 
                               for _ in range(self.n_agents)]
        self.optimizers = [optim.Adam(net.parameters(), lr=self.learning_rate) 
                          for net in self.q_networks]
        self.replay_buffers = [ReplayBuffer(self.memory_size) for _ in range(self.n_agents)]
        
        for i in range(self.n_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
    
    def get_actions(self, observations: List[Any]) -> List[int]:
        actions = []
        for i, obs in enumerate(observations):
            if random.random() > self.epsilon:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_values = self.q_networks[i](obs_tensor)
                action = q_values.max(1)[1].item()
            else:
                action = random.randint(0, self.action_size - 1)
            actions.append(action)
        return actions
    
    def update_networks(self):
        for i in range(self.n_agents):
            if len(self.replay_buffers[i]) < self.batch_size:
                continue
            
            states, actions, rewards, next_states, dones = self.replay_buffers[i].sample(self.batch_size)
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            
            current_q = self.q_networks[i](states).gather(1, actions.unsqueeze(1))
            next_q = self.target_networks[i](next_states).max(1)[0].detach()
            target_q = rewards + (0.99 * next_q * ~dones)
            
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()
    
    def train_episode(self) -> Dict[str, Any]:
        self.env.reset()
        episode_rewards = []
        episode_steps = 0
        done = False
        
        prev_obs = self.env.get_obs()
        
        while not done and episode_steps < 1000:
            actions = self.get_actions(prev_obs)
            rewards, done = self.env.step(actions)
            next_obs = self.env.get_obs()
            
            for i in range(self.n_agents):
                self.replay_buffers[i].push(prev_obs[i], actions[i], rewards[i], 
                                          next_obs[i], done)
            
            prev_obs = next_obs
            episode_rewards.append(rewards)
            episode_steps += 1
            
            self.update_networks()
        
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        if episode_steps % self.target_update == 0:
            for i in range(self.n_agents):
                self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
        
        total_rewards = [sum(agent_rewards) for agent_rewards in zip(*episode_rewards)]
        
        return {
            'rewards': total_rewards,
            'steps': episode_steps,
            'extra_info': {
                'trainer_type': 'dqn',
                'epsilon': self.epsilon
            }
        }