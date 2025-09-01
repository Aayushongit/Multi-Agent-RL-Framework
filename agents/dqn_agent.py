import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Any
from .base_agent import BaseAgent

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

class DQNAgent(BaseAgent):
    
    def __init__(self, obs_size: int, action_size: int, agent_id: int = 0, 
                 hidden_size: int = 128, learning_rate: float = 0.001,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, 
                 epsilon_min: float = 0.01, memory_size: int = 10000,
                 batch_size: int = 32, target_update: int = 10):
        super().__init__(obs_size, action_size, agent_id)
        
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        self.q_network = DQN(obs_size, hidden_size, action_size).to(self.device)
        self.target_network = DQN(obs_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.training = True
    
    def get_action(self, observation: Any) -> int:
        if self.training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        q_values = self.q_network(obs_tensor)
        return q_values.max(1)[1].item()
    
    def update(self, experience: dict) -> None:
        self.memory.append(experience)
        
        if len(self.memory) >= self.batch_size:
            self._replay()
        
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _replay(self):
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e['state'] for e in batch]).to(self.device)
        actions = torch.LongTensor([e['action'] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e['next_state'] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e['done'] for e in batch]).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (0.99 * next_q * ~dones)
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self, filepath: str) -> None:
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']