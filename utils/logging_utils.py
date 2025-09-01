import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

class TrainingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        self.logger = setup_logger("TrainingLogger", self.log_file)
        self.episode_data = []
    
    def log_episode(self, episode: int, rewards: list, steps: int, **kwargs):
        total_reward = sum(rewards)
        avg_reward = total_reward / len(rewards)
        
        log_msg = f"Episode {episode}: Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Steps: {steps}"
        for key, value in kwargs.items():
            log_msg += f", {key}: {value}"
        
        self.logger.info(log_msg)
        
        episode_info = {
            'episode': episode,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'steps': steps,
            **kwargs
        }
        self.episode_data.append(episode_info)
    
    def save_data(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.log_dir, f"episode_data_{timestamp}.json")
        
        import json
        with open(filename, 'w') as f:
            json.dump(self.episode_data, f, indent=2)