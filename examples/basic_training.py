#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_utils import make_env
from trainers import DQNTrainer, RandomTrainer
from config import ConfigManager

def main():
    config = ConfigManager.get_default_configs()['dqn_cleaner']
    
    env = make_env(
        config.environment.env_name,
        N_agent=config.environment.n_agents,
        map_size=config.environment.map_size,
        seed=config.environment.seed
    )
    
    trainer_config = {
        'learning_rate': config.training.learning_rate,
        'hidden_size': config.training.hidden_size,
        'batch_size': config.training.batch_size,
        'epsilon_start': config.training.epsilon_start,
        'epsilon_end': config.training.epsilon_end,
        'epsilon_decay': config.training.epsilon_decay,
        'target_update': config.training.target_update,
        'memory_size': config.training.memory_size,
        'log_dir': config.evaluation.log_dir
    }
    
    trainer = DQNTrainer(env, trainer_config)
    
    print(f"Starting training with {config.trainer_type} on {config.environment.env_name} environment")
    print(f"Training for {config.training.num_episodes} episodes")
    
    results = trainer.train(config.training.num_episodes)
    
    print("Training completed!")
    print(f"Final average reward: {sum(results['episode_rewards'][-100:]) / 100:.2f}")
    
    print("\nEvaluating trained model...")
    eval_results = trainer.evaluate(config.evaluation.num_eval_episodes)
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    main()