#!/usr/bin/env python3

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_utils import make_env
from trainers import DQNTrainer, PPOTrainer, RandomTrainer
from config import ConfigManager

def get_trainer(trainer_type: str, env, config):
    trainer_map = {
        'dqn': DQNTrainer,
        'ppo': PPOTrainer,
        'random': RandomTrainer
    }
    
    if trainer_type.lower() not in trainer_map:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Available: {list(trainer_map.keys())}")
    
    trainer_config = {
        'learning_rate': config.training.learning_rate,
        'hidden_size': config.training.hidden_size,
        'batch_size': config.training.batch_size,
        'log_dir': config.evaluation.log_dir
    }
    
    if trainer_type.lower() == 'dqn':
        trainer_config.update({
            'epsilon_start': config.training.epsilon_start,
            'epsilon_end': config.training.epsilon_end,
            'epsilon_decay': config.training.epsilon_decay,
            'target_update': config.training.target_update,
            'memory_size': config.training.memory_size
        })
    elif trainer_type.lower() == 'ppo':
        trainer_config.update({
            'gamma': config.training.gamma,
            'gae_lambda': config.training.gae_lambda,
            'clip_epsilon': config.training.clip_epsilon,
            'value_coef': config.training.value_coef,
            'entropy_coef': config.training.entropy_coef,
            'update_epochs': config.training.update_epochs
        })
    
    return trainer_map[trainer_type.lower()](env, trainer_config)

def main():
    parser = argparse.ArgumentParser(description='Train a multi-agent RL model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--env', type=str, default='Cleaner', help='Environment name')
    parser.add_argument('--trainer', type=str, default='dqn', choices=['dqn', 'ppo', 'random'], help='Trainer type')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--save', type=str, help='Path to save trained model')
    
    args = parser.parse_args()
    
    if args.config:
        config = ConfigManager.load_config(args.config)
    else:
        config = ConfigManager.get_default_configs()['dqn_cleaner']
        config.environment.env_name = args.env
        config.trainer_type = args.trainer
        config.training.num_episodes = args.episodes
        config.environment.n_agents = args.agents
        if args.seed:
            config.environment.seed = args.seed
    
    print(f"Training Configuration:")
    print(f"  Environment: {config.environment.env_name}")
    print(f"  Trainer: {config.trainer_type}")
    print(f"  Episodes: {config.training.num_episodes}")
    print(f"  Agents: {config.environment.n_agents}")
    print(f"  Seed: {config.environment.seed}")
    
    try:
        env = make_env(
            config.environment.env_name,
            N_agent=config.environment.n_agents,
            map_size=getattr(config.environment, 'map_size', 15),
            seed=config.environment.seed
        )
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Make sure the environment exists and follows the naming convention")
        return 1
    
    trainer = get_trainer(config.trainer_type, env, config)
    
    print("\nStarting training...")
    results = trainer.train(config.training.num_episodes)
    
    print("\nTraining completed!")
    final_rewards = results['episode_rewards'][-100:] if len(results['episode_rewards']) >= 100 else results['episode_rewards']
    print(f"Final average reward (last 100 episodes): {sum(final_rewards) / len(final_rewards):.2f}")
    
    print("\nRunning evaluation...")
    eval_results = trainer.evaluate(config.evaluation.num_eval_episodes)
    print(f"Evaluation - Average reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Evaluation - Average steps: {eval_results['avg_steps']:.2f} ± {eval_results['std_steps']:.2f}")
    
    if args.save and hasattr(trainer, 'save_model'):
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        trainer.save_model(args.save)
        print(f"Model saved to {args.save}")
    
    return 0

if __name__ == "__main__":
    exit(main())