#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_utils import make_env
from trainers import DQNTrainer, PPOTrainer, RandomTrainer
from evaluation import Benchmarker, ResultsVisualizer
from config import ConfigManager

def main():
    config = ConfigManager.get_default_configs()['dqn_cleaner']
    
    env = make_env(
        config.environment.env_name,
        N_agent=config.environment.n_agents,
        map_size=config.environment.map_size,
        seed=config.environment.seed
    )
    
    base_config = {
        'log_dir': config.evaluation.log_dir,
        'hidden_size': config.training.hidden_size
    }
    
    trainers = {
        'DQN': DQNTrainer(env, {
            **base_config,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995
        }),
        'PPO': PPOTrainer(env, {
            **base_config,
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'gae_lambda': 0.95
        }),
        'Random': RandomTrainer(env, base_config)
    }
    
    benchmarker = Benchmarker("benchmark_results")
    visualizer = ResultsVisualizer("benchmark_plots")
    
    results = {}
    
    print("Starting benchmark comparison...")
    
    for name, trainer in trainers.items():
        print(f"\nBenchmarking {name}...")
        result = benchmarker.benchmark_trainer(
            trainer, env, 
            num_episodes=500, 
            num_runs=3
        )
        results[name] = result
    
    comparison = benchmarker.compare_trainers(list(results.values()))
    print("\nBenchmark Results:")
    print(f"Performance Ranking: {comparison['performance_ranking']}")
    print(f"Training Time Ranking: {comparison['training_time_ranking']}")
    print(f"Convergence Speed Ranking: {comparison['convergence_ranking']}")
    
    learning_curves = {}
    for name, result in results.items():
        all_rewards = []
        for run in result['raw_results']:
            all_rewards.extend(run['episode_rewards'])
        learning_curves[name] = all_rewards
    
    visualizer.plot_learning_curves(learning_curves, "Benchmark Learning Curves")
    visualizer.plot_training_comparison(results)
    visualizer.create_summary_report(results)
    
    benchmarker.save_results("benchmark_results.json")
    
    print("\nBenchmark completed! Check benchmark_plots/ directory for visualizations.")

if __name__ == "__main__":
    main()