#!/usr/bin/env python3

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_utils import make_env, get_env_list
from trainers import DQNTrainer, PPOTrainer, RandomTrainer
from evaluation import Benchmarker, ResultsVisualizer
from config import ConfigManager

def create_trainer(trainer_type: str, env, base_config: dict):
    trainers_map = {
        'dqn': lambda: DQNTrainer(env, {
            **base_config,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update': 10,
            'memory_size': 10000
        }),
        'ppo': lambda: PPOTrainer(env, {
            **base_config,
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'update_epochs': 4
        }),
        'random': lambda: RandomTrainer(env, base_config)
    }
    
    if trainer_type.lower() not in trainers_map:
        raise ValueError(f"Unknown trainer: {trainer_type}. Available: {list(trainers_map.keys())}")
    
    return trainers_map[trainer_type.lower()]()

def main():
    parser = argparse.ArgumentParser(description='Benchmark multi-agent RL trainers')
    parser.add_argument('--env', type=str, default='Cleaner', help='Environment name')
    parser.add_argument('--trainers', nargs='+', default=['dqn', 'ppo', 'random'], 
                       choices=['dqn', 'ppo', 'random'], help='Trainers to benchmark')
    parser.add_argument('--episodes', type=int, default=500, help='Episodes per run')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per trainer')
    parser.add_argument('--agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--output', type=str, default='benchmark_results', help='Output directory')
    parser.add_argument('--list-envs', action='store_true', help='List available environments')
    
    args = parser.parse_args()
    
    if args.list_envs:
        envs = get_env_list()
        print("Available environments:")
        for env in envs:
            print(f"  - {env}")
        return 0
    
    print(f"Benchmark Configuration:")
    print(f"  Environment: {args.env}")
    print(f"  Trainers: {args.trainers}")
    print(f"  Episodes per run: {args.episodes}")
    print(f"  Runs per trainer: {args.runs}")
    print(f"  Agents: {args.agents}")
    print(f"  Seed: {args.seed}")
    print(f"  Output directory: {args.output}")
    
    try:
        env = make_env(
            args.env,
            N_agent=args.agents,
            map_size=15,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error creating environment: {e}")
        return 1
    
    base_config = {
        'log_dir': f"{args.output}/logs",
        'hidden_size': 128
    }
    
    benchmarker = Benchmarker(f"{args.output}/logs")
    visualizer = ResultsVisualizer(f"{args.output}/plots")
    
    results = {}
    trainers_created = {}
    
    print(f"\nCreating trainers...")
    for trainer_name in args.trainers:
        try:
            trainer = create_trainer(trainer_name, env, base_config)
            trainers_created[trainer_name.upper()] = trainer
            print(f"  ✓ {trainer_name.upper()}")
        except Exception as e:
            print(f"  ✗ {trainer_name.upper()}: {e}")
    
    if not trainers_created:
        print("No trainers could be created. Exiting.")
        return 1
    
    print(f"\nStarting benchmark with {len(trainers_created)} trainers...")
    
    for name, trainer in trainers_created.items():
        print(f"\n{'='*50}")
        print(f"Benchmarking {name}")
        print(f"{'='*50}")
        
        try:
            result = benchmarker.benchmark_trainer(
                trainer, env, 
                num_episodes=args.episodes, 
                num_runs=args.runs
            )
            results[name] = result
            
            print(f"✓ {name} completed")
            print(f"  Final performance: {result['final_performance']['mean']:.2f} ± {result['final_performance']['std']:.2f}")
            print(f"  Training time: {result['training_time']['mean']:.2f}s ± {result['training_time']['std']:.2f}s")
            print(f"  Convergence: {result['convergence']['mean_episodes']:.0f} episodes")
            
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    if not results:
        print("No successful benchmark runs. Exiting.")
        return 1
    
    print(f"\n{'='*50}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*50}")
    
    comparison = benchmarker.compare_trainers(list(results.values()))
    print(f"Performance Ranking: {' > '.join(comparison['performance_ranking'])}")
    print(f"Speed Ranking: {' > '.join(comparison['training_time_ranking'])}")
    print(f"Convergence Ranking: {' > '.join(comparison['convergence_ranking'])}")
    
    print(f"\nGenerating visualizations...")
    
    learning_curves = {}
    for name, result in results.items():
        all_rewards = []
        for run in result['raw_results']:
            all_rewards.extend(run['episode_rewards'])
        learning_curves[name] = all_rewards
    
    try:
        visualizer.plot_learning_curves(learning_curves, f"Benchmark Results - {args.env}")
        visualizer.plot_training_comparison(results)
        visualizer.create_summary_report(results)
        print(f"✓ Plots saved to {args.output}/plots/")
    except Exception as e:
        print(f"✗ Visualization error: {e}")
    
    try:
        benchmarker.save_results(f"{args.output}/benchmark_results.json")
        print(f"✓ Results saved to {args.output}/benchmark_results.json")
    except Exception as e:
        print(f"✗ Save error: {e}")
    
    return 0

if __name__ == "__main__":
    exit(main())