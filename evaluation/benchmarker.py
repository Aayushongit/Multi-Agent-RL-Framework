import time
import numpy as np
from typing import Dict, List, Any, Tuple
from utils.logging_utils import setup_logger
from .metrics import MetricsCollector

class Benchmarker:
    
    def __init__(self, log_dir: str = "benchmark_logs"):
        self.logger = setup_logger("Benchmarker", f"{log_dir}/benchmark.log")
        self.metrics_collector = MetricsCollector()
        self.results = {}
    
    def benchmark_trainer(self, trainer, env, num_episodes: int = 100, 
                         num_runs: int = 5) -> Dict[str, Any]:
        trainer_name = trainer.__class__.__name__
        self.logger.info(f"Starting benchmark for {trainer_name}")
        
        run_results = []
        
        for run in range(num_runs):
            self.logger.info(f"Run {run + 1}/{num_runs}")
            start_time = time.time()
            
            results = trainer.train(num_episodes)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            evaluation_results = trainer.evaluate(num_episodes=20)
            
            run_result = {
                'run': run,
                'training_time': training_time,
                'episode_rewards': results['episode_rewards'],
                'episode_steps': results['episode_steps'],
                'eval_avg_reward': evaluation_results['avg_reward'],
                'eval_std_reward': evaluation_results['std_reward'],
                'eval_avg_steps': evaluation_results['avg_steps']
            }
            
            run_results.append(run_result)
        
        benchmark_result = self._aggregate_results(trainer_name, run_results)
        self.results[trainer_name] = benchmark_result
        
        self.logger.info(f"Benchmark completed for {trainer_name}")
        return benchmark_result
    
    def _aggregate_results(self, trainer_name: str, run_results: List[Dict]) -> Dict[str, Any]:
        training_times = [r['training_time'] for r in run_results]
        eval_rewards = [r['eval_avg_reward'] for r in run_results]
        eval_steps = [r['eval_avg_steps'] for r in run_results]
        
        final_rewards = []
        convergence_episodes = []
        
        for result in run_results:
            rewards = result['episode_rewards']
            final_rewards.append(np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards))
            
            window_size = 10
            moving_avg = []
            for i in range(window_size, len(rewards)):
                avg = np.mean(rewards[i-window_size:i])
                moving_avg.append(avg)
            
            if len(moving_avg) > 20:
                threshold = np.percentile(moving_avg[-20:], 90)
                for i, avg in enumerate(moving_avg):
                    if avg >= threshold:
                        convergence_episodes.append(i + window_size)
                        break
                else:
                    convergence_episodes.append(len(rewards))
            else:
                convergence_episodes.append(len(rewards))
        
        return {
            'trainer_name': trainer_name,
            'num_runs': len(run_results),
            'training_time': {
                'mean': np.mean(training_times),
                'std': np.std(training_times),
                'min': np.min(training_times),
                'max': np.max(training_times)
            },
            'final_performance': {
                'mean': np.mean(final_rewards),
                'std': np.std(final_rewards),
                'min': np.min(final_rewards),
                'max': np.max(final_rewards)
            },
            'evaluation_rewards': {
                'mean': np.mean(eval_rewards),
                'std': np.std(eval_rewards),
                'min': np.min(eval_rewards),
                'max': np.max(eval_rewards)
            },
            'convergence': {
                'mean_episodes': np.mean(convergence_episodes),
                'std_episodes': np.std(convergence_episodes)
            },
            'raw_results': run_results
        }
    
    def compare_trainers(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        comparison = {
            'trainers': [],
            'training_time_ranking': [],
            'performance_ranking': [],
            'convergence_ranking': []
        }
        
        for result in results:
            comparison['trainers'].append(result['trainer_name'])
        
        sorted_by_time = sorted(results, key=lambda x: x['training_time']['mean'])
        sorted_by_performance = sorted(results, key=lambda x: x['final_performance']['mean'], reverse=True)
        sorted_by_convergence = sorted(results, key=lambda x: x['convergence']['mean_episodes'])
        
        comparison['training_time_ranking'] = [r['trainer_name'] for r in sorted_by_time]
        comparison['performance_ranking'] = [r['trainer_name'] for r in sorted_by_performance]
        comparison['convergence_ranking'] = [r['trainer_name'] for r in sorted_by_convergence]
        
        return comparison
    
    def save_results(self, filename: str):
        import json
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.logger.info(f"Results saved to {filename}")