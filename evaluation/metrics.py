import numpy as np
from typing import List, Dict, Any

class MetricsCollector:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_times = []
        self.custom_metrics = {}
    
    def add_episode(self, rewards: List[float], steps: int, time_taken: float, **kwargs):
        self.episode_rewards.append(rewards)
        self.episode_steps.append(steps)
        self.episode_times.append(time_taken)
        
        for key, value in kwargs.items():
            if key not in self.custom_metrics:
                self.custom_metrics[key] = []
            self.custom_metrics[key].append(value)
    
    def get_reward_statistics(self) -> Dict[str, float]:
        if not self.episode_rewards:
            return {}
        
        total_rewards = [sum(rewards) for rewards in self.episode_rewards]
        avg_agent_rewards = [np.mean(rewards) for rewards in self.episode_rewards]
        
        return {
            'total_reward_mean': np.mean(total_rewards),
            'total_reward_std': np.std(total_rewards),
            'total_reward_max': np.max(total_rewards),
            'total_reward_min': np.min(total_rewards),
            'avg_agent_reward_mean': np.mean(avg_agent_rewards),
            'avg_agent_reward_std': np.std(avg_agent_rewards)
        }
    
    def get_step_statistics(self) -> Dict[str, float]:
        if not self.episode_steps:
            return {}
        
        return {
            'steps_mean': np.mean(self.episode_steps),
            'steps_std': np.std(self.episode_steps),
            'steps_max': np.max(self.episode_steps),
            'steps_min': np.min(self.episode_steps)
        }
    
    def get_time_statistics(self) -> Dict[str, float]:
        if not self.episode_times:
            return {}
        
        return {
            'time_mean': np.mean(self.episode_times),
            'time_std': np.std(self.episode_times),
            'time_total': np.sum(self.episode_times),
            'time_per_step': np.sum(self.episode_times) / np.sum(self.episode_steps) if self.episode_steps else 0
        }
    
    def get_learning_curve(self, window_size: int = 100) -> List[float]:
        if not self.episode_rewards or len(self.episode_rewards) < window_size:
            return [sum(rewards) for rewards in self.episode_rewards]
        
        total_rewards = [sum(rewards) for rewards in self.episode_rewards]
        smoothed_rewards = []
        
        for i in range(len(total_rewards)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            smoothed_rewards.append(np.mean(total_rewards[start_idx:end_idx]))
        
        return smoothed_rewards
    
    def get_convergence_analysis(self, threshold_percentile: float = 90) -> Dict[str, Any]:
        if len(self.episode_rewards) < 20:
            return {'converged': False, 'convergence_episode': None}
        
        total_rewards = [sum(rewards) for rewards in self.episode_rewards]
        final_performance = np.mean(total_rewards[-20:])
        threshold = np.percentile(total_rewards, threshold_percentile)
        
        window_size = 10
        for i in range(window_size, len(total_rewards)):
            window_avg = np.mean(total_rewards[i-window_size:i])
            if window_avg >= threshold:
                return {
                    'converged': True,
                    'convergence_episode': i,
                    'final_performance': final_performance,
                    'threshold': threshold
                }
        
        return {
            'converged': False,
            'convergence_episode': None,
            'final_performance': final_performance,
            'threshold': threshold
        }
    
    def get_agent_cooperation_metrics(self) -> Dict[str, float]:
        if not self.episode_rewards:
            return {}
        
        cooperation_scores = []
        fairness_scores = []
        
        for rewards in self.episode_rewards:
            if len(rewards) > 1:
                reward_variance = np.var(rewards)
                reward_mean = np.mean(rewards)
                
                cooperation_score = reward_mean
                fairness_score = 1 / (1 + reward_variance) if reward_variance > 0 else 1.0
                
                cooperation_scores.append(cooperation_score)
                fairness_scores.append(fairness_score)
        
        if not cooperation_scores:
            return {}
        
        return {
            'cooperation_mean': np.mean(cooperation_scores),
            'cooperation_std': np.std(cooperation_scores),
            'fairness_mean': np.mean(fairness_scores),
            'fairness_std': np.std(fairness_scores)
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        return {
            'reward_stats': self.get_reward_statistics(),
            'step_stats': self.get_step_statistics(),
            'time_stats': self.get_time_statistics(),
            'convergence': self.get_convergence_analysis(),
            'cooperation': self.get_agent_cooperation_metrics(),
            'custom_metrics': {k: {
                'mean': np.mean(v),
                'std': np.std(v),
                'min': np.min(v),
                'max': np.max(v)
            } for k, v in self.custom_metrics.items() if v}
        }