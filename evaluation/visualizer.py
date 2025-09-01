import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import os

class ResultsVisualizer:
    
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        plt.style.use('default')
    
    def plot_learning_curves(self, results: Dict[str, List[float]], title: str = "Learning Curves"):
        plt.figure(figsize=(12, 8))
        
        for trainer_name, rewards in results.items():
            episodes = range(len(rewards))
            plt.plot(episodes, rewards, label=trainer_name, linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.save_dir, f"{title.lower().replace(' ', '_')}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_training_comparison(self, benchmark_results: Dict[str, Any]):
        trainers = list(benchmark_results.keys())
        metrics = ['final_performance', 'training_time', 'convergence']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            if metric == 'final_performance':
                means = [benchmark_results[t][metric]['mean'] for t in trainers]
                stds = [benchmark_results[t][metric]['std'] for t in trainers]
                ax.bar(trainers, means, yerr=stds, capsize=5)
                ax.set_ylabel('Final Performance')
                ax.set_title('Final Performance Comparison')
                
            elif metric == 'training_time':
                times = [benchmark_results[t][metric]['mean'] for t in trainers]
                ax.bar(trainers, times)
                ax.set_ylabel('Training Time (seconds)')
                ax.set_title('Training Time Comparison')
                
            elif metric == 'convergence':
                episodes = [benchmark_results[t][metric]['mean_episodes'] for t in trainers]
                ax.bar(trainers, episodes)
                ax.set_ylabel('Episodes to Convergence')
                ax.set_title('Convergence Speed Comparison')
            
            ax.tick_params(axis='x', rotation=45)
        
        axes[3].axis('off')
        
        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "training_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_reward_distribution(self, results: Dict[str, List[float]]):
        plt.figure(figsize=(12, 8))
        
        data = [rewards for rewards in results.values()]
        labels = list(results.keys())
        
        plt.boxplot(data, labels=labels)
        plt.ylabel('Total Reward')
        plt.title('Reward Distribution Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.save_dir, "reward_distribution.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_episode_lengths(self, results: Dict[str, List[int]]):
        plt.figure(figsize=(12, 8))
        
        for trainer_name, steps in results.items():
            episodes = range(len(steps))
            plt.plot(episodes, steps, label=trainer_name, alpha=0.7)
        
        plt.xlabel('Episode')
        plt.ylabel('Episode Length (Steps)')
        plt.title('Episode Length Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.save_dir, "episode_lengths.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_agent_performance(self, agent_rewards: List[List[float]], agent_names: List[str] = None):
        if agent_names is None:
            agent_names = [f"Agent {i}" for i in range(len(agent_rewards[0]))]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        episodes = range(len(agent_rewards))
        agent_data = list(zip(*agent_rewards))
        
        for i, (agent_data, name) in enumerate(zip(agent_data, agent_names)):
            ax1.plot(episodes, agent_data, label=name)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Individual Agent Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        final_rewards = [rewards[-100:] if len(rewards) >= 100 else rewards for rewards in agent_data]
        ax2.boxplot(final_rewards, labels=agent_names)
        ax2.set_ylabel('Final Reward Distribution')
        ax2.set_title('Final Performance Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.save_dir, "agent_performance.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_summary_report(self, benchmark_results: Dict[str, Any], 
                            output_file: str = "benchmark_report.html"):
        html_content = """
        <html>
        <head><title>Benchmark Report</title></head>
        <body>
        <h1>Multi-Agent RL Benchmark Report</h1>
        """
        
        for trainer_name, results in benchmark_results.items():
            html_content += f"""
            <h2>{trainer_name}</h2>
            <ul>
                <li>Final Performance: {results['final_performance']['mean']:.2f} ± {results['final_performance']['std']:.2f}</li>
                <li>Training Time: {results['training_time']['mean']:.2f} ± {results['training_time']['std']:.2f} seconds</li>
                <li>Convergence: {results['convergence']['mean_episodes']:.0f} episodes</li>
            </ul>
            """
        
        html_content += "</body></html>"
        
        filepath = os.path.join(self.save_dir, output_file)
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        return filepath