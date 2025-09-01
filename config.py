import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class TrainingConfig:
    num_episodes: int = 1000
    learning_rate: float = 0.001
    batch_size: int = 32
    hidden_size: int = 128
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update: int = 10
    memory_size: int = 10000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4

@dataclass
class EnvironmentConfig:
    env_name: str = "Cleaner"
    n_agents: int = 2
    map_size: int = 15
    max_episode_steps: int = 1000
    seed: Optional[int] = None

@dataclass
class EvaluationConfig:
    num_eval_episodes: int = 20
    eval_frequency: int = 100
    save_frequency: int = 500
    log_dir: str = "logs"
    save_dir: str = "models"
    plot_dir: str = "plots"

@dataclass
class ExperimentConfig:
    name: str = "default_experiment"
    trainer_type: str = "dqn"
    training: TrainingConfig = None
    environment: EnvironmentConfig = None
    evaluation: EvaluationConfig = None
    
    def __post_init__(self):
        if self.training is None:
            self.training = TrainingConfig()
        if self.environment is None:
            self.environment = EnvironmentConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()

class ConfigManager:
    
    @staticmethod
    def load_config(config_path: str) -> ExperimentConfig:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        training_config = TrainingConfig(**data.get('training', {}))
        env_config = EnvironmentConfig(**data.get('environment', {}))
        eval_config = EvaluationConfig(**data.get('evaluation', {}))
        
        return ExperimentConfig(
            name=data.get('name', 'default_experiment'),
            trainer_type=data.get('trainer_type', 'dqn'),
            training=training_config,
            environment=env_config,
            evaluation=eval_config
        )
    
    @staticmethod
    def save_config(config: ExperimentConfig, config_path: str):
        data = asdict(config)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2)
    
    @staticmethod
    def get_default_configs() -> Dict[str, ExperimentConfig]:
        return {
            'dqn_cleaner': ExperimentConfig(
                name='dqn_cleaner_experiment',
                trainer_type='dqn',
                training=TrainingConfig(num_episodes=2000, learning_rate=0.001),
                environment=EnvironmentConfig(env_name='Cleaner', n_agents=2, map_size=15),
                evaluation=EvaluationConfig(num_eval_episodes=50)
            ),
            'ppo_soccer': ExperimentConfig(
                name='ppo_soccer_experiment',
                trainer_type='ppo',
                training=TrainingConfig(num_episodes=5000, learning_rate=0.0003),
                environment=EnvironmentConfig(env_name='Soccer', n_agents=4),
                evaluation=EvaluationConfig(num_eval_episodes=30)
            ),
            'random_benchmark': ExperimentConfig(
                name='random_baseline',
                trainer_type='random',
                training=TrainingConfig(num_episodes=1000),
                environment=EnvironmentConfig(env_name='Cleaner', n_agents=2),
                evaluation=EvaluationConfig(num_eval_episodes=100)
            )
        }
    
    @staticmethod
    def create_config_templates(output_dir: str = "configs"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        configs = ConfigManager.get_default_configs()
        
        for name, config in configs.items():
            config_path = os.path.join(output_dir, f"{name}.yaml")
            ConfigManager.save_config(config, config_path)
        
        return list(configs.keys())