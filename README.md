# Multi-Agent RL Framework

> **Forked from [Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment)**

This repository extends the original multi-agent environments with a comprehensive training and evaluation framework including DQN, PPO trainers, benchmarking tools, and visualization capabilities.

## 🚀 New Framework Features

- **Advanced Trainers**: DQN, PPO, and Random baseline implementations
- **Evaluation Tools**: Comprehensive benchmarking and metrics collection
- **Visualization**: Training curves, performance comparisons, and result analysis
- **CLI Tools**: Easy-to-use command-line training and benchmarking scripts
- **Configuration Management**: YAML/JSON config system for experiments
- **Package Structure**: Proper Python package with dependencies

## 📦 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Basic training example
python examples/basic_training.py

# Benchmark different algorithms
python scripts/benchmark.py --env Cleaner --trainers dqn ppo random

# Train specific model
python scripts/train.py --env Soccer --trainer ppo --episodes 2000
```

---

# Original Multi-Agent Learning Environments

The original environments for Multi Agent Reinforcement Learning. Some are single agent version that can be used for algorithm testing. Documents for each environment can be found in the corresponding pdf files in each directory. These are toy problems, though some of them are still challenging to solve. Available environments include:


## Multi Agent Soccer Game
![image](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/README/Soccer.gif)


## Multi Agent Rescue
![image](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/README/Rescue.gif)

## Multi Agent Cleaner
![image](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/README/Cleaner.gif)

## Multi Agent Move Box
![image](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/README/MoveBox.gif)


## Multi Agent Catching Pig
![image](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/README/CatchPigs.gif)


## Multi Drones Monitoring
![image](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/README/Drones.gif)


## Multi Agent Maze Running
![image](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/README/FindGoal.gif)


## Multi Agent Find Treasure
![image](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/README/FindTreasure.gif)


## Firefighters
![image](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/README/FireFighter.png)


## Go Together
![image](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/README/GoTogether.gif)


## Warehouse
![image](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/README/Warehouse.gif)


## Opposite
![image](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/README/Opposite.png)


## Dependency
OpenCV, swig


## Multi-Agent Environment Standard

**Assumption:**

Each agent works synchronously.


**Member Functions**

reset()

reward_list, done = step(action_list)

obs_list = get_obs()



reward_list records the single step reward for each agent, it should be a list like [reward1, 	reward2,......]. The length should be the same as the number of agents. Each element in the 	list should be a integer.

done True/False, mark when an episode finishes.

<font color=Blue>action_list</font> records the single step action instruction for each agent, it should be a list like [action1, 	action2,...]. The length should be the same as the number of agents. Each element in the 	list should be a non-negative integer.

<font color=Blue>obs_list</font> records the single step observation for each agent, it should be a list like [obs1, obs2,...]. The length should be the same as the number of agents. Each element in the 	list can be any form of data, but should be in same dimension, usually a list of variables or 	an image.


**Typical Monte Carlo Procedures**

reset environment by calling reset()
get initial observation get_obs()
for i in range(max_MC_iter):
  get action_list from controller
  apply action by step()
  record returned reward list
  record new observation by get_obs()
  
**Citation**

 Cite the environment of the following paper as:
 ```
@inproceedings{jiang2021multi,
  title={Multi-agent reinforcement learning with directed exploration and selective memory reuse},
  author={Jiang, Shuo and Amato, Christopher},
  booktitle={Proceedings of the 36th Annual ACM Symposium on Applied Computing},
  pages={777--784},
  year={2021}
}
```

