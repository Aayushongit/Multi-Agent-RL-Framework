from base_env import BaseMultiAgentEnv
import numpy as np
from typing import List, Tuple, Any

class EnvAdapter(BaseMultiAgentEnv):
    
    def __init__(self, original_env):
        self.env = original_env
        super().__init__(self.env.N_agent)
    
    def reset(self) -> None:
        super().reset()
        self.env.reset()
    
    def step(self, action_list: List[int]) -> Tuple[List[float], bool]:
        super().step(action_list)
        
        reward = self.env.step(action_list)
        
        individual_rewards = [reward / self.n_agents for _ in range(self.n_agents)]
        
        done = self._check_done()
        
        return individual_rewards, done
    
    def get_obs(self) -> List[Any]:
        if hasattr(self.env, 'get_global_obs'):
            global_obs = self.env.get_global_obs()
            obs_list = []
            for i in range(self.n_agents):
                agent_obs = self._get_agent_observation(global_obs, i)
                obs_list.append(agent_obs.flatten())
            return obs_list
        else:
            return [np.zeros(self.get_obs_space_size()) for _ in range(self.n_agents)]
    
    def _get_agent_observation(self, global_obs: np.ndarray, agent_id: int) -> np.ndarray:
        if hasattr(self.env, 'agt_pos_list') and agent_id < len(self.env.agt_pos_list):
            agent_pos = self.env.agt_pos_list[agent_id]
            view_size = 5
            
            map_height, map_width = global_obs.shape[:2]
            
            start_row = max(0, agent_pos[0] - view_size // 2)
            end_row = min(map_height, agent_pos[0] + view_size // 2 + 1)
            start_col = max(0, agent_pos[1] - view_size // 2)
            end_col = min(map_width, agent_pos[1] + view_size // 2 + 1)
            
            local_obs = np.zeros((view_size, view_size, global_obs.shape[2]))
            
            local_obs[
                (start_row - (agent_pos[0] - view_size // 2)):(end_row - (agent_pos[0] - view_size // 2)),
                (start_col - (agent_pos[1] - view_size // 2)):(end_col - (agent_pos[1] - view_size // 2))
            ] = global_obs[start_row:end_row, start_col:end_col]
            
            return local_obs
        else:
            return global_obs
    
    def _check_done(self) -> bool:
        if hasattr(self.env, 'occupancy'):
            dirty_spots = np.sum(self.env.occupancy == 2)
            if dirty_spots == 0:
                return True
        
        return self.is_episode_done()
    
    def get_action_space_size(self) -> int:
        return 4
    
    def get_obs_space_size(self) -> int:
        if hasattr(self.env, 'map_size'):
            return 5 * 5 * 3
        return 100
    
    def render(self, mode: str = 'human'):
        if hasattr(self.env, 'render'):
            return self.env.render()
        return None