import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class MyTaskSettableEnv(MultiAgentEnv):
    def __init__(self, config):
        self.task = None
        self.agent1 = "agent_1"
        self.agent2 = "agent_2"
        self.observation_space = gym.spaces.Discrete(5)  # Replace with your actual observation space
        self.action_space = gym.spaces.Discrete(2)  # Replace with your actual action space
        self.max_steps = 100  # Set the maximum number of steps per episode

    def reset(self):
        self.task = np.random.randint(5)  # Randomly set the task for agent 2
        self.current_step = 0
        self.agent_2_performance = 0
        return {
            self.agent1: self._get_obs(self.agent1),
            self.agent2: self._get_obs(self.agent2)
        }

    def step(self, action_dict):
        assert self.agent1 in action_dict and self.agent2 in action_dict

        # Update the performance of agent 2 based on its action
        if action_dict[self.agent2] == self.task:
            self.agent_2_performance += 1

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Calculate the rewards for both agents
        reward_dict = {
            self.agent1: self.agent_2_performance,  # Reward for agent 1 based on agent 2's performance
            self.agent2: 0  # You can define a different reward structure for agent 2 if needed
        }

        obs_dict = {
            self.agent1: self._get_obs(self.agent1),
            self.agent2: self._get_obs(self.agent2)
        }

        return obs_dict, reward_dict, done, {}

    def _get_obs(self, agent):
        # Replace this with logic to generate observations for each agent
        return np.zeros(self.observation_space.n)

    def render(self, mode='human'):
        # Implement rendering if needed
        pass

    def close(self):
        # Implement any cleanup here
        pass
