import gymnasium as gym
import numpy as np

from gymnasium import spaces

class fishing_1s1a(gym.Env):
	""" a single-species fishing model """
	def __init__(self, config = {}):
		#
		self.r = config.get("r", 0.5)
		self.K = config.get("K", 1)
		self.sigma = config.get("sigma", 0.1)
		#
		self.cost = config.get("cost", 0.1)
		#
		self.pop_bound = config.get("pop_bound", 2 * self.K)
		self.init_pop = config.get("init_pop", np.float32([0.7]))
		self.init_sigma = config.get("init_sigma", 0.05)
		self.tmax = config.get("tmax", 100)
		self.thresh = config.get("thresh", 0.05)
		#
		self.action_space = spaces.Box(
			np.float32([-1]),
			np.float32([+1]),
			)
		self.observation_space = spaces.Box(
			np.float32([-1]),
			np.float32([+1]),
			)

		self.reset()

	def reset(self, *, seed=42, options=None):
		self.timestep = 0
		self.pop = self.init_pop * (1 + self.init_sigma * np.random.normal())
		self.state = self.pop_to_state(self.pop)
		info = {}
		return self.state, info


	def step(self, action):
		#
		self.pop = self.state_to_pop(self.state)
		self.pop = np.clip(self.pop, [0], [self.pop_bound])
		#
		action = np.clip(action, [-1], [1])
		effort = self.action_to_effort(action)
		harvest = effort * self.pop[0]
		cost = self.cost[0] * effort
		#
		self.pop -= harvest
		self.pop = np.clip(self.pop, [0], [self.pop_bound])
		self.pop += self.r * self.pop * (1 - self.pop / self.K) + self.sigma * np.random.normal() * self.pop
		self.pop = np.clip(self.pop, [0], [self.pop_bound])
		self.state = self.pop_to_state(self.pop)
		#
		obs = self.state
		reward = harvest - cost
		terminated = False
		truncated = False
		info = {}
		#
		if self.pop < self.thresh:
			reward -= self.tmax / (self.timestep + 1)
			terminated = True
		if self.timestep >= self.tmax:
			terminated = True
		#
		self.timestep += 1
		return obs, reward, terminated, truncated, info

	def pop_to_state(self, pop):
		return 2 * pop / self.pop_bound - 1

	def state_to_pop(self, state):
		return (state + 1) * self.pop_bound / 2

	def action_to_effort(self, action):
		return (action + 1) / 2

	def effort_to_action(self, effort):
		return effort * 2 - 1

