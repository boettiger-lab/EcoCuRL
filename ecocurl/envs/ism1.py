import gymnasium as gym
import numpy as np

from gymnasium import spaces

class ISM_linear(gym.Env):
	""" 
	1 species model for invsive species management. 
	linear damage as function of species variable
	"""
	def __init__(self, config = {}):
		#
		# growth
		self.r = config.get("r", 0.5)
		self.K = config.get("K", 1)
		self.sigma = config.get("sigma", 0.1)
		#
		# removal
		self.cost = config.get("cost", 0.1)
		self.removal_saturation = config.get("removal_saturation", 0.7) # c in eq below
		self.removal_efficiency = config.get("removal_efficiency", 6) # b in eq below
		###
		# removal_rate = c * (1 - exp( - action * b))
		###
		#
		# episode init data
		self.pop_bound = config.get("pop_bound", 2 * self.K)
		self.init_pop = config.get("init_pop", np.float32([0.7]))
		self.init_sigma = config.get("init_sigma", 0.05)
		self.tmax = config.get("tmax", 100)
		#
		# ecosystem damage: linear for now
		self.damage_coeff = config.get("damage_coeff", 0.3)
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
		removal_rate = (
			self.removal_saturation * (
				1 - np.exp(- self.removal_efficiency * action[0])
			)
		)
		cost = self.cost * removal_rate
		#
		self.pop -= removal_rate * self.pop
		self.pop = np.clip(self.pop, [0], [self.pop_bound])
		self.pop += self.r * self.pop * (1 - self.pop / self.K) + self.sigma * np.random.normal() * self.pop
		self.pop = np.clip(self.pop, [0], [self.pop_bound])
		self.state = self.pop_to_state(self.pop)
		#
		obs = self.state
		reward = - self.damage_coeff * self.pop[0] - cost
		terminated = False
		truncated = False
		info = {}
		#
		if self.timestep >= self.tmax:
			terminated = True
		#
		self.timestep += 1
		return obs, reward, terminated, truncated, info

	def pop_to_state(self, pop):
		return 2 * pop / self.pop_bound - 1

	def state_to_pop(self, state):
		return (state + 1) * self.pop_bound / 2


