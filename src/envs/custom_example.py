import gymnasium as gym
import numpy as np

from gymnasium import spaces
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override

class fishing_env(gym.Env):
	""" basic env whose variations will be the curriculum """

	def __init__(self, obs_noise, harvest_noise):

		# user inputs
		self.obs_noise = obs_noise
		self.harvest_noise = harvest_noise

		# constants
		self.K = 0.8
		self.r = 0.3
		self.init_state = np.float32([0.5])
		self.init_noise = 0.05
		self.ep_len = 200

		self.observation_space = spaces.Box(
			np.float32([-1]),
			np.float32([1]),
			dtype = np.float32
			)

		self.action_space = spaces.Box(
			np.float32([-1]),
			np.float32([1]),
			dtype = np.float32
			)

		self.state = self.reset()

	def reset(self,  *, seed=42, options=None):
		self.state = self.init_state + self.init_noise * np.random.normal()
		self.timestep = 0
		return self.state, {}

	def step(self, action):
		
		# regularize
		action = np.clip([-1], [1], action)
		pop = self.state_to_pop(self.state)
		pop_start = pop.copy()

		# extract
		effort = self.action_to_effort(action)
		quota = pop * effort
		# print(f"quota: {quota}, noise: {self.harvest_noise}")
		harvest = np.clip(
			quota + self.harvest_noise * np.random.normal(),
			np.float32([0]), 
			pop,
			)

		# dynamics
		pop -= harvest
		if pop <= 0:
			pop = np.float32([0])
		pop += (
			self.r * pop * (1 - pop / self.K) 
			)
		self.state = self.pop_to_state(pop)
		observation = self.state * (1 + self.obs_noise * np.random.normal() )

		# reward, check for episode end
		reward = harvest[0]
		terminated = False
		if (self.timestep >= self.ep_len) or pop < 0.1:
			terminated = True
		
		info = {'p': pop_start[0], 'p prime': pop[0], 'harvest': harvest, 'timestep': self.timestep}

		self.timestep += 1

		return observation, reward, terminated, False, info

	def action_to_effort(self, action):
		""" [-1,1] to [0,1] effort """
		return (action + 1) / 2.

	def effort_to_action(self, effort):
		""" [0,1] to [-1,1] effort """
		return 2 * effort - 1

	def state_to_pop(self, state):
		return (state + 1) / 2.

	def pop_to_state(self, pop):
		return pop * 2 - 1



class curriculum_fishing_env(TaskSettableEnv):
	"""
	4 curriculum levels:
		 state noise | harvest noise
	0: no          | no    
	1: yes         | no
	2: no          | yes
	3: yes         | yes
	"""

	def __init__(self, config: EnvContext):
		#
		self.cur_level = config.get("start_level", 0)
		self.env = self._make_env()
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space
		self.switch_env = False

	def _make_env(self):
		obs_noise = 0.2
		harvest_noise = 0.1
		self.CURRICULUM = {
			0: {"obs_noise": 0.0,         "harvest_noise": 0.0}, 
			1: {"obs_noise": obs_noise, "harvest_noise": 0.0},
			2: {"obs_noise": 0.0,         "harvest_noise": harvest_noise},
			3: {"obs_noise": obs_noise, "harvest_noise": harvest_noise},
		}
		return fishing_env(**self.CURRICULUM[self.cur_level])

	def step(self, action):
		obs, rew, terminated, truncated, info = self.env.step(action)
		# Make rewards scale with the level exponentially:
		# Level 1: x1
		# Level 2: x10
		# Level 3: x100, etc...

		reward = (10 ** (self.cur_level)) * (rew / self.env.ep_len) * 100
		# print(f"lvl: {self.cur_level}, reward: {reward}, step: {self.env.timestep}")
		return (
			obs, 
			reward, 
			terminated, truncated, info
		)

	def reset(self, *, seed=None, options=None):
		if self.switch_env:
			self.switch_env = False
			self.env = self._make_env()
		return self.env.reset(seed=None, options=None)

	@override(TaskSettableEnv)
	def sample_tasks(self, n_tasks):
		"""Implement this to sample n random tasks."""
		return [np.random.randint(4) for _ in range(n_tasks)]

	@override(TaskSettableEnv)
	def get_task(self):
		"""Implement this to get the current task (curriculum level)."""
		return self.cur_level

	@override(TaskSettableEnv)
	def set_task(self, task):
		"""Implement this to set the task (curriculum level) for this env."""
		self.cur_level = task
		self.switch_env = True






















