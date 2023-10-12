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
			quota * (1 + self.harvest_noise * np.random.normal()),
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
		observed_pop = pop * (1 + self.obs_noise * np.random.normal() )
		observation = self.pop_to_state(observed_pop)

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

	def __init__(self, config: EnvContext):
		#
		self.cur_level = config.get("start_level", 0)
		self.env = self._make_env()
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space
		self.switch_env = False
		#
		# these need to be updated manually using esc_script.py
		self.esc_bchmrk_rewards = {
			0: 12.296863133224907,
			1: 12.237871552434372,
			2: 12.20860903089324,
			3: 12.175643578586078,
			4: 12.143807683700253,
			5: 12.1297425526276,
			6: 12.107404626147074,
			7: 12.091653084692341,
			8: 12.074724722590767,
			9: 12.052502681886224,
			10: 12.026987076674892,
		}

	def _make_env(self):
		obs_noise = 0.2
		harvest_noise = 0.2
		self.CURRICULUM = {
			0: {"obs_noise": 0.00 * obs_noise, "harvest_noise": 0.00 * harvest_noise}, 
			1: {"obs_noise": 0.40 * obs_noise, "harvest_noise": 0.40 * harvest_noise},
			2: {"obs_noise": 0.50 * obs_noise, "harvest_noise": 0.50 * harvest_noise},
			3: {"obs_noise": 0.60 * obs_noise, "harvest_noise": 0.60 * harvest_noise},
			4: {"obs_noise": 0.70 * obs_noise, "harvest_noise": 0.70 * harvest_noise},
			5: {"obs_noise": 0.75 * obs_noise, "harvest_noise": 0.75 * harvest_noise},
			6: {"obs_noise": 0.80 * obs_noise, "harvest_noise": 0.80 * harvest_noise},
			7: {"obs_noise": 0.85 * obs_noise, "harvest_noise": 0.85 * harvest_noise},
			8: {"obs_noise": 0.90 * obs_noise, "harvest_noise": 0.90 * harvest_noise},
			9: {"obs_noise": 0.95 * obs_noise, "harvest_noise": 0.95 * harvest_noise},
			10: {"obs_noise": 1.0 * obs_noise, "harvest_noise": 1.00 * harvest_noise},
		}
		return fishing_env(**self.CURRICULUM[self.cur_level])

	def step(self, action):
		obs, rew, terminated, truncated, info = self.env.step(action)
		# Make rewards scale with the level exponentially:
		# Level 1: x1
		# Level 2: x10
		# Level 3: x100, etc...

		reward = (rew / self.esc_bchmrk_rewards[self.cur_level]) * 10**(self.cur_level) 

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






















