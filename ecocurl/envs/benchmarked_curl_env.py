import gymnasium as gym
import numpy as np

from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.utils.annotations import override

# TBD:
#
# TaskSettableEnv which samples env_config randomly with increasing
# entropy as the curriculum level increases

class benchmarkedEnv(gym.Env):
	""" wraps around an env, rewards are rescaled by a benchmark """
	def __init__(self, raw_env: gym.Env, benchmark: float):
		self.raw_env = raw_env
		self.observation_space = self.raw_env.observation_space
		self.action_space = self.raw_env.action_space
		self.benchmark = benchmark
		#
		assert benchmark > 1e-4, (
			f"benchmarkedEnv err: benchmark must be positive and large enough! "
			"Currently benchmark = {benchmark}."
		)

	def reset(self, *, seed=42, options=None):
		return self.raw_env.reset(self, seed=seed, options=options)

	def step(self, action):
		obs, raw_reward, term, trunc, info = self.raw_env.step(action)
		reward = raw_reward / benchmark
		return obs, reward, term, trunc, info


class discrBenchMultitasker(TaskSettableEnv):
	""" discrete benchmarked multitasker:  
	
	curriculum RL env designed to train an agent to be able to perform
	well accross the entire range of tasks available to it.

	possible tasks are a discrete finite set.
	"""
	def __init__(self, config: dict):
		"""
		config:
			base_env_cls = gym.Env class
			task_indices = [list of task indices / labels]
				-> must be hashable
			task_configs = dict of the form {task_index: config dict for base benchmarked env}
			task_bmks    = dict of the form {task_index: benchmark for the task}
			lvl_to_task_list = dict of the form {
				curriculum lvl: list from which to sample tasks at this lvl
			}
					# eg. if task_indices = [0, 1, 2]
					#
					# 0: [0], 1: [0, 1], 2: [0,1,2],
					#
					# then at curriculum lvl 0, only task 0 is sampled,
					# at lvl 1, task 0 and task 1 are sampled w 50% prob
					# at lvl 2, task 0, 1, 2, are sampled w 33% prob
		"""
		#
		self.needed_cfg_elements = ['base_env_cls', 'task_indices', 'task_configs', 'task_bmks', 'lvl_to_task_list']
		self._config_check(config)
		#
		for name in self.needed_cfg_elements:
			setattr(self, name, config[name])
		#
		self.n_lvls = len(lvl_to_range)
		self.lvl = 0
		#
		self.base_benchmarked_env = self._make_env()
		self.observation_space = self.base_benchmarked_env.observation_space
		self.action_space = self.base_benchmarked_env.action_space
		self.switch_env = False

	def _make_env(self):
		task = np.random.choice(self.lvl_to_task_list[self.lvl])
		env_cfg = task_configs[task]
		env_bmk = task_benchmarks[task]
		#
		return benchmarkedEnv(
			raw_env = self.base_env_cls(config = env_cfg),
			benchmark = env_bmk,
		)

	def _config_check(self, config):
		""" checks that config has necessary elements. """
		missing_elements = []
		for name in self.needed_cfg_elements:
			if not name in config:
				missing_elements.append(name)
		#
		assert len(missing_elements) == 0, (
			f"the following elements are missing from the config dict passed to "
			f"the benchmarked_curl class constructor:\n"
			f"{missing_elements}.\n"
			f"config passed:\n"
			f"{config}"
		)
	#
	def reset(self, *, seed=42, options=None):
		if self.switch_env:
			self.switch_env = False
			self.base_benchmarked_env = self._make_env()
		return self.base_benchmarked_env.reset(seed, options)
	#
	def step(self, action):
		obs, rew, term, trunc, info = self.base_benchmarked_env.step(action)
		return obs, rew * 10**(self.lvl), term, trunc, info
	#
	# changing, sampling, getting curriculum level
	#
	def _sample_lvl(self, n_samples):
		return [np.random.choice(self.n_lvls) for _ in range(n_samples)]
	#
	def _get_lvl(self):
		"""Implement this to get the current task (curriculum level)."""
		return self.lvl
	#
	def _set_lvl(self, lvl):
		"""Implement this to set the task (curriculum level) for this env."""
		self.lvl = lvl
		self.switch_env = True
	#
	# compatibility with TaskSettableEnv API (where "tasks" mean curriculum level.)
	#
	@override(TaskSettableEnv)
	def sample_tasks(self, n_tasks):
		"""Implement this to sample n random tasks."""
		return _sample_lvl(n_samples=n_tasks)

	@override(TaskSettableEnv)
	def get_task(self):
		"""Implement this to get the current task (curriculum level)."""
		return self._get_lvl()

	@override(TaskSettableEnv)
	def set_task(self, task):
		"""Implement this to set the task (curriculum level) for this env."""
		self._set_lvl()

# class benchmarked_curl(TaskSettableEnv):
# 	""" a curr. l. env whose rewards are normalized by curr.-lvl.-specific benchamrks. """
# 	def __init__(self, config: dict):
# 		#
# 		self.needed_cfg_elements = ['base_env_cls', 'curr_to_params', 'curr_benchmarks']
# 		self._config_check(config)
# 		#
# 		for name in needed_cfg_elements:
# 			setattr(self, name, config[name])
# 		#
# 		self.static_base_env_config = config.get('static_base_env_config', {})
# 		#
# 		self.n_lvls = len(curr_to_params)
# 		self.task_options = list(curr_to_params) # list(dict) = list of keys
# 		self.curr_lvl = self.task_options[0]
# 		self.switch_env = False
# 		#
# 		self.base_env = self._make_env()
# 		self.observation_space = self.base_env.observation_space
# 		self.action_space = self.base_env.action_space

# 	def reset(self, *, seed=42, options=None):
# 		if self.switch_env:
# 			self.switch_env = False
# 			self.base_env = self._make_env()
# 		return self.base_env.reset(seed, options)


# 	def step(self, action):
# 		obs, rew, term, trunc, info = self.base_env.step(action)
# 		rescaled_rew = rew / self.curr_benchmarks[self.curr_lvl]
# 		return (
# 			obs, rescaled_rew, term, trunc, info
# 		)

# 	def _config_check(self, config):
# 		""" checks that config has necessary elements. """
# 		for name in self.needed_cfg_elements:
# 			if not name in config:
# 				missing_elements.append(name)

# 		assert len(missing_elements) == 0, (
# 			f"the following elements are missing from the config dict passed to "
# 			f"the benchmarked_curl class constructor:\n"
# 			f"{missing_elements}.\n"
# 			f"config passed:\n"
# 			f"{config}"
# 		)

# 	def _make_env(self):
# 		env_config = {
# 			**self.curr_to_params[self.curr_lvl],
# 			**self.static_base_env_config
# 		}
# 		return self.base_env_cls(env_config)

# 	def reset(self, *, seed=None, options=None):
# 		if self.switch_env:
# 			self.switch_env = False
# 			self.env = self._make_env()
# 		return self.env.reset(seed=None, options=None)

# 	@override(TaskSettableEnv)
# 	def sample_tasks(self, n_tasks):
# 		"""Implement this to sample n random tasks."""
# 		return [np.random.choice(self.task_options) for _ in range(n_tasks)]

# 	@override(TaskSettableEnv)
# 	def get_task(self):
# 		"""Implement this to get the current task (curriculum level)."""
# 		return self.cur_level

# 	@override(TaskSettableEnv)
# 	def set_task(self, task):
# 		"""Implement this to set the task (curriculum level) for this env."""
# 		self.cur_level = task
# 		self.switch_env = True



