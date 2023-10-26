import gymnasium as gym
import logging
import numpy as np

from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.utils.annotations import override

logging.basicConfig(level=logging.DEBUG)

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
		self.reset()

	def reset(self, *, seed=42, options=None):
		return self.raw_env.reset(seed=seed, options=options)

	def step(self, action):
		obs, raw_reward, term, trunc, info = self.raw_env.step(action)
		reward = raw_reward / self.benchmark
		return obs, reward, term, trunc, info

class benchmarkedRandEnv(gym.Env):
	""" randomly sampled attribute """
	def __init__(self, raw_env: gym.Env, attr_name: str, attr_sample_set: list, attr_idx_to_bmk: list):
		self.raw_env = raw_env
		self.attr_name = attr_name
		self.attr_sample_set = attr_sample_set
		self.attr_idx_to_bmk = attr_idx_to_bmk
		self.observation_space = self.raw_env.observation_space
		self.action_space = self.raw_env.action_space

		self.reset()

	def reset(self, *, seed=42, options=None):
		self._task_idx = np.random.randint(len(self.attr_sample_set))
		new_attr_val = self.attr_sample_set[self._task_idx]
		print(f"\n\nbenchmarkedRandEnv.reset(): task idx: {self._task_idx}, {self.attr_name} value: {new_attr_val}\n\n")
		setattr(self.raw_env, self.attr_name, new_attr_val)
		return self.raw_env.reset(seed=seed, options=options)

	def step(self, action):
		obs, raw_rew, term, trunc, info = self.raw_env.step(action)
		rew = raw_rew / self.attr_idx_to_bmk[self._task_idx]
		return obs, rew, term, trunc, info


class discrBenchMultitaskerV2(TaskSettableEnv):
	#
	# TBD!! actually rewrite the code for V2!
	#
	def __init__(self, config: dict):
		"""
		config:
			base_env_cls = gym.Env class
			base_env_cfg = config dict for base_env_cls
			task_indices = [list of task indices / labels]
				-> must be hashable
			task_bmks    = dict of the form {task_index: benchmark for the task}
			task_configs = dict of the form {task_index: config dict for base benchmarked env}
				-> hardwired: all variation should be in a single parameter, which should be an attribute of the base_env
			randomized_attr = name of the attribute that will be randomized
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
		self.needed_cfg_elements = [
			'base_env_cls', 'base_env_cfg', 'task_indices', 'task_configs', 
			'task_bmks', 'randomized_attr', 'lvl_to_task_list',
		]
		self._config_check(config)
		#
		for name in self.needed_cfg_elements:
			setattr(self, name, config[name])
		#
		self.n_lvls = len(self.lvl_to_task_list)
		self.lvl = 0
		#
		self.base_benchmarked_env = self._make_env()
		self.observation_space = self.base_benchmarked_env.observation_space
		self.action_space = self.base_benchmarked_env.action_space
		self.switch_env = False

	def _make_env(self):
		raw_env = self.base_env_cls(config = self.base_env_cfg)
		attr_name = self.randomized_attr
		attr_sample_set = [ 
			self.task_configs[i][attr_name] for i in self.lvl_to_task_list[self.lvl] 
		]
		print(f"\n\n\nattr_sample_set = {attr_sample_set}\n\n\n")
		attr_idx_to_bmk = {idx: self.task_bmks[idx] for idx, task in enumerate(attr_sample_set)}
		#
		return benchmarkedRandEnv(
			raw_env = raw_env,
			attr_name = attr_name,
			attr_sample_set = attr_sample_set,
			attr_idx_to_bmk = attr_idx_to_bmk,
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
		return self.base_benchmarked_env.reset(seed=seed, options=options)
	#
	def step(self, action):
		obs, rew, term, trunc, info = self.base_benchmarked_env.step(action)
		return obs, rew * 10**(self.lvl), term, trunc, info
	#
	# changing, sampling, getting curriculum level
	#
	def _sample_lvl(self, n_samples):
		return [np.random.randint(self.n_lvls) for _ in range(n_samples)]
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
		self._set_lvl(task)


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
		self.n_lvls = len(self.lvl_to_task_list)
		self.lvl = 0
		#
		self.base_benchmarked_env = self._make_env()
		self.observation_space = self.base_benchmarked_env.observation_space
		self.action_space = self.base_benchmarked_env.action_space
		self.switch_env = False

	def _make_env(self):
		task = np.random.choice(self.lvl_to_task_list[self.lvl])
		env_cfg = self.task_configs[task]
		env_bmk = self.task_bmks[task]
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
		return self.base_benchmarked_env.reset(seed=seed, options=options)
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
		self._set_lvl(task)

