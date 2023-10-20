import gymnasium as gym
import numpy as np

from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

# TBD:
#
# TaskSettableEnv which samples env_config randomly with increasing
# entropy as the curriculum level increases

class benchmarked_curl(TaskSettableEnv):
	""" a curr. l. env whose rewards are normalized by curr.-lvl.-specific benchamrks. """
	def __init__(self, config: dict):
		#
		self.needed_cfg_elements = ['base_env_cls', 'curr_to_params', 'curr_benchmarks']
		self._config_check(config)
		#
		for name in needed_cfg_elements:
			setattr(self, name, config[name])
		#
		self.base_env_config = config.get('static_base_env_config', {})
		#
		self.n_lvls = len(curr_to_params)
		self.task_options = list(curr_to_params) # list(dict) = list of keys
		self.curr_lvl = self.task_options[0]
		self.switch_env = False
		#
		self.base_env = self._make_env()
		self.observation_space = self.base_env.observation_space
		self.action_space = self.base_env.action_space

	def reset(self, *, seed=42, options=None):
		if self.switch_env:
			self.switch_env = False
			self.base_env = self._make_env()
		return self.base_env.reset(seed, options)


	def step(self, action):
		obs, rew, term, trunc, info = self.base_env.step(action)
		rescaled_rew = rew / self.curr_benchmarks[self.curr_lvl]
		return (
			obs, rescaled_rew, term, trunc, info
		)

	def _config_check(self, config):
		""" checks that config has necessary elements. """
		for name in self.needed_cfg_elements:
			if not name in config:
				missing_elements.append(name)

		assert len(missing_elements) == 0, (
			f"the following elements are missing from the config dict passed to "
			f"the benchmarked_curl class constructor:\n"
			f"{missing_elements}.\n"
			f"config passed:\n"
			f"{config}"
		)

	def _make_env(self):
		env_config = {
			**self.curr_to_params[self.curr_lvl],
			**self.static_base_env_config
		}
		return self.base_env_cls(env_config)

	def reset(self, *, seed=None, options=None):
		if self.switch_env:
			self.switch_env = False
			self.env = self._make_env()
		return self.env.reset(seed=None, options=None)

	@override(TaskSettableEnv)
	def sample_tasks(self, n_tasks):
		"""Implement this to sample n random tasks."""
		return [np.random.choice(self.task_options) for _ in range(n_tasks)]

	@override(TaskSettableEnv)
	def get_task(self):
		"""Implement this to get the current task (curriculum level)."""
		return self.cur_level

	@override(TaskSettableEnv)
	def set_task(self, task):
		"""Implement this to set the task (curriculum level) for this env."""
		self.cur_level = task
		self.switch_env = True



