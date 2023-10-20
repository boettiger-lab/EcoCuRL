import gymnasium as gym
import numpy as np

from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

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
		self.curr_lvl = list(curr_to_params)[0] # list(dict) = list of keys
		#
		self.base_env = self._make_env()
		self.observation_space = self.base_env.observation_space
		self.action_space = self.base_env.action_space

	def reset(self, *, seed=42, options=None):
		...

	def step(self, action):
		...

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



