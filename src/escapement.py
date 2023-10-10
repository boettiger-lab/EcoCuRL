import numpy as np
from typing import List

import ray

@ray.remote
class escapement_policy:

	def __init__(
		self,
		n_sp: int,
		n_act: int,
		controlled_sp: List = None,
		max_esc: float = 10, # maximum escapement (used to bound esc optimization)
		numeric_threshold: float = 0.001,
		):
		self.n_sp=n_sp
		self.n_act=n_act
		if controlled_sp is None:
			self.controlled_sp = list(range(n_act))
		else:
			self.controlled_sp=controlled_sp
		self.max_esc=max_esc
		self.numeric_threshold = numeric_threshold
		#
		# to be overwritten later:
		self.optimized_esc = None
		self.optimized_policy_fn = None

	def compute_effort(self, esc_level, variable):
		"""computes fishing effort for a single species."""
		if (variable <= esc_level) or (variable <= self.numeric_threshold):
			# second clause for the odd case where esc < nm_thresh
			return 0.
		else:
			return (variable - esc_level) / variable # effort units

	def policy_factory(self, esc_vec):
		"""
		generates policy labeled by esc_vec.

		returns policy that maps: pop [iterable, len > n_act]  --> np.ndarray [len = n_act]
		"""
		esc_dict = dict(zip(self.controlled_sp, esc_vec)) # {sp_index: escapement_level}
		return lambda pop: np.float32([
			self.compute_effort(esc_dict[i], pop[i]) for i in self.controlled_sp
			])

	def sample_policy(self):
		esc_vec = np.float32(
			[
			self.max_esc * np.random.rand() for _ in range(self.n_act)
			]
			)
		return self.policy_factory(esc_vec)

	def effort_to_action(self, effort: np.ndarray):
		""" [0,1] to [-1,1] space """
		return effort * 2 - 1

	def sample_policy_reward(self, esc_vec, env):
		policy = self.policy_factory(esc_vec)
		episode_reward = 0
		observation, _ = env.reset()
		for t in range(env.metadata.tmax):
			pop = env.env.state_to_pop(observation) # natural units
			action = self.effort_to_action(policy(pop))
			observation, reward, terminated, done, info = env.step(action)
			episode_reward += reward
			#
			if done or terminated:
				break
		return episode_reward

	def evaluate_policy(self, esc_vec, env, N=50):
		# return ray.get(
		# 	[self.sample_policy_reward.remote(esc_vec, env) for _ in range(N)]
		# 	)
		return [self.sample_policy_reward(esc_vec, env) for _ in range(N)]

	def rand_policy_search(self, env, num_samples=1_000, verbose=False):
		"""
		env is a base_env.ray_eco_env object
		"""
		#
		# setup
		current_best = np.zeros(self.n_act)
		best_histogram = self.evaluate_policy(current_best, env)
		best_avg = np.mean(best_histogram)
		best_std = np.std(best_histogram)
		#
		# loop
		for i in range(num_samples):
			esc_vec = np.random.rand(self.n_act)
			rew_histogram = self.evaluate_policy(esc_vec, env)
			avg_rew = np.mean(rew_histogram)
			std_rew = np.std(rew_histogram)
			if avg_rew > best_avg:
				current_best = esc_vec
				best_avg = avg_rew
				best_std = std_rew

			if verbose:
				if i == 0:
					print("Current best:\n" 
								"-------------")
				print(
							f"esc = {current_best}, "
							f"avg = {best_avg:.3f} +/- {best_std:.3f}, "
							f"sample nr. {i}",
							end="\r",
					)
		return {'esc': current_best, 'avg_rew': best_avg, 'std_rew': best_std}

	def optimize(self, env, method: str = "rand", verbose=False):
		
		if method != "rand":
			raise Warning(f"{type(self).__name__}: Only optimization method currently available is 'rand'.")
		
		best_dict = self.rand_policy_search(env, verbose=verbose)

		self.optimized_esc = best_dict['esc']
		self.optimized_policy_fn = self.policy_factory(self.optimized_esc)

		return self.optimized_policy_fn

