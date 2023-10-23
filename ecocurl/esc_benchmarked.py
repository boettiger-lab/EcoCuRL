import logging
import numpy as np

from ecocurl.escapement import sample_esc_benchmark, escapement_policy
from ecocurl.benchmarked_curl_env import benchmarked_curl

# logging.basicConfig(level=logging.WARNING)

def get_EscBmks(
	env_cls, 
	index_to_config, 
	escapement_policy_kwargs = {},
	verbose = False,
):
	curr_benchmarks = {}
	#
	for lvl, config in index_to_config.items():
		base_env = env_cls(**config)
		#
		if len(base_env.action_space.shape) > 1:
			logging.warning(f"get_EscBmked arg base_env has an action_space with non-flat shape: {base_env.action_space.shape}")
		n_act = base_env.action_space.shape[0]
		#
		if hasattr(base_env, "n_sp"):
			n_sp = base_env.n_sp
		else:
			if len(base_env.observation_space.shape) > 1:
				logging.warning(f"get_EscBmked arg base_env has an observation_space with non-flat shape: {base_env.observation_space.shape}")
			n_sp = base_env.observation_space.shape[0]
		#
		esc_obj = escapement_policy(
			n_sp=n_sp,
			n_act=n_act,
			verbose = verbose,
			**escapement_policy_kwargs,
		)
		policies, benchmarks = sample_esc_benchmark(base_env, esc_obj, samples=100)
		i_opt = np.argmax(benchmarks)
		opt_esc = policies[i_opt]
		opt_bmk = benchmarks[i_opt]
		del policies
		del benchmarks
		#
		curr_benchmarks[lvl] = opt_bmk
	#
	return curr_benchmarks
	

