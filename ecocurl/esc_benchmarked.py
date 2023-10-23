import logging
import numpy as np

from ecocurl.escapement import sample_esc_benchmark, escapement_policy
from ecocurl.benchmarked_curl_env import benchmarked_curl

# logging.basicConfig(level=logging.WARNING)

def get_EscBmked(
	base_env_cls, 
	curr_to_params, 
	static_base_env_config = {},
	escapement_policy_kwargs,
):
	curr_benchmarks = {}
	#
	for lvl, params in curr_to_params.items():
		base_env = base_env_cls(**params, **static_base_env_config)
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
			verbose=True,
			**escapement_policy_kwargs,
		)
		policies, benchmarks = sample_esc_benchmark(base_env, esc_obj)
		i_opt = np.argmax(benchmarks)
		opt_esc = policies[i_opt]
		opt_bmk = benchmarks[i_opt]
		del policies
		del benchmarks
		#
		curr_benchmarks[lvl] = opt_bmk
	#
	return curr_benchmarks
	

