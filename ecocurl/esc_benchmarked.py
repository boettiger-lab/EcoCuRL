from collections.abc import Iterable
import logging
import numpy as np

from ecocurl.escapement import sample_esc_benchmark, escapement_policy

logging.basicConfig(level=logging.INFO)

def get_EscBmks(
	env_cls, 
	index_to_config, 
	escapement_policy_kwargs = {},
	verbose = False,
	log_fname = None,
):
	curr_benchmarks = {}
	#
	for lvl, config in index_to_config.items():
		logging.info(f"Processing curriculum level: {lvl}, config {config}")
		base_env = env_cls(config=config)
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
		# debug logging
		if isinstance(opt_esc, Iterable):
			logging.debug(
				f"opt idx = {i_opt}, "
				f"opt_esc = {[f'{el:.3f}' for el in opt_esc]}, "
				f"opt_bmk = {opt_bmk:.3f}."
			)
		else:
			logging.debug(
				f"opt idx = {i_opt}, "
				f"opt_esc = {opt_esc}, "
				f"opt_bmk = {opt_bmk:.3f}."
			)
		del policies
		del benchmarks
		#
		if log_fname:
			with open(log_fname, "w") as logfile:
				logfile.write(f"lvl {lvl}, r = {index_to_config[lvl]['r']}: opt. esc = {opt_esc}, benchmark = {bmk:.3f}\n")
		curr_benchmarks[lvl] = opt_bmk
	#
	return curr_benchmarks
	

