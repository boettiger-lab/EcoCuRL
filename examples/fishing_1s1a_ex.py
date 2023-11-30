from ecocurl.envs import fishing_1s1a, discrBenchMultitasker, discrBenchMultitaskerV2
from ecocurl import get_EscBmks

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.env.apis.task_settable_env import TaskType, TaskSettableEnv
from ray.rllib.env.env_context import EnvContext

"""
this script will train a multitasker agent which learns to fish over a 
variety of different parameter values for a single-species fishery model 
of the form:

X -> X + r * X * (1 - X / K) + noise
"""

task_indices = [0, 1, 2, 3, 4]
randomized_attr = 'K'
index_to_config = {
	0: {randomized_attr: 1},
	1: {randomized_attr: 1.2},
	2: {randomized_attr: 0.8},
	3: {randomized_attr: 1.4},
	4: {randomized_attr: 0.6},
}
lvl_to_task_list = {
	0: [0],
	1: [0,1],
	2: [0,1,2],
	3: [0,1,2,3],
	4: [0,1,2,3,4],
}
n_lvls = len(lvl_to_task_list)

restrict = False
if restrict:
	restriction = 2
	task_indices = task_indices[:restriction]
	index_to_config = {
		idx: cfg for idx, cfg in index_to_config.items() if idx < restriction
	}
	lvl_to_task_list = {
		idx: tsk_lst for idx, tsk_lst in lvl_to_task_list.items() if idx < restriction
	}
	n_lvls = len(lvl_to_task_list)


base_env_cls = fishing_1s1a


benchmarks = get_EscBmks(
	env_cls = base_env_cls,
	index_to_config = index_to_config,
	randomized_attr=randomized_attr,
	verbose = False,
	log_fname = "benchmarks_1s1a_log.txt",
	)

print("benchmarks done:")
for lvl, bmk in benchmarks.items():
	print(f"{lvl}: {bmk:.3f}")


curl_env =discrBenchMultitaskerV2(
	config = {
		'base_env_cls': base_env_cls,
		'base_env_cfg': {},
		'task_indices': task_indices,
		'task_configs': index_to_config,
		'task_bmks': benchmarks,
		'randomized_attr': randomized_attr,
		'lvl_to_task_list': lvl_to_task_list,
	}
)

# curl_env = discrBenchMultitasker(
# 	config = {
# 		'base_env_cls': base_env_cls,
# 		'task_indices': task_indices,
# 		'task_configs': index_to_config,
# 		'task_bmks': benchmarks,
# 		'lvl_to_task_list': lvl_to_task_list,
# 	}
# )

def linear_curriculum_fn(
	train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
	
	new_lvl = 0
	graduation_rate = 0.97
	for lvl in range(n_lvls-1):
		# up to n_lvls-2 since, once you graduate to n_lvls-1 (the maximum lvl)
		# you cannot graduate any further.
		if train_results["episode_reward_mean"] > graduation_rate * 10**(lvl):
			# print(f"graduated to lvl {lvl+1}")
			new_lvl = lvl+1
		else:
			print(f"graduated to lvl {new_lvl}")
	#
	print(
		f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
		f"\nR={train_results['episode_reward_mean']}"
		f"\nSetting env to curriculum lvl={new_lvl}"
	)
	return new_lvl

if __name__ == "__main__":

	config = (
		PPOConfig()
		.environment(
			discrBenchMultitaskerV2,
			env_config = {
				'base_env_cls': base_env_cls,
				'base_env_cfg': {},
				'task_indices': task_indices,
				'task_configs': index_to_config,
				'task_bmks': benchmarks,
				'randomized_attr': randomized_attr,
				'lvl_to_task_list': lvl_to_task_list,
			},
			env_task_fn=linear_curriculum_fn,
		)
		.rollouts(num_rollout_workers=25, num_envs_per_worker=5)
		.resources(num_gpus=2)
	)

	stop = {
		"training_iteration": 250,
		"timesteps_total": 1_000_000,
		"episode_reward_mean": 0.98 * 10 ** (n_lvls-1),
	}

	tuner = tune.Tuner(
		"PPO",
		param_space=config.to_dict(),
		run_config=air.RunConfig(stop=stop, verbose=2),
	)
	print("tuner defined")
	results = tuner.fit()
	print("tuner fit")

	# check_learning_achieved(results, args.stop_reward)
	ray.shutdown()

	

