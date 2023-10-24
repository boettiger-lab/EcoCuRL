from ecocurl.envs import fishing_1s1a, discrBenchMultitasker
from ecocurl import get_EscBmks

import ray
from ray import air, tune

from ray.rllib.env.apis.task_settable_env import TaskType, TaskSettableEnv
from ray.rllib.env.env_context import EnvContext

"""
this script will train a multitasker agent which learns to fish over a 
variety of different parameter values for a single-species fishery model 
of the form:

X -> X + r * X * (1 - X / K) + noise
"""

task_indices = [0, 1, 2, 3, 4]
index_to_config = {
	0: {'r': 0.5},
	1: {'r': 0.45},
	2: {'r': 0.55},
	3: {'r': 0.4},
	4: {'r': 0.6},
}

base_env_cls = fishing_1s1a


benchmarks = get_EscBmks(
	env_cls = base_env_cls,
	index_to_config = index_to_config,
	verbose = False,
	)

print("benchmarks done:")
for lvl, bmk in benchmarks.items():
	print(f"{lvl}: {bmk:.3f}")

lvl_to_task_list = {
	0: [0],
	1: [0,1],
	2: [0,1,2],
	3: [0,1,2,3],
	4: [0,1,2,3,4],
}
n_lvls = len(lvl_to_task_list)

curl_env = discrBenchMultitasker(
	config = {
		'base_env_cls': base_env_cls,
		'task_indices': task_indices,
		'task_configs': index_to_config,
		'task_bmks': benchmarks,
		'lvl_to_task_list': lvl_to_task_list,
	}
)

def linear_curriculum_fn(
	train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
	
	new_lvl = 0
	graduation_rate = 0.95
	for lvl in range(n_lvls):
		if train_results["episode_reward_mean"] > graduation_rate * 10**(lvl):
			new_lvl = lvl
	#
	print(
		f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
		f"\nR={train_results['episode_reward_mean']}"
		f"\nSetting env to curriculum lvl={new_lvl}"
	)
	return new_lvl

if __name__ == "__main__":
	ray.init()

	config = (
		PPOConfig()
		.environment(
			discrBenchMultitasker,
			env_config = {
				'base_env_cls': base_env_cls,
				'task_indices': task_indices,
				'task_configs': index_to_config,
				'task_bmks': benchmarks,
				'lvl_to_task_list': lvl_to_task_list,
			},
			env_task_fn=linear_curriculum_fn,
		)
		.rollouts(num_rollout_workers=25, num_envs_per_worker=5)
		.resources(num_gpus=2)
	)

	stop = {
		"training_iteration": 1000,
		"timesteps_total": 10_000_000,
		"episode_reward_mean": 10 ** n_lvls,
	}

	tuner = tune.Tuner(
		"PPO",
		param_space=config.to_dict(),
		run_config=air.RunConfig(stop=stop, verbose=2),
	)
	print("tuner defined")
	results = tuner.fit()
	print("tuner fit")

	if args.as_test:
		check_learning_achieved(results, args.stop_reward)
	ray.shutdown()

