

# https://github.com/ray-project/ray/blob/master/rllib/examples/curriculum_learning.py
"""
Example of a curriculum learning setup using the `TaskSettableEnv` API
and the env_task_fn config.

This example shows:
  - Writing your own curriculum-capable environment using gym.Env.
  - Defining a env_task_fn that determines, whether and which new task
    the env(s) should be set to (using the TaskSettableEnv API).
  - Using Tune and RLlib to curriculum-learn this env.

You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import argparse
import numpy as np
import os

import ray
from ray import air, tune
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
# from ray.rllib.examples.env.curriculum_capable_env import CurriculumCapableEnv
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls


from ecocurl.envs.noisy_fishing import curlNoisyFishing

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=5_000, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=20_000_000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=2_000.0,
    help="Reward at which we stop training.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

def linear_curriculum_fn(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Args:
        train_results: The train results returned by Algorithm.train().
        task_settable_env: A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx: The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.

    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    graduation_rate = 0.95
    new_task=0
    if train_results["episode_reward_mean"] > graduation_rate:
        new_task=1
    if train_results["episode_reward_mean"] > 10 * graduation_rate:
        new_task=2
    if train_results["episode_reward_mean"] > 100 * graduation_rate:
        new_task=3
    if train_results["episode_reward_mean"] > 1000 * graduation_rate:
        new_task=4
    if train_results["episode_reward_mean"] > 10_000 * graduation_rate:
        new_task=5
    if train_results["episode_reward_mean"] > 100_000 * graduation_rate:
        new_task=6
    if train_results["episode_reward_mean"] > 1_000_000 * graduation_rate:
        new_task=7
    if train_results["episode_reward_mean"] > 10_000_000 * graduation_rate:
        new_task=8
    if train_results["episode_reward_mean"] > 100_000_000 * graduation_rate:
        new_task=9
    if train_results["episode_reward_mean"] > 1_000_000_000 * graduation_rate:
        new_task=10
    # new_task = int(np.log10(train_results["episode_reward_mean"]))
    # new_task = max(min(new_task, 3), 0)

    print(
        f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
        f"\nR={train_results['episode_reward_mean']}"
        f"\nSetting env to task={new_task}"
    )
    return new_task


if __name__ == "__main__":
  args = parser.parse_args()
  ray.init()
  print("parsed args, ray init")

  # Can also register the env creator function explicitly with:
  # register_env(
  #     "curriculum_env", lambda config: CurriculumCapableEnv(config))

  config = (
  get_trainable_cls(args.run)
  .get_default_config()
  # or "curriculum_env" if registered above
  .environment(
    curlNoisyFishing,
    env_config={"start_level": 0},
    env_task_fn=linear_curriculum_fn,
  )
  .framework(args.framework)
  .rollouts(num_rollout_workers=25, num_envs_per_worker=5)
  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
  .resources(num_gpus=2
  # int(os.environ.get("RLLIB_NUM_GPUS", "0"))
  )
  )
  print("config")

  stop = {
  "training_iteration": args.stop_iters,
  "timesteps_total": args.stop_timesteps,
  "episode_reward_mean": 10_000_000_000,
  }

  tuner = tune.Tuner(
    args.run,
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=2),
  )
  print("tuner defined")
  results = tuner.fit()
  print("tuner fit")

  if args.as_test:
    check_learning_achieved(results, args.stop_reward)
  ray.shutdown()























if __name__ == "__main__":
	...