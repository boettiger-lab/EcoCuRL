import pandas as pd
import ray
import numpy as np

from escapement import escapement_policy
from envs.custom_example import fishing_env
from envs.custom_example import curriculum_fishing_env # only for the curriculum

CURRICULUM = curriculum_fishing_env(config={"start_level":0}).CURRICULUM
esc_obj = escapement_policy(n_sp=1, n_act=1, controlled_sp=[0], max_esc=1)

@ray.remote
def benchmark(esc_vec, esc_obj, env):
    results = [esc_obj.sample_policy_reward(esc_vec, env) for _ in range(100)]
    # print(f"escapement = {esc_vec} done!")
    return np.mean(results)
        
def sample_esc_benchmark(lvl, esc_obj, samples=1000):
    env = fishing_env(**CURRICULUM[lvl])
    policies = [esc_obj.sample_policy() for _ in range(1000)]
    return policies, np.array(ray.get([benchmark.remote(esc_vec, esc_obj, env) for esc_vec in policies]))

# def get_stats(policies, benchmarks):

#     return (
#         {esc: np.mean(results) for esc, results in benchmarks}, 
#         {esc: np.std(results) for esc, results in benchmarks},
#         )

def find_best_policy(policies, esc_results):
    optimal_idx = np.argmax(esc_results)
    return policies[optimal_idx], esc_results[optimal_idx]

lvl_escapement_benchmarks = {}
for lvl in range(4):
    policies, esc_results = sample_esc_benchmark(lvl, esc_obj, samples=1000)
    esc, rew = find_best_policy(policies, esc_results)
    lvl_escapement_benchmarks[lvl] = {'esc': esc, 'rew': rew}
    print(f"level {lvl}: ",{'esc': esc, 'rew': rew})


print("escapement benchmarks:\n"
      "----------------------")
for lvl, val in lvl_escapement_benchmarks.items():
    print(
        f"{lvl}: {lvl_escapement_benchmarks[lvl]['rew']}"
        )

# esc_results = {i: 
#     ray.get(sample_esc_benchmark(lvl=i, esc=esc))
#     for i in range(4)
# }

# print("results:\n"
#       "--------\n"
#       "{"
#     )
# for i, result in esc_results:
#     print(
#         f"{i}: {result}"
#         )
# print("}")