import pandas as pd
import ray
import numpy as np

from escapement import escapement_policy
from envs.custom_example import fishing_env
from envs.custom_example import curriculum_fishing_env # only for the curriculum

CURRICULUM = curriculum_fishing_env(config={"start_level":0}).CURRICULUM
esc = escapement_policy.remote(n_sp=1, n_act=1, controlled_sp=[0], max_esc=1)

@ray.remote
def sample_esc_benchmark(lvl, esc, samples=1000):
    env = fishing_env(**CURRICULUM[lvl])
    policies = ray.get([esc.sample_policy.remote() for _ in range(1000)])
    return [
        (
            str(esc_vec),
            ray.get([
                esc.sample_policy_reward.remote(esc_vec, env) for _ in range(50)
            ])
        )
        for esc_vec in policies
    ]

def get_stats(benchmarks):
    return (
        {esc: np.mean(results) for esc, results in benchmarks}, 
        {esc: np.std(results) for esc, results in benchmarks},
        )


esc_results = {i: 
    get_stats(
        ray.get(sample_esc_benchmark.remote(
            lvl=i, esc=esc)
        ))[0]
    for i in range(4)
}

print("results:\n"
      "--------\n"
      "{"
    )
for i, result in esc_results:
    print(
        f"{i}: {result}"
        )
print("}")