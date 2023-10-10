import pandas as pd
import ray

from escapement import escapement_policy
from envs.custom_example import fishing_env
from envs.custom_example import curriculum_fishing_env # only for the curriculum

CURRICULUM = curriculum_fishing_env(config={"start_level":0}).CURRICULUM
esc = escapement_policy.remote(n_sp=1, n_act=1, controlled_sp=[0], max_esc=1)
esc_results = {lvl: ray.get(esc.rand_policy_search.remote(env=fishing_env(**details), verbose=True)) for lvl, details in CURRICULUM.items()}

print("results:\n"
      "--------\n"
      "{"
    )
for i, result in esc_results:
    print(
        f"{i}: {result['avg_rew']}"
        )
print("}")