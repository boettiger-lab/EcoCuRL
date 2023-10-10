import pandas as pd

from escapement import escapement_policy
from envs.custom_example import fishing_env
from envs.custom_example import curriculum_fishing_env # only for the curriculum

esc = escapement_policy(n_sp=1, n_act=1, controlled_sp=[0], max_esc=1)
esc_results = {i: esc.rand_policy_search(env=fishing_env(*lvl)) for i, lvl in CURRICULUM.items()}

print("results:\n"
      "--------\n"
      "{"
    )
for i, result in esc_results:
    print(
        f"{i}: {result['avg_rew']}"
        )
print("}")