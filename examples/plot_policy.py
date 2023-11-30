#####################
## evaluate / plot ##
#####################

if __name__ == "__main__":
	import numpy as np
	import os
	import pandas as pd

	# manual for now
	experiment="fishing_1s1a"
	checkpoint_address = os.path.join(
		"..",
		"..",
		"ray_results",
		"PPO_2023-11-30_20-28-13",
		"PPO_discrBenchMultitaskerV2_fc438_00000_0_2023-11-30_20-28-13",
		"checkpoint_000000",
	)
	plot_target = os.path.join(
		"..",
		"plots",
		experiment,
	)
	os.makedirs(plot_target, exist_ok=True)

	from ray.rllib.algorithms.algorithm import Algorithm

	restored = Algorithm.from_checkpoint(checkpoint_address)

	## 1D->1D policy
	obs_set_ = np.linspace([-1], [+1], num=100)
	obs_set = [obs[0] for obs in obs_set_]
	pop_set = [(obs_scalar+1)/2 for obs_scalar in obs_set]
	act_set = [(restored.compute_single_action(obs)[0]+1)/2 for obs in obs_set_]

	policy = pd.DataFrame(
		{
			"observation": pop_set,
			"fishing_quota": act_set*pop_set,
		}
	)

	ax = policy.plot(
		title="Fishing under r uncertainty policy",
		x="observation",
		y="fishing_quota",
		xlim=[0,1],
		ylim=[0,1],
		kind="scatter",
	) # ax = matplotlib.axes.Axes
	fig = ax.get_figure()
	fig.savefig(
		os.path.join(plot_target, "policy.png")
	)