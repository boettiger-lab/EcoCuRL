def simulate(env, policy):
	env.reset()
	timeseries = []
	actionseries = []
	terminated = False
	while not terminated:
		s = env.state
		p = env.state_to_pop(s)
		a = policy(p)
		action = env.effort_to_action(a)
		obs, rew, terminated, done, info = env.step(action)
		timeseries.append(p)
		actionseries.append(a)
		print(info)
	return timeseries, actionseries