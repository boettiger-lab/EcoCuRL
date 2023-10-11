def simulate(env, policy):
     env.reset()
     timeseries = []
     actionseries = []
     terminated = False
     while not terminated:
             s = env.state
             p = env.state_to_pop(s)
             a = policy(p[0])
             action = [env.effort_to_action(a)]
             obs, rew, terminated, done, info = env.step(action)
             timeseries.append(s)
             actionseries.append(a)
    return timeseries, actionseries