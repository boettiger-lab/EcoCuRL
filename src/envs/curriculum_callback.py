import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks

class CurrLCallbacks(DefaultCallbacks):
    """ tbd: make specific to our case """
    def on_train_result(self, algorithm, result, **kwargs):
        if result["episode_reward_mean"] > 200:
            task = 2
        elif result["episode_reward_mean"] > 100:
            task = 1
        else:
            task = 0
        algorithm.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_task(task)))