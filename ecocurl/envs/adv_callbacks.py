from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.policy import Policy

from ray.rllib.env import BaseEnv
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.evaluation import Episode, RolloutWorker

from ray.rllib.utils.annotations import (
    ExperimentalAPI,
    override,
    PublicAPI,
    DeveloperAPI,
)

import gymnasium as gym
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type


"""         on env_2 episode start
                   |
_______________    |     _________________        ___________
|              | action  |  ___________  |   act  |         |
|              |----------->| a1_port |  | <------|         |
|              |         |  |_________|  |        | agent_2 |
|   agent_1    |         |               |        |         |
|              |         |     env_2     | ------>|         |
|              |         |               |   rew  |         |
|______________|         |_______________|   obs  |_________|
     ^                                                |     
     |              reward ~ -ep_reward               |
     |------------------------------------------------|
                  obs = previous acion, episode reward
"""

# based on https://github.com/ray-project/ray/blob/master/rllib/ 
#          examples/custom_metrics_and_callbacks.py#L134C5-L149C52
def get_AdvCallbck(
    agent_1: str,
    agent_2: str,
    a1_port: str,
    ):
    """ returns a DefaultCallbacks class """
    # will return AdversarialCallbacks
    class AdversarialCallbacks(DefaultCallbacks):
        """ Callbacks object tailred to training AdversarialExampleEnv. """
        def __init__(self):
            super().__init__()
            self.agent_1 = agent_1
            self.agent_2 = agent_2
            self.a1_port = a1_port

        def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
        ):

            assert episode.length <= 0, (
                "ERROR: `on_episode_start()` callback should be called right "
                f"after env reset! it was called at timestep {episode.length}"
            )

            episode.user_data[f"{self.agent_1}_reward"] = []
            episode.hist_data[f"{self.agent_1}_reward"] = []

            episode.user_data[f"{self.agent_2}_reward"] = []
            episode.hist_data[f"{self.agent_2}_reward"] = []

            episode.user_data[f"{self.a1_port}_val"] = []

        def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
        ):
            assert episode.length > 0, (
                "ERROR: `on_episode_step()` callback should not be called right "
                "after env reset!"
            )

            if episode.length == 1:
                #
                r_now = episode.last_info_for("__common__")[f"{self.a1_port}_value"]
                episode.user_data[f"{self.a1_port}_val"].append(r_now)
                #

            if episode.length > 1:
                #
                #
                rew1 = episode.last_info_for("__common__")[f"{self.agent_1}_reward"]
                rew2 = episode.last_info_for(self.agent_2)[f"{self.agent_2}_reward"]
                #
                episode.user_data[f"{self.agent_1}_reward"].append(rew1)
                episode.user_data[f"{self.agent_2}_reward"].append(rew2)

        def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
            ):
            avg_rew1 = np.mean(episode.user_data[f"{self.agent_1}_reward"])
            avg_rew2 = np.mean(episode.user_data[f"{self.agent_2}_reward"])
            r_val = episode.user_data[f"{self.a1_port}_val"][0]

            episode.custom_metrics["avg_rew1"] = avg_rew1
            episode.custom_metrics["avg_rew2"] = avg_rew2
            episode.custom_metrics[f"{self.a1_port}_val"] = r_val

        def on_train_result(self, algorithm, result, **kwargs):
            # you can mutate the result dict to add new fields to return
            result["callback_ok"] = True

            # obs
            rew_avg = result['sampler_results']['episode_reward_mean'] 
            r_avg = result['custom_metrics'][f'{self.a1_port}_val_mean']
            obs = np.float32([rew_avg, r_avg])
            
            # policy
            agent_1_policy = algorithm.get_policy(self.agent_1)
            agent_1_action = agent_1_policy.compute_single_action(obs)

            # task
            task = (agent_1_action[0] + 1 ) / 2 # trying to avoid tasks < 0, but I don't get why actions lie outside of action space?
            task = np.clip(task, [0], [1])

            # # obs -> confirmed: this is equal to obs
            # obs_val_ = result['custom_metrics']['avg_rew2_mean']
            # obs_ = np.float32([obs_val])

            # agent 1 tracking
            agent_1_rew = result['custom_metrics']['avg_rew1_mean']

            if True:
                print("\n"*2)
                print("On train result")
                # dict_pretty_print(result)
                # print("custom_metrics: ", result['custom_metrics'])
                print('agent 1 obs = [avg rew, avg r] = ', obs)
                print("agent 1 action: ", agent_1_action[0])
                print("task: ", task)
                print("agent 1 rew on train batch: ", agent_1_rew)
                print("\n"*2)

            algorithm.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.set_task(
                        task
                        )))

            del result["custom_metrics"]
    #
    return AdversarialCallbacks