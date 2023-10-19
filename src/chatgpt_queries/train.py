import ray
from ray import tune
from ray.rllib.env import MultiAgentEnv

# from env import MyTaskSettableEnv as MyEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy import Policy

from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker

from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv, MultiAgentEnvWrapper
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv


from ray.rllib.utils.annotations import (
    ExperimentalAPI,
    override,
    PublicAPI,
    DeveloperAPI,
)

from util import dict_pretty_print

class MyEnv(MultiAgentEnv, TaskSettableEnv):
    """ initially a fishery one bc I already understand the reward structure. 
    
    variable: reproduction rate r
    """
    def __init__(self, config = None):
        self.task = None
        self.seed = lambda *args, **kwargs: 42
        #
        self.agent1 = "agent_1"
        self.agent2 = "agent_2"
        self.agents = {self.agent1, self.agent2}
        self._agent_ids = set(self.agents)
        #
        self.observation_space = gym.spaces.Box(
            np.float32([0]), 
            np.float32([1]),
        )  
        self.action_space = gym.spaces.Box(
            np.float32([0]), 
            np.float32([1]),
        )
        self.max_steps = 20  # Set the maximum number of steps per episode
        #
        # pop dynamics
        self.init_pop = np.float32([0.7])
        #
        # io
        self.verbose = 1

    def reset(self, *, seed=42, options=None):
        self.timestep = 0
        self.pop = self.init_pop
        #
        self.agent_2_performance = 0
        self.cur_level = None
        #
        infos = {}
        obs = {
            self.agent1: np.float32([0.]),
        }
        return obs, infos


    def step(self, action_dict):

        task = action_dict.get(self.agent1, None)
        if task is not None:
            self.set_task(task)
            obs = {self.agent2: self.init_pop}
            rew = {self.agent1: 0, self.agent2: 0}
            terminateds = {self.agent2: False, '__all__': False}
            truncateds = {self.agent2: False, '__all__': False}
            infos = {}

            if self.verbose >= 2:
                print(f"""
                step summary [task setting step]:

                timestep = {self.timestep}
                action_dict = {action_dict}
                rewards = {rew}
                obs = {obs}
                """
                )

            return obs, rew, terminateds, truncateds, infos

        if (self.cur_level is None):
            raise ValueError(
                "No cur_level set by agent1. Have you reset the env?"
            )

        agent_2_action = action_dict.get(self.agent2, None)
        if (agent_2_action is None):
            raise ValueError(
                "task was lready set by agent1 but agent2 did not choose an action."
            )
        agent_2_action = np.clip(agent_2_action, [0], [1])


        self.pop_beginning = self.pop.copy()
        harvest = self.pop * agent_2_action
        cost = 0.05 * agent_2_action
        self.pop = self.pop - harvest
        self.pop += self.r * self.pop * (1 - self.pop / self.K)
        penalty = (0.2 - self.pop) * int(self.pop < 0.2) # only get penalty below threshold

        rew2 = (harvest - cost - penalty)[0]
        rew1 = -rew2 * self.r # devalue the easy strategy of just choosing low r values

        self.timestep += 1
        done = {
            self.agent1: self.timestep >= self.max_steps, 
            '__all__': self.timestep >= self.max_steps,
        }

        # Calculate the rewards for both agents
        rew_dict = {
            self.agent1: rew1,  # Reward for agent 1 based on agent 2's performance
            self.agent2: rew2  # You can define a different reward structure for agent 2 if needed
        }

        obs_dict = {
            self.agent2: self.pop
        }

        if self.verbose >= 2:
            print(f"""
            step summary:

            timestep = {self.timestep}
            action_dict = {action_dict}
            r, K = {self.r}, {self.K}
            pop  = {self.pop_beginning}
            pop' = {self.pop}
            harv = {harvest}
            cost = {cost}
            rewards = 1: {rew1}, 2: {rew2}
            obs = {obs_dict}
            """
            )

        infos = {'agent_1': {'agent_1_reward': rew1}, '__common__': {'agent_2_reward': rew2}}

        return obs_dict, rew_dict, done, {'__all__': False}, info

    @override(TaskSettableEnv)
    def sample_tasks(self, n_tasks):
        """Implement this to sample n random tasks."""
        return [np.random.rand() for _ in range(n_tasks)]

    @override(TaskSettableEnv)
    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        return self.cur_level

    @override(TaskSettableEnv)
    def set_task(self, task):
        """Implement this to set the task (curriculum level) for this env."""
        self.cur_level = task
        self.switch_env = True
        r_vals = {'min': 0.05, 'max': 0.95}
        self.r = r_vals['min'] + (r_vals['max'] - r_vals['min']) * task[0]
        self.K = 1

        if self.verbose >= 2:
            print("in set_task(): ")
            print(f"task: {task}, r = {self.r}")
            print(2*"\n")

    @PublicAPI
    def get_agent_ids(self):
        """Returns a set of agent ids in the environment.

        Returns:
            Set of agent ids.
        """
        if not isinstance(self._agent_ids, set):
            self._agent_ids = set(self._agent_ids)
        return self._agent_ids

    def render(self, mode='human'):
        # Implement rendering if needed
        pass

    def close(self):
        # Implement any cleanup here
        pass

    @ExperimentalAPI
    def action_space_contains(self, x) -> bool:
        """Checks if the action space contains the given action.

        Args:
            x: Actions to check.

        Returns:
            True if the action space contains all actions in x.
        """
        if (
            not hasattr(self, "_action_space_in_preferred_format")
            or self._action_space_in_preferred_format is None
        ):
            self._action_space_in_preferred_format = (
                self._check_if_action_space_maps_agent_id_to_sub_space()
            )
        if self._action_space_in_preferred_format:
            return all(self.action_space[agent].contains(x[agent]) for agent in x)

        if log_once("action_space_contains"):
            logger.warning(
                "action_space_contains() of {} has not been implemented. "
                "You "
                "can either implement it yourself or bring the observation "
                "space into the preferred format of a mapping from agent ids "
                "to their individual observation spaces. ".format(self)
            )
        return True

    @ExperimentalAPI
    def action_space_sample(self, agent_ids: list = None):
        """Returns a random action for each environment, and potentially each
            agent in that environment.

        Args:
            agent_ids: List of agent ids to sample actions for. If None or
                empty list, sample actions for all agents in the
                environment.

        Returns:
            A random action for each environment.
        """
        if (
            not hasattr(self, "_action_space_in_preferred_format")
            or self._action_space_in_preferred_format is None
        ):
            self._action_space_in_preferred_format = (
                self._check_if_action_space_maps_agent_id_to_sub_space()
            )
        if self._action_space_in_preferred_format:
            if agent_ids is None:
                agent_ids = self.get_agent_ids()
            samples = self.action_space.sample()
            return {
                agent_id: samples[agent_id]
                for agent_id in agent_ids
                if agent_id != "__all__"
            }
        logger.warning(
            f"action_space_sample() of {self} has not been implemented. "
            "You can either implement it yourself or bring the observation "
            "space into the preferred format of a mapping from agent ids "
            "to their individual observation spaces."
        )
        return {}

    @ExperimentalAPI
    def observation_space_sample(self, agent_ids: list = None):
        """Returns a random observation from the observation space for each
        agent if agent_ids is None, otherwise returns a random observation for
        the agents in agent_ids.

        Args:
            agent_ids: List of agent ids to sample actions for. If None or
                empty list, sample actions for all agents in the
                environment.

        Returns:
            A random action for each environment.
        """

        if (
            not hasattr(self, "_obs_space_in_preferred_format")
            or self._obs_space_in_preferred_format is None
        ):
            self._obs_space_in_preferred_format = (
                self._check_if_obs_space_maps_agent_id_to_sub_space()
            )
        if self._obs_space_in_preferred_format:
            if agent_ids is None:
                agent_ids = self.get_agent_ids()
            samples = self.observation_space.sample()
            samples = {agent_id: samples[agent_id] for agent_id in agent_ids}
            return samples
        if log_once("observation_space_sample"):
            logger.warning(
                "observation_space_sample() of {} has not been implemented. "
                "You "
                "can either implement it yourself or bring the observation "
                "space into the preferred format of a mapping from agent ids "
                "to their individual observation spaces. ".format(self)
            )
        return {}


    # fmt: off
    # __grouping_doc_begin__
    def with_agent_groups(
        self,
        groups: Dict[str, List],
        obs_space: gym.Space = None,
            act_space: gym.Space = None):
        """Convenience method for grouping together agents in this env.

        An agent group is a list of agent IDs that are mapped to a single
        logical agent. All agents of the group must act at the same time in the
        environment. The grouped agent exposes Tuple action and observation
        spaces that are the concatenated action and obs spaces of the
        individual agents.

        The rewards of all the agents in a group are summed. The individual
        agent rewards are available under the "individual_rewards" key of the
        group info return.

        Agent grouping is required to leverage algorithms such as Q-Mix.

        Args:
            groups: Mapping from group id to a list of the agent ids
                of group members. If an agent id is not present in any group
                value, it will be left ungrouped. The group id becomes a new agent ID
                in the final environment.
            obs_space: Optional observation space for the grouped
                env. Must be a tuple space. If not provided, will infer this to be a
                Tuple of n individual agents spaces (n=num agents in a group).
            act_space: Optional action space for the grouped env.
                Must be a tuple space. If not provided, will infer this to be a Tuple
                of n individual agents spaces (n=num agents in a group).

        Examples:
            >>> from ray.rllib.env.multi_agent_env import MultiAgentEnv
            >>> class MyMultiAgentEnv(MultiAgentEnv): # doctest: +SKIP
            ...     # define your env here
            ...     ... # doctest: +SKIP
            >>> env = MyMultiAgentEnv(...) # doctest: +SKIP
            >>> grouped_env = env.with_agent_groups(env, { # doctest: +SKIP
            ...   "group1": ["agent1", "agent2", "agent3"], # doctest: +SKIP
            ...   "group2": ["agent4", "agent5"], # doctest: +SKIP
            ... }) # doctest: +SKIP
        """

        from ray.rllib.env.wrappers.group_agents_wrapper import \
            GroupAgentsWrapper
        return GroupAgentsWrapper(self, groups, obs_space, act_space)

    # __grouping_doc_end__
    # fmt: on

    @PublicAPI
    def to_base_env(
        self,
        make_env = None,
        num_envs: int = 1,
        remote_envs: bool = False,
        remote_env_batch_wait_ms: int = 0,
        restart_failed_sub_environments: bool = False,
    ) -> "BaseEnv":
        """Converts an RLlib MultiAgentEnv into a BaseEnv object.

        The resulting BaseEnv is always vectorized (contains n
        sub-environments) to support batched forward passes, where n may
        also be 1. BaseEnv also supports async execution via the `poll` and
        `send_actions` methods and thus supports external simulators.

        Args:
            make_env: A callable taking an int as input (which indicates
                the number of individual sub-environments within the final
                vectorized BaseEnv) and returning one individual
                sub-environment.
            num_envs: The number of sub-environments to create in the
                resulting (vectorized) BaseEnv. The already existing `env`
                will be one of the `num_envs`.
            remote_envs: Whether each sub-env should be a @ray.remote
                actor. You can set this behavior in your config via the
                `remote_worker_envs=True` option.
            remote_env_batch_wait_ms: The wait time (in ms) to poll remote
                sub-environments for, if applicable. Only used if
                `remote_envs` is True.
            restart_failed_sub_environments: If True and any sub-environment (within
                a vectorized env) throws any error during env stepping, we will try to
                restart the faulty sub-environment. This is done
                without disturbing the other (still intact) sub-environments.

        Returns:
            The resulting BaseEnv object.
        """
        from ray.rllib.env.remote_base_env import RemoteBaseEnv

        if remote_envs:
            env = RemoteBaseEnv(
                make_env,
                num_envs,
                multiagent=True,
                remote_env_batch_wait_ms=remote_env_batch_wait_ms,
                restart_failed_sub_environments=restart_failed_sub_environments,
            )
        # Sub-environments are not ray.remote actors.
        else:
            env = MultiAgentEnvWrapper(
                make_env=make_env,
                existing_envs=[self],
                num_envs=num_envs,
                restart_failed_sub_environments=restart_failed_sub_environments,
            )

        return env

    @DeveloperAPI
    def _check_if_obs_space_maps_agent_id_to_sub_space(self) -> bool:
        """Checks if obs space maps from agent ids to spaces of individual agents."""
        return (
            hasattr(self, "observation_space")
            and isinstance(self.observation_space, gym.spaces.Dict)
            and set(self.observation_space.spaces.keys()) == self.get_agent_ids()
        )

    @DeveloperAPI
    def _check_if_action_space_maps_agent_id_to_sub_space(self) -> bool:
        """Checks if action space maps from agent ids to spaces of individual agents."""
        return (
            hasattr(self, "action_space")
            and isinstance(self.action_space, gym.spaces.Dict)
            and set(self.action_space.keys()) == self.get_agent_ids()
        )



# based on https://github.com/ray-project/ray/blob/master/rllib/ 
#          examples/custom_metrics_and_callbacks.py#L134C5-L149C52
class CustomCallbacks(DefaultCallbacks):

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

        episode.user_data["agent_1_reward"] = []
        episode.hist_data["agent_1_reward"] = []

        episode.user_data["agent_2_reward"] = []
        episode.hist_data["agent_2_reward"] = []

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

        print(5*"\n")
        print("episode:")
        dict_pretty_print(episode.__dict__)
        print(5*"\n")


        rew1 = episode.last_info_for()["agent_1_reward"]
        rew2 = episode.last_info_for()["agent_2_reward"]

        episode.user_data["agent_1_reward"].append(rew1)
        episode.user_data["agent_2_reward"].append(rew2)

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
        avg_rew1 = np.mean(episode.user_data["agent_1_reward"])
        avg_rew2 = np.mean(episode.user_data["agent_2_reward"])

        episode.custom_metrics["avg_rew1"] = avg_rew1
        episode.custom_metrics["avg_rew2"] = avg_rew2

    def on_train_result(self, algorithm, result, **kwargs):

        # obs
        obs_val = result['sampler_results']['episode_reward_mean'] 
        obs = np.float32([obs_val])

        # policy
        agent_1_policy = algorithm.get_policy("agent_1")
        agent_1_action = agent_1_policy.compute_single_action(obs)

        # task
        task = (agent_1_action[0] + 1 ) / 2 # trying to avoid tasks < 0, but I don't get why actions lie outside of action space?
        task = np.clip(task, [0], [1])

        # obs
        obs_val_ = result['custom_metrics']['agent_2_reward']
        obs_ = np.float32([obs_val])

        if True:
            print("\n"*5)
            print("On train result")
            print(agent_1_policy)
            print("obs:  ", obs)
            print("obs_: ", obs_)
            print("action: ", agent_1_action[0])
            print("task: ", task)
            print("\n"*5)

        algorithm.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_task(
                    task
                    )))

# Create a multi-agent training configuration
env = MyEnv()

def policy(agent_id: str):
    i = {"agent_1": 1, "agent_2": 2}[agent_id]

    policy_config = PPOConfig.overrides(
                    model={
                        "custom_model": ["agent_1", "agent_2"][i % 2],
                    },
                    gamma=0.99,
                )
    return PolicySpec(config=policy_config)

policies = {agent_id: policy(agent_id) for agent_id in ["agent_1", "agent_2"]}

config = {
    "env": MyEnv,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": lambda agent_id, *args, **kwargs: agent_id,
    },
    "callbacks_class": CustomCallbacks,
        # "on_episode_start": set_task_callback,
}

# Initialize Ray and train the agents
ray.init()
tune.run(
    "PPO",
    config=config,
    stop={"training_iteration": 100},  # Define your stopping criteria
    checkpoint_at_end=True,
)
ray.shutdown()
