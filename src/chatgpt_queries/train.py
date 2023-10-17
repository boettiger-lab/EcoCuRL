import ray
from ray import tune
from ray.rllib.env import MultiAgentEnv

# from env import MyTaskSettableEnv as MyEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv, MultiAgentEnvWrapper


from ray.rllib.utils.annotations import (
    ExperimentalAPI,
    override,
    PublicAPI,
    DeveloperAPI,
)

from util import dict_pretty_print

class MyEnv(MultiAgentEnv):
    def __init__(self, config = None):
        self.task = None
        self.seed = lambda *args, **kwargs: 42
        #
        self.agent1 = "agent_1"
        self.agent2 = "agent_2"
        self.agents = {self.agent1, self.agent2}
        self._agent_ids = set(self.agents)
        #
        self.observation_space = gym.spaces.Discrete(5)  # Replace with your actual observation space
        self.action_space = gym.spaces.Discrete(2)  # Replace with your actual action space
        self.max_steps = 100  # Set the maximum number of steps per episode

    def reset(self, *, seed=42, options=None):
        self.task = np.random.randint(5)  # Randomly set the task for agent 2
        self.current_step = 0
        self.agent_2_performance = 0
        print(5*"\n",self._agent_ids,5*"\n")
        infos = {}
        obs = {
            self.agent1: self._get_obs(self.agent1),
            self.agent2: self._get_obs(self.agent2)
        }
        return obs, infos


    def step(self, action_dict):
        assert self.agent1 in action_dict and self.agent2 in action_dict

        # Update the performance of agent 2 based on its action
        if action_dict[self.agent2] == self.task:
            self.agent_2_performance += 1

        self.current_step += 1
        done = {self.agent1: self.current_step >= self.max_steps}

        # Calculate the rewards for both agents
        reward_dict = {
            self.agent1: self.agent_2_performance,  # Reward for agent 1 based on agent 2's performance
            self.agent2: 0  # You can define a different reward structure for agent 2 if needed
        }

        obs_dict = {
            self.agent1: self._get_obs(self.agent1),
            self.agent2: self._get_obs(self.agent2)
        }

        return obs_dict, reward_dict, done, {'__all__': False}, {}

    def _get_obs(self, agent):
        # Replace this with logic to generate observations for each agent
        return 0

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



# Callback function for agent 1 to set the task for agent 2
def set_task_callback(info):
    print("\n"*5 + "info: ")
    dict_pretty_print(info.__dict__)
    print("\n"*5)
    # obs = info["obs"][info["agent"]]
    # agent_1_policy = info.policy_mapping_fn("agent_1")
    # agent_2_task = agent_1_policy.compute_actions([obs])[0]  # Use agent 1's policy to determine the task
    # info["policy"].model.agent2_task = agent_2_task

class CustomCallbacks(DefaultCallbacks):

    def on_train_result(self, algorithm, result, **kwargs):
        print("result type: ", type(result))
        print("algo: ", algorithm)
        print("kwargs: ")
        dict_pretty_print(kwargs)
        print("results: ")
        print(result)
        print(5*"\n")
        # if result["episode_reward_mean"] > 200:
        #     task = 2
        # elif result["episode_reward_mean"] > 100:
        #     task = 1
        # else:
        #     task = 0
        # algorithm.workers.foreach_worker(
        #     lambda ev: ev.foreach_env(
        #         lambda env: env.set_task(task)))

# Create a multi-agent training configuration
env = MyEnv()
config = {
    "env": MyEnv,
    "multiagent": {
        "policies": {
            "agent_1": (None, env.observation_space, env.action_space, {}),
            "agent_2": (None, env.observation_space, env.action_space, {}),
        },
        "policy_mapping_fn": lambda agent_id, *args, **kwargs: "agent_1" if agent_id == "agent_1" else "agent_2",
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
