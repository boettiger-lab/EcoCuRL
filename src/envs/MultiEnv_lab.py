"""
PLAYGROUND


here I will start from the source code of the MultiAgentEnv abstract class and try to 
create a MultiAgentEnv of my liking.

use  https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py#L432C5-L432C21

as a guide (which constructs an explicit env from a single-agent env.)
"""


import gymnasium as gym
import logging
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type

from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.utils.annotations import (
    ExperimentalAPI,
    override,
    PublicAPI,
    DeveloperAPI,
)
from ray.rllib.utils.typing import (
    AgentID,
    EnvCreator,
    EnvID,
    EnvType,
    MultiAgentDict,
    MultiEnvDict,
)
from ray.util import log_once

# If the obs space is Dict type, look for the global state under this key.
ENV_STATE = "state"

logger = logging.getLogger(__name__)

def logistic(params, pop):
    p = params
    return pop + p['r'] * pop * (1 - pop / p['k'])


@PublicAPI
class MA_logistic_env(TaskSettableEnv):
    """An environment that hosts multiple independent agents.

    Agents are identified by (string) agent ids. Note that these "agents" here
    are not to be confused with RLlib Algorithms, which are also sometimes
    referred to as "agents" or "RL agents".

    The preferred format for action- and observation space is a mapping from agent
    ids to their individual spaces. If that is not provided, the respective methods'
    observation_space_contains(), action_space_contains(),
    action_space_sample() and observation_space_sample() have to be overwritten.
    """

    def __init__(self):
        """ states/observations and actions are in [0,1]. """

        # algorithmics
        self.observation_space = gym.spaces.Box(
            [0],
            [1],
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            [0],
            [1],
            dtype=np.float32
        )
        self._agent_ids = {'env_setter', 'env_controller'}

        # ecology / pop dynamics
        self.init_state = env_config.get(
            'init_state', 
            np.float32([0.5])
        )
        self.dynamics = logistic
        self.pop_threshold1 = 0.2 # above which the population becomes a problem
        self.pop_threshold2 = 0.7 # above which the problem ends
        self.tmax = 100

        # curriculum learning
        # self.curr_lvl = curr_lvl
        # self.num_lvls = 10
        self.switch_env = False



        # F Note: need to understand these format variables
        #
        # Do the action and observation spaces map from agent ids to spaces
        # for the individual agents?
        if not hasattr(self, "_action_space_in_preferred_format"):
            self._action_space_in_preferred_format = None
        if not hasattr(self, "_obs_space_in_preferred_format"):
            self._obs_space_in_preferred_format = None

    @PublicAPI
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Resets the env and returns observations from ready agents.

        Args:
            seed: An optional seed to use for the new episode.

        Returns:
            New observations for each ready agent.

        Examples:
            >>> from ray.rllib.env.multi_agent_env import MultiAgentEnv
            >>> class MyMultiAgentEnv(MultiAgentEnv): # doctest: +SKIP
            ...     # Define your env here. # doctest: +SKIP
            ...     ... # doctest: +SKIP
            >>> env = MyMultiAgentEnv() # doctest: +SKIP
            >>> obs, infos = env.reset(seed=42, options={}) # doctest: +SKIP
            >>> print(obs) # doctest: +SKIP
            {
                "car_0": [2.4, 1.6],
                "car_1": [3.4, -3.2],
                "traffic_light_1": [0, 3, 5, 1],
            }
        """
        """
        observations: 
            env_setter: 0 = must set env_specifier, 1 = no need to set env_specifier
            env_controller: population of the system
        """
        self.timestep = 0
        self.state = self.init_state
        obs = {'env_setter': np.float32([0])}
        return obs, {}

    @PublicAPI
    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns:
            Tuple containing 1) new observations for
            each ready agent, 2) reward values for each ready agent. If
            the episode is just started, the value will be None.
            3) Terminated values for each ready agent. The special key
            "__all__" (required) is used to indicate env termination.
            4) Truncated values for each ready agent.
            5) Info values for each agent id (may be empty dicts).

        Examples:
            >>> env = ... # doctest: +SKIP
            >>> obs, rewards, terminateds, truncateds, infos = env.step(action_dict={
            ...     "car_0": 1, "car_1": 0, "traffic_light_1": 2,
            ... }) # doctest: +SKIP
            >>> print(rewards) # doctest: +SKIP
            {
                "car_0": 3,
                "car_1": -1,
                "traffic_light_1": 0,
            }
            >>> print(terminateds) # doctest: +SKIP
            {
                "car_0": False,    # car_0 is still running
                "car_1": True,     # car_1 is terminated
                "__all__": False,  # the env is not terminated
            }
            >>> print(infos) # doctest: +SKIP
            {
                "car_0": {},  # info for car_0
                "car_1": {},  # info for car_1
            }
        """
        env_specifier = action_dict.get('env_setter', None)
        if env_specifier is not None:
            # env_setter only gets an observation on reset
            # --> so it should only provide an action right
            #     after reset. From there on, env_specifier = None.
            self.env_specifier = env_specifier
            self.set_env()
            return (
                {'env_controller': self.state}, # observation
                {'env_controller': 0}, # reward
                {'env_controller': False, "__all__": False}, # terminated
                {'env_controller': False, "__all__": False}, # truncated
                {}, # infos
            )

        if self.env_specifier is None:
            raise ValueError(
                "No env_specifier value chosen by 'env_setter'. Have you reset the env?"
            )

        #
        # from here on, self.env_specifier has a value
        removal_effort = action_dict['env_controller'][0]
        logger.debug(f"removal effort: {removal_effort}")

        self.state = self.state * (1 - 0.3 * removal_effort)
        logger.debug(f"post removal state: {self.state}")

        self.state = self.dynamics(self.params, self.state.copy())
        logger.debug(f"post dynamics state: {self.state}")

        setter_reward += (
            - 1 # penalized per timestep survived
            + 0.05 * removal_effort 
            + self.damage_quantifier(self.pop_threshold1, self.state)
        ) 
        solver_reward -= setter_reward
        logger.debug(
            f"setter rew: {setter_reward:.3f}, " 
            f"solver rew: {solver_reward:.3f}"
        )

        solver_terminated = False
        if self.state > self.pop_threshold2:
            solver_terminated = True
            # this is a bit ad-hoc:
            setter_reward += 100 * ((100 - self.timestep) ** 2) / (100 ** 2)

        if self.timestep >= 200:
            solver_terminated = True

        return (
            {'env_solver': self.state}, # obs
            {'env_setter': setter_reward, 'env_solver': solver_reward}, # rew
            {'env_solver': solver_terminated, '__all__': solver_terminated}, # terminateds
            {'env_setter': False, 'env_solver': False}, # truncateds
            {}, # infos
        )

    def set_env(self):
        """ higher curriculum lvl = larger spread of possible r values """

        # r_rng_start = [0.45, 0.55]
        # r_rng_end = [0.05, 0.95]
        # r_rng_diff = r_rng_end - r_rng_start

        # [r_floor, r_ceil]  = r_rng_start + r_rng_diff * (self.curr_lvl / self.num_lvls)
        # r_width = r_ceil - r_floor

        r_floor = 0.05
        r_ceil = 0.95

        self.params = {
            'r': r_floor + r_width * self.env_specifier[0],
            'K': 0.8,
        }
        logger.debug(f"env_specifier: {self.env_specifier}, r: {self.params['r']}")

    def damage_quantifier(self, threshold, pop):
    """ quantifies ecosystem damage. """
    if pop <= threshold:
        return 0
    else:
        return pop[0] - threshold

    @ExperimentalAPI
    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        """Checks if the observation space contains the given key.

        Args:
            x: Observations to check.

        Returns:
            True if the observation space contains the given all observations
                in x.
        """
        if (
            not hasattr(self, "_obs_space_in_preferred_format")
            or self._obs_space_in_preferred_format is None
        ):
            self._obs_space_in_preferred_format = (
                self._check_if_obs_space_maps_agent_id_to_sub_space()
            )
        if self._obs_space_in_preferred_format:
            for key, agent_obs in x.items():
                if not self.observation_space[key].contains(agent_obs):
                    return False
            if not all(k in self.observation_space.spaces for k in x):
                if log_once("possibly_bad_multi_agent_dict_missing_agent_observations"):
                    logger.warning(
                        "You environment returns observations that are "
                        "MultiAgentDicts with incomplete information. "
                        "Meaning that they only contain information on a subset of"
                        " participating agents. Ignore this warning if this is "
                        "intended, for example if your environment is a turn-based "
                        "simulation."
                    )
            return True

        logger.warning(
            "observation_space_contains() of {} has not been implemented. "
            "You "
            "can either implement it yourself or bring the observation "
            "space into the preferred format of a mapping from agent ids "
            "to their individual observation spaces. ".format(self)
        )
        return True

    @ExperimentalAPI
    def action_space_contains(self, x: MultiAgentDict) -> bool:
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
    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
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
    def observation_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
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

    @PublicAPI
    def get_agent_ids(self) -> Set[AgentID]:
        """Returns a set of agent ids in the environment.

        Returns:
            Set of agent ids.
        """
        if not isinstance(self._agent_ids, set):
            self._agent_ids = set(self._agent_ids)
        return self._agent_ids

    @PublicAPI
    def render(self) -> None:
        """Tries to render the environment."""

        # By default, do nothing.
        pass

    # fmt: off
    # __grouping_doc_begin__
    def with_agent_groups(
        self,
        groups: Dict[str, List[AgentID]],
        obs_space: gym.Space = None,
            act_space: gym.Space = None) -> "MultiAgentEnv":
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
        make_env: Optional[Callable[[int], EnvType]] = None,
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