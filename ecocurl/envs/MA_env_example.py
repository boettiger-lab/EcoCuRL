import gymnasium as gym
import numpy as np
import random

from dataclasses import dataclass

from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.examples.env.mock_env import MockEnv, MockEnv2
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from ray.rllib.utils.deprecation import Deprecated

# https://github.com/ray-project/ray/blob/master/rllib/examples/env/multi_agent.py#L361C7-L361C25


@dataclass
class questions:
    """ for readability of GuessTheNumberGame().step """
    lower_than = 0
    higher_than = 1
    equal_to = 2

@dataclass
class agents
    master = 0 # like the dungeon master in dnd, sets the number which is like setting the board
    asker = 1


DEFAULT_OBS = 0


class GuessTheNumberGame(MultiAgentEnv):
    """
    We have two players, 0 and 1. Agent 0 has to pick a number between 0, MAX-1
    at reset. Agent 1 has to guess the number by asking N questions of whether
    of the form of "a <number> is higher|lower|equal to the picked number. The
    action space is MultiDiscrete [3, MAX]. For the first index 0 means lower,
    1 means higher and 2 means equal. The environment answers with yes (1) or
    no (0) on the reward function. Every time step that agent 1 wastes agent 0
    gets a reward of 1. After N steps the game is terminated. If agent 1
    guesses the number correctly, it gets a reward of 100 points, otherwise it
    gets a reward of 0. On the other hand if agent 0 wins they win 100 points.
    The optimal policy controlling agent 1 should converge to a binary search
    strategy.
    """

    """
    Notes:

    Obs:
        {agent: observation} = {0 or 1: 0 or 1}
            --> observatons are a bit static in this env, there are only two observations
                {0:0} = agent 0 is given a 'thumbs up' that its action was registered (setting self._number)
                {1:0} = agent 1 is given a 'nudge' that its action did have an effect (i.e. it was a guess
                        round rather than a _number setting round (i.e. the first round).)

    Actions:
        earlier note:
            {0: question, 1: number} or (equivalently) [question, number] like [questions.lower_than, 10]

        for agent 0:
            0-th entry means nothing
            1-st entry means the number that agent 1 needs to guess

        for agent 1:
            0-th entry means the question asked (lower / higher / equal)
            1-st entry means the number in question
            e.g. {0: 1, 1: 9} means 'is the number (self._number) higher than 9?'
    """

    MAX_NUMBER = 3
    MAX_STEPS = 20

    def __init__(self, config):
        super().__init__()
        self._agent_ids = {agents.master, agents.asker}

        self.max_number = config.get("max_number", self.MAX_NUMBER)
        self.max_steps = config.get("max_steps", self.MAX_STEPS)

        self._number = None

        # [0 or 1] -- equivalently: {0: 0 or 1}
        self.observation_space = gym.spaces.Discrete(2)

        # obs = [(0, 1, or 2), (x for x in range(max_number)] are observations
        self.action_space = gym.spaces.MultiDiscrete([3, self.max_number]) 

    def reset(self, *, seed=None, options=None):
        self._step = 0
        self._number = None
        # agent 0 has to pick a number. So the returned obs does not matter.
        return {agents.master: DEFAULT_OBS}, {}

    def step(self, action_dict):
        # get 'master' agent's action
        game = action_dict.get(agents.master, None)

        if game is not None:
            # ignore the first part of the action and look at the number
            self._number = game[1]
            # next obs should tell agent 1 to start guessing. --> !
            # the returned reward and dones should be on agent 0 who picked a
            # number.
            return (
                {agents.master: DEFAULT_OBS}, # obs
                {agents.master: 0}, # rew
                {agents.master: False, "__all__": False}, # terminated
                {agents.master: False, "__all__": False}, # truncated
                {},
            )

        # the first timestep should set a self._number
        if self._number is None:
            raise ValueError(
                "No number is selected by agent 0. Have you restarted "
                "the environment?"
            )

        # what follows runs if: game is not none (i.e. agents.master called a number)
        #                       and self._number is not none either (i.e., the first
        #                       round already happened).
        #
        #                       the error that is caught above is if the env is not
        #                       reset at the start of a game, so agents.asker is prompted
        #                       for the first move instead of agents.master.

        # get agent 1's action
        direction, number = action_dict.get(agents.asker)
        info = {}
        # always the same, we don't need agent 0 to act ever again, agent 1 should keep
        # guessing. [F Note: communication of the question answers is done through rewards, not observations]
        obs = {agents.asker: DEFAULT_OBS}
        guessed_correctly = False
        terminated = {agents.asker: False, "__all__": False}
        truncated = {agents.asker: False, "__all__": False}
        # everytime agent 1 does not guess correctly agent 0 gets a reward of 1.
        if direction == questions.lower:  # lower
            reward = {
                agents.asker: int(number > self._number), 
                agents.master: 1,
            }
        elif direction == questions.higher:  # higher
            reward = {
                agents.asker: int(number < self._number), 
                agents.master: 1,
            }
        else:  # equal
            guessed_correctly = number == self._number
            reward = {
                agents.asker: guessed_correctly * 100, 
                agents.master: guessed_correctly * -100,
            }
            terminated = {agents.asker: guessed_correctly, "__all__": guessed_correctly}

        self._step += 1
        if self._step >= self.max_steps:  # max number of steps episode is over
            truncated["__all__"] = True
            if not guessed_correctly:
                reward[agents.master] = 100  # agent 0 wins
        return obs, reward, terminated, truncated, info


