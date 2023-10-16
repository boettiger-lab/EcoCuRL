import ray
from ray import tune
from ray.rllib.env import MultiAgentEnv

from env import MyTaskSettableEnv as MyEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Define your custom TaskSettableEnv here

# Callback function for agent 1 to set the task for agent 2
def set_task_callback(info):
    obs = info["obs"][info["agent"]]
    agent_1_policy = info["policy_map"]["agent_1"]
    agent_2_task = agent_1_policy.compute_actions([obs])[0]  # Use agent 1's policy to determine the task
    info["policy"].model.agent2_task = agent_2_task

class CustomCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        set_task_callback(episode)  # Call your set_task_callback here

# Create a multi-agent training configuration
env = MyEnv()
config = {
    "env": MyEnv,
    "multiagent": {
        "policies": {
            "agent_1": (None, env.observation_space, env.action_space, {}),
            "agent_2": (None, env.observation_space, env.action_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: "agent_1" if agent_id == "agent_1" else "agent_2",
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
