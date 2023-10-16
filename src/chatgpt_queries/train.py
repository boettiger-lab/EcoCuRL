import ray
from ray import tune
from ray.rllib.env import MultiAgentEnv

from env import MyTaskSettableEnv as MyEnv

# Define your custom TaskSettableEnv here

# Callback function for agent 1 to set the task for agent 2
def set_task_callback(info):
    agent_2_task = np.random.randint(5)  # Set a random task for agent 2
    info["policy"].model.agent2_task = agent_2_task

# Create a multi-agent training configuration
config = {
    "env": MyEnv,
    "multiagent": {
        "policies": {
            "agent_1": (None, MyEnv.observation_space, MyEnv.action_space, {}),
            "agent_2": (None, MyEnv.observation_space, MyEnv.action_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: "agent_1" if agent_id == "agent_1" else "agent_2",
    },
    "callbacks": {
        "on_episode_start": set_task_callback,
    },
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
