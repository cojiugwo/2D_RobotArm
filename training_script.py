
import ray
from ray. rllib.algorithms.ppo import PPOConfig
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from arm4_latest_final_no_velocity import RoboticArmEnv



# Import your environment class


# Initialize Ray
ray.init()


def create_environment(config):
    return RoboticArmEnv(render=True, random_target=False)


# Define the stopping criteria
# stopping_criteria = {
#      "time_total_s": 100,  # Replace with your desired training time
#  }


# Register your environment with RLlib
register_env("RoboticArmEnv", create_environment)

# Define RLlib configuration
config = {
    "env": "RoboticArmEnv",
    "framework": "tf",  # Use "tf" for TensorFlow or "torch" for PyTorch
    "num_workers": 1,  # Number of worker processes
    "num_envs_per_worker": 1,
    #"train_batch_size": 1000,
    "num_sgd_iter": 3,#20
    "sgd_minibatch_size": 512,# 128
    "lr": 5e-4,
    "gamma": 0.99,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "clip_param": 0.2,
    "normalize_rewards": True,
    "model": {
        "free_log_std": True,
        "fcnet_hiddens": [256, 256, 256],
        "fcnet_activation": "tanh",
    },
    "create_env_on_driver": True,  # Ensure the local worker has an environment for evaluation
    #"callbacks": EarlyStopping(stopping_criteria),
}

# Initialize the PPO Trainer
trainer = PPOTrainer(config=config)

# Set the evaluation interval
evaluation_interval = 100  # Replace with your desired evaluation interval

# Train the agent
for i in range(5):  # Set the desired number of training iterations
    result = trainer.train()
    print(f"Iteration {i}: {result}")

    # Add periodic evaluation
    if i % evaluation_interval == 0:
        evaluate_result = trainer.evaluate()
        print(f"Evaluation at iteration {i}: {evaluate_result}")

# Save the trained model (optional)
trainer.save("trained_model")

# Shutdown Ray when done
ray.shutdown()