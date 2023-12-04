from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
import matplotlib.pyplot as plt
from ray.tune.registry import register_env
from arm4_latest_final_no_velocity import RoboticArmEnv


def create_environment(config):
    return RoboticArmEnv(render=True, random_target=False)


register_env("RoboticArmEnv", create_environment)

config = (PPOConfig().environment("RoboticArmEnv")
          .rollouts(num_rollout_workers=0, create_env_on_local_worker=True))

pretty_print(config.to_dict())

algo = config.build()

episode_rewards = []
episode_lengths = []

for i in range(1000):
    result = algo.train()
    print(f"Iter: {i}, avg.reward: {result['episode_reward_mean']}, eps.length: {result['episode_len_mean']}")

    episode_rewards.append(result['episode_reward_mean'])
    episode_lengths.append(result['episode_len_mean'])

# Plot the learning curves
plt.figure()
plt.plot(episode_rewards, marker='o', label='Episode Mean Rewards', color='b')
plt.plot(episode_lengths, marker='s', label='Episode Lengths', color='g')
plt.title('Training Performance')
plt.xlabel('Iteration')
plt.ylabel('Reward')
#plt.grid()
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('training_performance_plot.png')
plt.show()

#print(pretty_print(resu