
from env.A_2D_RobotArmEnv import RoboticArmEnv

if __name__ == "__main__":
    env = RoboticArmEnv(render=True, random_target=True)
    #print(env.observation_space)
    #print(f'obs_space: {env.observation_space}, act_space: {env.action_space}')

    # Main loop
    for episode in range(100):
        env.reset()
        total_reward = 0

        for i in range(5):  # You can set a different episode length if needed
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            #print(f'step: {i}, obs: {obs}, reward: {reward}, done: {done}, info: {info}')
            total_reward += reward
            if done:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()
