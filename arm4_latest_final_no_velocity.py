import math
import math
import numpy as np
import gym
from gym.spaces import Box

import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module='pygame')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='pkg_resources')

import pygame


class RoboticArmEnv(gym.Env):
    def __init__(self, render=False, random_target=False, *args, **kwargs):
        pygame.init()

        self.theta_dim = 3
        self.theta_high = np.array([np.pi, np.pi, np.pi])
        self.theta_low = np.array([-np.pi, -np.pi, -np.pi])

        self.random_target = random_target

        self.threshold = 5  # 1, 0.1  could try 0.5, 0.05, etc.

        self.theta = np.zeros(self.theta_dim)

        self.max_steps = 50
        self.current_step = 1

        self.render_screen = None
        self.width, self.height = 800, 600
        self.arm_color = (0, 0, 0)
        self.target_color = (255, 0, 0)
        self.font = pygame.font.Font(None, 36)

        self.target = self.get_target()

        self.reward = 0
        self.prev_distance = 0

        # Define joint limit angles
        self.joint_min_angles = [-np.pi, -np.pi, -np.pi]
        self.joint_max_angles = [np.pi, np.pi, np.pi]

        self.observation_space = Box(low=np.array([-np.inf] * 9), high=np.array([np.inf] * 9), dtype=np.float32)

        self.action_space = Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)

        # Initialize rendering based on the user's choice
        self.render_enabled = render
        if self.render_enabled:
            self.initialize_render()

    def initialize_render(self):
        if self.render_screen is None:
            self.render_screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("2D Robotic Arm Environment")

    def get_observation(self):
        return np.concatenate(
            [self.theta,
             self.get_end_effector_position(),
             self.target, [self.width // 2, self.height // 2]])

    def reset(self):
        self.theta = [0, 0, 0]  # np.random.uniform(self.theta_low, self.theta_high, size=self.theta_dim)
        self.current_step = 1
        self.reward = 0
        if not self.random_target:
            # Do nothing, keep the existing target
            pass
        else:
            self.target = self.get_target()

        self.prev_distance = np.linalg.norm(self.get_end_effector_position() - self.target)
        return self.get_observation()

    def get_target(self):
        max_arm_length = self.theta_dim * 100
        target_radius = random.uniform(0, max_arm_length)
        target_theta = random.uniform(0, 2 * math.pi)
        target_x = target_radius * math.cos(target_theta)
        target_y = target_radius * math.sin(target_theta)
        target_x += self.width // 2
        target_y += self.height // 2
        return np.array([target_x, target_y])

    def calculate_joint_penalties(self):
        joint_penalties = np.zeros(self.theta_dim)
        for i in range(self.theta_dim):
            if self.theta[i] < self.joint_min_angles[i]:
                joint_penalties[i] = -0.1 * (self.theta[i] - self.joint_min_angles[i])
            elif self.theta[i] > self.joint_max_angles[i]:
                joint_penalties[i] = -0.1 * (self.theta[i] - self.joint_max_angles[i])
        return joint_penalties

    def get_reward(self, curr_distance, action, joint_penalties):
        # reward_distance1 = self.prev_distance - curr_distance
        reward_distance = -curr_distance
        reward_control = -0.1 * np.square(action).sum()
        reward = reward_distance + reward_control  # + reward_distance1
        reward += np.sum(joint_penalties)

        # Scale down the reward
        reward_scale = 0.01  # Adjust this scaling factor as needed
        reward *= reward_scale
        return reward

    def is_done(self, curr_distance):
        return curr_distance <= self.threshold or self.current_step == self.max_steps

    def step(self, action):
        control_input = np.clip(action, -1, 1)

        # Calculate joint penalties based on the current joint angles
        joint_penalties = self.calculate_joint_penalties()

        for i in range(self.theta_dim):
            self.theta[i] += control_input[i]
            self.theta[i] = np.clip(self.theta[i], self.joint_min_angles[i], self.joint_max_angles[i])

        curr_distance = np.linalg.norm(self.get_end_effector_position() - self.target)
        reward = self.get_reward(curr_distance, action, joint_penalties)

        done = self.is_done(curr_distance)

        self.reward = reward
        self.prev_distance = curr_distance
        self.current_step += 1
        obs = self.get_observation()

        # Render if enabled
        if self.render_enabled:
            self.render()

        return obs, reward, done, {}

    def get_end_effector_position(self):
        x = self.width // 2
        y = self.height // 2
        for angle in range(len(self.theta)):
            x += 100 * np.cos(np.sum(self.theta[:angle + 1]))
            y += 100 * np.sin(np.sum(self.theta[:angle + 1]))
        return np.array([x, y])

    def render(self):
        if self.render_screen is not None:
            self.render_screen.fill((255, 255, 255))

            # Draw grid
            grid_spacing = 50
            x_center, y_center = self.width // 2, self.height // 2
            for x in range(0, self.width, grid_spacing):
                pygame.draw.line(self.render_screen, (200, 200, 200), (x, 0), (x, self.height))
            for y in range(0, self.height, grid_spacing):
                pygame.draw.line(self.render_screen, (200, 200, 200), (0, y), (self.width, y))

            # Draw coordinate axes
            pygame.draw.line(self.render_screen, (0, 0, 0), (x_center, 0), (x_center, self.height), 2)
            pygame.draw.line(self.render_screen, (0, 0, 0), (0, y_center), (self.width, y_center), 2)

            # Draw arm segments
            x, y = x_center, y_center
            for i, angle in enumerate(self.theta):
                segment_length = 100
                segment_x = x + segment_length * np.cos(angle)
                segment_y = y + segment_length * np.sin(angle)

                segment_color = (255, 0, 0) if i % 2 == 0 else (0, 0, 255)

                pygame.draw.circle(self.render_screen, segment_color, (int(x), int(y)), 5)
                pygame.draw.line(self.render_screen, segment_color, (x, y), (int(segment_x), int(segment_y)), 5)
                x, y = segment_x, segment_y

            # Draw target and end effector
            pygame.draw.circle(self.render_screen, (255, 0, 0), self.target.astype(int), 10)
            pygame.draw.circle(self.render_screen, (0, 0, 0), (int(x), int(y)), 10)

            # Draw reward text
            reward_text = self.font.render(f'Reward: {self.reward:.2f}', True, (0, 0, 0))
            self.render_screen.blit(reward_text, (10, 10))

            pygame.display.update()
            pygame.time.Clock().tick(10)

    def close(self):
        pygame.quit()
        self.render_screen = None

# The rest of your code for training the agent remains unchanged
env = RoboticArmEnv(render=True, random_target=True)
print(f'obs_space_shape: {env.observation_space.shape}')
print(f'get_observation()_shape:{env.get_observation()}')

# Main loop
if __name__ == "__main__":
    for episode in range(50):
        env.reset()
        total_reward = 0

        for i in range(10):  # You can set a different episode length if needed
            action = env.action_space.sample()
            print(f'action_sam: {action}')
            obs, reward, done, _ = env.step(action)
            print(f'step: {i}, obs: {obs}, reward: {reward}, done: {done}')
            total_reward += reward
            if done:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()