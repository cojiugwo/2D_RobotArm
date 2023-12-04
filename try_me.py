import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from arm4_latest_final_no_velocity import RoboticArmEnv


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_prob = F.softmax(self.fc2(x), dim=-1)
        return action_prob

class PPOAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, clip_ratio=0.2):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio

    def compute_returns(self, rewards):
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        return returns

    def train_step(self, states, actions, old_probs, returns):
        # Convert lists to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_probs = torch.FloatTensor(old_probs)
        returns = torch.FloatTensor(returns)

        # Compute new action probabilities
        new_probs = self.policy(states).gather(1, actions.unsqueeze(1))

        # Compute surrogate loss
        ratio = new_probs / (old_probs + 1e-8)
        clip_loss = -torch.min(ratio, torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)) * returns
        loss = torch.mean(clip_loss)

        # Optimize policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes):
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        for episode in range(num_episodes):
            states, actions, rewards, old_probs = [], [], [], []
            state = env.reset()

            while True:
                # Collect data
                state = np.array(state, dtype=np.float32)
                action_prob = self.policy(torch.FloatTensor(state))
                action_dist = Categorical(action_prob)
                action = action_dist.sample()
                next_state, reward, done, _ = env.step(action.detach().numpy())

                # Save data
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                old_probs.append(action_prob[action].item())

                state = next_state

                if done:
                    # Compute returns
                    returns = self.compute_returns(rewards)

                    # Train policy
                    self.train_step(states, actions, old_probs, returns)

                    if episode % 10 == 0:
                        print(f"Episode {episode}, Total Reward: {sum(rewards)}")

                    break

if __name__ == "__main__":
    env = RoboticArmEnv(render=False, random_target=False)
    agent = PPOAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0])

    agent.train(env, num_episodes=1000)

    # After training, you can use the trained policy to control the robot
    state = env.reset()
    for _ in range(100):
        action_prob = agent.policy(torch.FloatTensor(state))
        action = torch.argmax(action_prob).item()
        state, _, _, _ = env.step(action)
        env.render()