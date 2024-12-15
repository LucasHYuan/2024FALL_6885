import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import gym
import tensorflow.compat.v2 as tf
tf.random.set_seed(0)
from tensorflow import keras
from tensorflow.keras import Input

print("Gym version:", gym.__version__)
print("Numpy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
#Gym version: 0.25.2
#Numpy version: 1.26.4
#tensorflow version: 2.17.1

env = gym.make('MountainCar-v0')
env.reset(seed=0)
print('Observation space = {}'.format(env.observation_space))
print('Action space = {}'.format(env.action_space))
print('Position range = {}'.format((env.unwrapped.min_position,
        env.unwrapped.max_position)))
print('Velocity range = {}'.format((-env.unwrapped.max_speed,
        env.unwrapped.max_speed)))
print('Goal position = {}'.format(env.unwrapped.goal_position))

positions, velocities = [], []
observation = env.reset()
while True:
    positions.append(observation[0])
    velocities.append(observation[1])
    next_observation, reward, done, _ = env.step(2)
    if done:
        break
    observation = next_observation

if next_observation[0] > 0.5:
    print('Successfully reached the goal')
else:
    print('Failed to reach the goal')

# Plot position and velocity graphs
fig, ax = plt.subplots()
ax.plot(positions, label='position')
ax.plot(velocities, label='velocity')
ax.legend()
plt.show()

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                columns=['observation', 'action', 'reward',
                'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity
    
    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        
    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)

class DQNAgent:
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.001,
             replayer_capacity=10000, batch_size=64):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity) # Experience replay
         
        self.evaluate_net = self.build_network(input_size=observation_dim,
                output_size=self.action_n, **net_kwargs) # Evaluation network
        self.target_net = self.build_network(input_size=observation_dim,
                output_size=self.action_n, **net_kwargs) # Target network

        self.target_net.set_weights(self.evaluate_net.get_weights())

    def build_network(self, input_size, hidden_sizes, output_size,
                  activation=tf.nn.relu, output_activation=None,
                  learning_rate=0.01):
        model = keras.Sequential()
        model.add(Input(shape=(input_size,)))  # Explicitly define input layer
        for hidden_size in hidden_sizes:
                model.add(keras.layers.Dense(units=hidden_size, activation=activation))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation))  # Output layer

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)  # Optimizer
        model.compile(loss='mse', optimizer=optimizer)
        return model

        
    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation,
                done) # Store experience

        observations, actions, rewards, next_observations, dones = \
                self.replayer.sample(self.batch_size) # Experience replay

        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1. - dones) * next_max_qs
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done: # Update target network
            self.target_net.set_weights(self.evaluate_net.get_weights())

    def decide(self, observation): # Epsilon-greedy strategy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = self.evaluate_net.predict(observation[np.newaxis])
        return np.argmax(qs)

def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation,
                    done)
        if done:
            break
        observation = next_observation
    return episode_reward


# Initialize environment
env = gym.make('MountainCar-v0')
env.reset(seed=0)

# Initialize agent
agent = DQNAgent(
    env, 
    net_kwargs={'hidden_sizes': [64, 64]},  # Hidden layer configuration
    epsilon=0.1  # Exploration rate
)

epsilons = []
rewards = []
# Train agent
episodes = 1000
for episode in range(episodes):
    agent.epsilon = max(0.01, 0.1 * (0.99 ** episode))
    reward = play_qlearning(env, agent, train=True)
    epsilons.append(agent.epsilon)
    rewards.append(reward)

print(rewards)
print(epsilons)

rewards_group = []
for i in range(20):
    add_rewards = 0
    for j in range(50):
        add_rewards += rewards[50*i+j]
    result = add_rewards/50
    rewards_group.append(result)
    print(rewards_group)

epsilons_group = []
for i in range(20):
    add_epsilons = 0
    for j in range(50):
        add_epsilons += epsilons[50*i+j]
    result = add_epsilons/50
    epsilons_group.append(result)
    print(epsilons_group)


# Plot rewards over episodes
plt.plot(rewards)
plt.title("Episode Rewards Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

# Plot smoothed rewards
smooth_rewards = pd.Series(rewards).rolling(window=10).mean()
plt.plot(smooth_rewards)
plt.title("Smoothed Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()
