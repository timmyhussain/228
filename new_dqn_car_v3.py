# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 18:12:51 2021

@author: user
"""
import os
import numpy as np
import statistics
import collections
import tqdm
import tensorflow as tf
from tensorflow.keras import layers
import gym
import airgym

# def tf_env_step(action):
#     return tf.numpy_function(env.step, [action], 
#                        [tf.float32, tf.float32, tf.int32])

def tf_env_step(action):
    """different output for each state"""
    return tf.numpy_function(env.step, [action], 
                       [tf.float32, tf.int32, tf.float32, tf.float32, tf.int32])

def run_episode(stacked, action_history, state, model, max_steps):
    """
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    Runs a single episode to collect training data.
    """

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    stacked_shape = stacked.shape
    # lidar_shape = lidar.shape
    action_history_shape = action_history.shape
    state_shape = state.shape


    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        
        batched = [tf.expand_dims(x, 0) for x in [stacked, action_history, state]]
        
        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(*batched)
        
        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)
        
        # Store critic values
        values = values.write(t, tf.squeeze(value))
        
        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])
        
        # Apply action to the environment to get next state and reward
        stacked, action_history, goal, reward, done = tf_env_step(action)
        stacked.set_shape(stacked_shape)
        # lidar.set_shape(lidar_shape)
        action_history.set_shape(action_history_shape)
        state.set_shape(state_shape)
        
        # Store reward
        rewards = rewards.write(t, reward)
        
        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards

def get_expected_return(rewards, gamma, standardize=True, eps=1e-3):
    """
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
    Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    
    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
      reward = rewards[i]
      discounted_sum = reward + gamma * discounted_sum
      discounted_sum.set_shape(discounted_sum_shape)
      returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]
    
    if standardize:
      returns = ((returns - tf.math.reduce_mean(returns)) / 
                 (tf.math.reduce_std(returns) + eps))
    
    return returns

# huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
def compute_loss(action_probs, values, returns):
    """
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:
    Computes the combined actor-critic loss."""

    advantage = returns - values
  
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage) #previously negative
  
    critic_loss = mse_loss(values, returns)#previously huber loss
  
    return actor_loss + critic_loss

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


@tf.function
def train_step(stacked, action_history, state, model, optimizer, gamma, max_steps_per_episode):
    """
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
  Runs a model training step."""

    with tf.GradientTape() as tape:
    
        # Run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(
            stacked, action_history, state, model, max_steps_per_episode) 
      
        # Calculate expected returns
        returns = get_expected_return(rewards, gamma)
      
        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 
      
        # Calculating loss values to update our network
        loss = compute_loss(action_probs, values, returns)
    
    # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)
    
    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    episode_reward = tf.math.reduce_sum(rewards)
    
    return episode_reward

class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        
        self.conv1 = layers.Conv2D(4, 3, input_shape=(12,12,3), activation="relu")
        self.conv2 = layers.Conv2D(8, 2, activation="tanh")
        self.pool2d = layers.MaxPool2D(2)
        self.fc1 = layers.Dense(360, activation = "relu") #previously tanh
        self.fc2 = layers.Dense(180, activation = "relu") #previously tanh
        
        # self.conv3 = layers.Conv1D(16, 3, input_shape=(30,24), activation="relu")
        # self.conv4 = layers.Conv1D(32, 3, activation="tanh")
        # self.pool1d = layers.MaxPool1D(2)
        self.fc3 = layers.Dense(32, activation="relu") #previously tanh
        self.embedding_1 = layers.Embedding(input_dim=(num_actions), output_dim=1, input_length=5)
        # self.embedding_2 = layers.Embedding(input_dim=(12), output_dim=1, input_length=2)
        self.fc4 = layers.Dense(2, activation="relu") #previously tanh
        
        self.actor = layers.Dense(num_actions) #previously softmax
        self.critic = layers.Dense(1)
        self.fc5 = layers.Dense(200, activation="relu")
        

    def call(self, stacked, actions, state):
        stacked_features = layers.Flatten()(self.pool2d(self.conv2(self.conv1(stacked))))
        stacked_features = self.fc3(stacked_features)
        # lidar_features = layers.Flatten()(self.pool1d(self.conv4(self.pool1d(self.conv3(lidar)))))
        action_features = layers.Flatten()(self.embedding_1(actions))
        state_features = self.fc4(state)
        # goal_features = layers.Flatten()(self.embedding_2(goal))
        
        x = layers.concatenate([stacked_features, action_features, state_features])
        
        x = self.fc5(x)
        
        return self.actor(x), self.critic(x)
   
env = gym.make("airsim-lidar-car-sample-v0",
               lidar_range=20)


model = ActorCritic(5)
path = "model_4/"
if not os.path.exists(path):
    os.makedirs(path)
    load=False
else:
    load = True
    



min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 200 #reviously 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100 
# consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.9

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

with tqdm.trange(max_episodes) as t:
  for i in t:
    # initial_state = tf.constant(env.reset(), dtype=tf.float32)
    # stacked, lidar, action_history, goal = env.reset()
    
    episode_reward = int(train_step(
        *[tf.convert_to_tensor(i) for i in env.reset()], model, optimizer, gamma, max_steps_per_episode))

    episodes_reward.append(episode_reward)
    running_reward = statistics.mean(episodes_reward)

    t.set_description(f'Episode {i}')
    t.set_postfix(
        episode_reward=episode_reward, running_reward=running_reward)

    # Show average episode reward every 10 episodes
    if i % 10 == 0:
      pass # print(f'Episode {i}: average reward: {avg_reward}')

    if running_reward > reward_threshold and i >= min_episodes_criterion:  
        break
    
    if (i == 1) and load:
        model.load_weights(path+"model_data.h5")
    if i % 5 ==0:
        model.save_weights(path+"model_data.h5")

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')