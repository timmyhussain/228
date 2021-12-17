# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 07:45:06 2021

@author: user
"""
import time
from numpy.core.defchararray import join
from helper import Grid, Model, get_error
from memory import Memory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import tensorflow.keras as keras

# obstacles = np.load("map1-new/obstacles.npy", allow_pickle=True)
# grid = Grid(10, 10, 10, -65.32, -53.20, obstacles)
# obstacles = [[]]
# grid.create_map()
# 
max_number_steps = 50


class Train:
    def __init__(self, Memory, Grid, Model, max_number_steps, epsilon, gamma, epochs):
        obstacles = np.load("map1-new/obstacles.npy", allow_pickle=True)
        self.grid = Grid(10, 10, 10, -65.32, -53.20, obstacles)
        self.memory = Memory(1000)
        self.max_number_steps = max_number_steps
        self.epsilon = epsilon
        self.gamma=gamma
        self.episode_belief_history= []
        self.episode_position_history = []
        self.epochs = epochs
        self.Model = Model(32, 0.01)
        # self.Model_2 = keras.models.load_model("lidar_model/model.h5")
        
    def run_episode(self):
        belief, obstacles, goal = self.grid.reset()
        belief_history = [belief]
        position_history = [self.grid.position_grid]
        for t in range(self.max_number_steps):
            action = self.get_action(belief, obstacles, goal)
            new_belief, new_obstacles, new_goal, state_reward, obstacle_reward, done = self.grid.step(belief, action)
            current_experience = (belief, obstacles, goal, action, state_reward, obstacle_reward, new_belief, new_obstacles)
            # print(obstacles)
            # print(action)
            # print(obstacle_reward)
            # time.sleep(3)
            # print((belief == new_belief).all())
            self.memory.add_sample(current_experience)
            belief= new_belief.copy()
            obstacles = new_obstacles.copy()
            belief_history.append(belief)
            position_history.append(self.grid.position_grid)
            if done:
                # self.grid.reset()
                break
        self.episode_belief_history.append(belief_history)
        self.episode_position_history.append(position_history)
    
    def train(self):
        for t in range(self.epochs):
            self.run_episode()
            self._replay()
            # self.epsilon = self.epsilon/(float(t)+1.0)
            if t % 100 == 0:
                self.epsilon = 0.9*self.epsilon
                print(t)
                self.Model.save_model("final_attempt_model_v2/")
        pass
    
    def get_action(self, belief, obstacles, goal):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 9)
        else:
            # obstacle_actions = self.Model_2.predict(np.array(obstacles, np.int32).reshape(1, 8))
            obstacle_actions, belief_actions = self.Model.model.predict({
                                                    "belief": belief.reshape(1,144), 
                                                    "obstacles": np.array(obstacles, np.int32).reshape(1, 8), 
                                                    "goal": np.array(goal, np.int32).reshape(1, 2)}
                                                    )
            joint_actions = np.abs(obstacle_actions)/np.sum(obstacle_actions) * np.abs(belief_actions)/np.sum(belief_actions)
            joint_actions = joint_actions/np.sum(joint_actions)
            return np.argmax(joint_actions)
      
        
    def _replay(self):
        '''Same idea as train but for a batch of size batch_size
        Input
         - model_a: boolean, True if training model A
        '''
        #Pull batch of size batch_size from Memory module
        batch = self.memory.get_batch(self.Model.batch_size)
        
        #belief, obstacles, action, reward, new_belief, new_obstacles
        #initialize lists to store data from batch
        beliefs = []
        obstacles = []
        goals = []
        actions = []
        state_rewards = []
        obstacle_rewards = []
        new_beliefs = []
        new_obstacles = []
        # alpha = 1/self._Model._get_n(model_a)
        
        #split experiences in batch and append to appropriate lists
        for exp in batch:
            
            beliefs.append(exp[0].flatten())
            obstacles.append(exp[1])
            goals.append(exp[2])
            actions.append(exp[3])
            state_rewards.append(exp[4])
            obstacle_rewards.append(exp[5])
            new_beliefs.append(exp[6].flatten())
            new_obstacles.append(exp[7])
            
        #turn current and next states into numpy arrays of size (batch_size, num_states) 
        beliefs = np.vstack(beliefs)
        obstacles = np.vstack(obstacles)
        goals = np.vstack(goals)
        actions = np.vstack(actions)
        state_rewards = np.vstack(state_rewards)
        obstacle_rewards = np.vstack(obstacle_rewards)
        new_beliefs = np.vstack(new_beliefs)
        new_obstacles = np.vstack(new_obstacles)
        
        # print(new_beliefs.shape)
        # print(actions.shape)
        
        #actions predicted by model for next state
        # print(self.Model.model.predict({"belief": new_beliefs, "obstacles": new_obstacles, "goal": goals}))
        # a_star = np.argmax(self.Model.model.predict({"belief": new_beliefs, "obstacles": new_obstacles, "goal": goals}), axis=1)
        obstacle_actions, belief_actions = self.Model.model.predict({"belief": new_beliefs, "obstacles": new_obstacles, "goal": goals})
        a_star = np.argmax(belief_actions, axis=1)
        
        #Q values predicted by model for current state (the argmax of these are already in actions list)
        # q_s = self.Model.model.predict({"belief": beliefs, "obstacles": obstacles, "goal": goals})
        obstacle_values, state_values = self.Model.model.predict({"belief": beliefs, "obstacles": obstacles, "goal": goals})
        # joint_actions = obstacle_actions*belief_actions
        # joint_actions = joint_actions/np.sum(joint_actions)
        state_values = state_values.copy()
        obstacle_values = obstacle_values.copy()
        
        #Q values predicted by not model being updated for next state
        # q_s_astar = self.Model.model.predict({"belief": new_beliefs, "obstacles": new_obstacles, "goal": goals})
        obstacle_values, state_values = self.Model.model.predict({"belief": new_beliefs, "obstacles": new_obstacles, "goal": goals})
        # joint_actions = obstacle_actions*belief_actions
        # joint_actions = joint_actions/np.sum(joint_actions)
        # q_s_astar = joint_actions.copy()
        new_state_values = state_values.copy()
        new_obstacle_values = obstacle_values.copy()

        
        #creating target array; basically batch version of function in train
        # print(q_s.shape)
        for ex in range(min(self.Model.batch_size, len(obstacle_values))):
            # n = self._Model._get_n(model_a, actions[ex])
            # alpha = 0.01
            # q_s[ex, actions[ex]] = q_s[ex, actions[ex]] + alpha*(rewards[ex] + \
            #                       self.gamma*q_s_astar[ex, a_star[ex]] - \
            #                       q_s[ex, actions[ex]])
                
            # q_s[ex, actions[ex]] = rewards[ex] + \
            #                       self.gamma*q_s_astar[ex, a_star[ex]]
            state_values[ex, actions[ex]] = state_rewards[ex] + \
                                  self.gamma*new_state_values[ex, a_star[ex]]

            obstacle_values[ex, actions[ex]] = obstacle_rewards[ex] #+ \
                                  #self.gamma*new_obstacle_values[ex, a_star[ex]]

            # print(obstacles[ex,:])
            # print(obstacle_values[ex, :])
            # print(actions[ex])
            # print(obstacle_rewards[ex])
            # time.sleep(3)
            #at the end of this we will have trained the model on batch_size examples
            #so increment n for model being updated batch_size times by calling this once within each loop 
            
            # self._Model._set_n(model_a, actions[ex])
        
        # n = self._Model._get_n(model_a, actions[ex])
        
        #Double Q-Learning (2010)
        
        # alpha = 4/n
        # if model_a:
            # K.set_value(self._Model.model_a.optimizer.learning_rate, alpha**0.8)
        # print(len(q_s))
        self.Model.model.fit({"belief": new_beliefs, "obstacles": new_obstacles, "goal": goals}, 
                    {"obstacle_actions": obstacle_values, "belief_actions": state_values}, 
                    verbose=1)


# trainer = Train(Memory, Grid, Model, 50, epsilon=1, gamma=0.9, epochs=10000)
# trainer.run_episode()
# trainer.train()

# T = 1
# for i in range(T):
#     current_state = grid.reset()
#     current_state = 

#%%

class Test:
    def __init__(self, Memory, Grid, Model, max_number_steps):
        obstacles = np.load("map1-new/obstacles.npy", allow_pickle=True)
        self.grid = Grid(10, 10, 10, -65.32, -53.20, obstacles)
        self.memory = Memory(1000)
        self.max_number_steps = max_number_steps
        self.episode_belief_history= []
        self.episode_position_history = []
        self.joint_model = keras.models.load_model("final_attempt_model_v2/trained_model.h5")
        self.Model = keras.models.load_model("belief_model/model.h5")
        self.Model_2 = keras.models.load_model("lidar_model/model.h5")
        self.gamma=0.9
        self.max_depth = 2
        self.error_history = []

        
    def run_episode(self):
        belief, obstacles, goal = self.grid.reset()
        belief_history = [belief]
        position_history = [self.grid.position_grid]
        for t in range(self.max_number_steps):
            # print(t)
            action = self.get_action(belief, obstacles, goal)
            # action = self.lookahead(belief, 2)[1]
            res = self.grid.step(belief, action)
            # current_experience = (belief, obstacles, goal, action, reward, new_belief, new_obstacles)
            # self.memory.add_sample(current_experience)
            belief= res[0].copy()
            obstacles = res[1].copy()
            belief_history.append(belief)
            position_history.append(self.grid.position_grid)
            if res[-1]:
                # self.grid.reset()
                break
        
            self.error_history.append(get_error(belief, *self.grid.goal_ix))
        self.episode_belief_history.append(belief_history)
        self.episode_position_history.append(position_history)
        
        # return self.episode_belief_history, self.episode_position_history

        
    def get_action(self, belief, obstacles, goal):
        # print(obstacles)

        # obstacle_actions = self.Model_2.predict(np.array(obstacles, np.int32).reshape(1,8))
        # belief_actions = self.Model.predict({
        #         "beliefs": belief.reshape(1,144), 
        #         "goals": np.array(goal, np.int32).reshape(1, 2)}
        #         )

        obstacle_actions, belief_actions = self.joint_model.predict({
                "belief": belief.reshape(1,144), 
                "obstacles": np.array(obstacles, np.int32).reshape(1, 8), 
                "goal": np.array(goal, np.int32).reshape(1, 2)}
                )

        a = np.abs(obstacle_actions)/np.sum(obstacle_actions) 
        b = np.abs(belief_actions)/np.sum(belief_actions)
        print(a)
        # print(b)
        joint_actions = a*b
        joint_actions = joint_actions/np.sum(joint_actions)
        # print(joint_actions)
        # print(obstacle_actions)
        # return np.argmax(obstacle_actions)
        return np.argmax(joint_actions)
    
    def lookahead(self, belief, depth):
        # print(depth)
        if depth == 0:
            q_vals = [self.grid.compute_reward(belief, *self.grid.action_dict[a])[0] for a in range(9)]
            return max(q_vals), np.argmax(q_vals)
            # return 
        elif depth == self.max_depth:
            q_vals = []
            for a in range(9):
                reward, done = self.grid.compute_reward(belief, *self.grid.action_dict[a], obstacles=True)
                if done:
                    q_vals.append(reward)
                else:
                    next_state = self.grid.propagate_belief(belief, *self.grid.action_dict[a])
                    # print(self.lookahead(next_state, depth-1)[0])
                    q_vals.append(reward + self.gamma*self.lookahead(next_state, depth-1)[0])
            return max(q_vals), np.argmax(q_vals)
        else:
            q_vals = []
            for a in range(9):
                reward, done = self.grid.compute_reward(belief, *self.grid.action_dict[a])
                if done:
                    q_vals.append(reward)
                else:
                    next_state = self.grid.propagate_belief(belief, *self.grid.action_dict[a])
                    # print(self.lookahead(next_state, depth-1)[0])
                    q_vals.append(reward + self.gamma*self.lookahead(next_state, depth-1)[0])
            return max(q_vals), np.argmax(q_vals)
        

# error_history = []
# plt.figure()
# plt.title("Belief Error over time for 5 episodes")
# for i in range(5):
#     print(i)
#     tester = Test(Memory, Grid, Model, 50)
#     error_history.append(tester.run_episode())
#     plt.plot(tester.error_history)
# plt.title()
# belief_history = []

# belief = grid.map - 1
# belief = belief/np.sum(belief)

# for i in range(10):
#     belief_history.append(belief-0.5*grid.map)
#     belief = grid.propagate_belief(belief, -np.pi/2, step=True)
    
    
tester = Test(Memory, Grid, Model, 50)
tester.run_episode()

fig, ax = plt.subplots()#figsize=[10, 10])
# plt.title(factor+ " over time")
sns.heatmap(2*tester.grid.goal_grid + tester.episode_belief_history[0][0] + tester.episode_position_history[0][0] - 0.5*tester.grid.map, vmax=1, square=True)

def init():
      sns.heatmap(2*tester.grid.goal_grid + tester.episode_belief_history[0][0] + tester.episode_position_history[0][0] - 0.5*tester.grid.map, vmax=1, square=True, cbar=False)
      # plt.xlabel("Drone")
      # plt.ylabel("Drone")

def animate(i):
    sns.heatmap(2*tester.grid.goal_grid + tester.episode_belief_history[0][i] + tester.episode_position_history[0][i] - 0.5*tester.grid.map, vmax=1, square=True, cbar=False)
    # plt.xlabel("Drone")
    # plt.ylabel("Drone")

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(tester.episode_belief_history[0]), repeat = True)
plt.show()
# anim.save('attempt.mp4')
