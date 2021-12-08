# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 12:52:26 2021

@author: user
"""
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_error(belief, true_x, true_y):
    error = 0
    for i in range(12):
        for j in range(12):
            dist = np.linalg.norm(np.array([i,j]) - np.array([true_x, true_y]))
            error += belief[i,j]*dist
    return error

def get_multiplier(x):
    return -x+1

def propagate_belief(belief, velocity, pose, map_data, T, distance=0, cell_size=10, eps=1e-3):
    # print(belief)
    c = 1/(cell_size/T) *velocity 

    # print(c)
    yaw = np.arctan2(pose.orientation.z_val, pose.orientation.w_val)*2
    angles = np.linspace(-np.pi, np.pi, num=9, endpoint=True)
    vectors = [[np.rint(np.cos(theta)), np.rint(np.sin(theta))] for theta in angles]
    direction = np.array(vectors[np.isclose(angles, yaw, atol=np.pi/8).nonzero()[0].min()])
    # print(direction)
    step = np.int32(math.floor(c))
    # print(step)
    # print(direction)
    new_belief_1 = get_multiplier(abs(c-step))*np.roll(belief, np.int32(step*direction), axis=(1,0))
    
    step = np.int32(math.ceil(c))
    new_belief_2 = get_multiplier(abs(c-step))* np.roll(belief, np.int32(step*direction), axis=(1,0))
    
    new_belief_3 = new_belief_1 + new_belief_2
    new_belief_3 = (1-map_data)*new_belief_3
    # print(sum(new_belief_3))
    # time.sleep(5)
    # print(new_belief_3)
    new_belief_3 = new_belief_3/np.sum(new_belief_3)
    return new_belief_3

def parse_lidarData(data):
    points = np.array(data.point_cloud, dtype="float32").reshape(-1, 3)
    return points[:, :2]

class Net:
    def __init__(self, path, name, map_size, activations=["tanh"]*3, noise=1e-3, load_model=False):
        self.name = name
        self.path = name+"/"
        # self.get_data(paths, map_size, noise)
        
        
        map_path = path
        self.map = np.load(map_path+"map_empty.npy".format(map_size=map_size))

        if not os.path.exists(self.path):
             os.makedirs(self.path)  
        self.checkpoint_dir = self.path+"checkpoints/" 
        self.checkpoint_path = self.path+"checkpoints/cp.ckpt"
        self.create_model(activations)
        if os.path.exists(self.checkpoint_dir):
            self.model.load_weights(self.checkpoint_path)
        pass
    
    def create_model(self, activations):
        act1, act2, act3 = activations
        
        #inputs
        lidar_input = keras.Input(shape=(720,), name="lidar")
        map_input = keras.Input(shape=(*self.map.shape,), name="map")
        
        lidar_features = layers.Dense(360, activation=act1)(lidar_input)
        lidar_features = layers.Dense(180, activation=act1)(lidar_features)
        
        lidar_features = layers.Dense(144, activation=act2)(lidar_features)
        lidar_features = layers.Dense(72, activation = act2)(lidar_features)
        
        #feature extraction for map
        map_features = layers.Conv1D(self.map.shape[0], 2, input_shape=(None, *self.map.shape))(map_input)
        map_features = layers.Conv1D(self.map.shape[0], 2, input_shape=(None, *self.map.shape))(map_features)
        map_features = layers.MaxPool1D(2)(map_features)
        map_features = layers.Flatten()(map_features)
        
        x = layers.concatenate([lidar_features, map_features])
        x = layers.Dense(132, activation = act3)(x)
        x = layers.Dropout(0.1)(x)
        
        #outputs
        position = layers.Dense(self.map.shape[0]*self.map.shape[1], name="position_belief", activation="softmax")(x)
        orientation = layers.Dense(self.map.shape[0]*self.map.shape[1], name="orientation_belief")(x)
        
        self.model = keras.Model(
                inputs = [lidar_input, map_input],
                outputs=[position, orientation])
        
        optimizer = keras.optimizers.Adam()

        loss = {"position_belief": keras.losses.MeanSquaredError(),
                "orientation_belief": keras.losses.MeanSquaredError()}#did ok with mse
        
        
        self.model.compile(optimizer=optimizer, loss=loss)#, metrics=metrics)
        
        keras.utils.plot_model(self.model, self.path+self.name+"_model.png", show_shapes=(True))
        pass
    
    def train(self, n_epochs, batch_size):
               
        cp_callback = keras.callbacks.ModelCheckpoint(self.checkpoint_path, save_weights_only=True)

        history = self.model.fit(
            {"lidar_input": self.train_examples, "map_input": self.map_examples} ,
            {"position_belief": self.train_position_labels, "orientation_belief": self.train_orientation_labels},
            epochs=n_epochs,
            batch_size=batch_size,
            callbacks=[cp_callback])
        return history
    
    def evaluate(self, inputs):
        lidar_input, map_input = inputs
        return self.model.predict(
            {"lidar_input": tf.convert_to_tensor(lidar_input.reshape(1, 720)), 
             "map_input": tf.convert_to_tensor(map_input.reshape(1, *self.map.shape))
             }
            )
    
class Grid:
    def __init__(self, N_x, N_y, cell_size, min_x, min_y, obstacles):
        self.nx = N_x + 10//cell_size
        self.ny = N_y + 10//cell_size
        self.min_x = min_x
        self.min_y = min_y
        self.cell_size = cell_size
        self.x_spacing = np.linspace(start=min_x, stop = min_x+(self.nx)*cell_size, num=self.nx+1)
        self.y_spacing = np.linspace(start=min_y, stop = min_y+(self.ny)*cell_size, num=self.ny+1)
        self.pseudo_grid = np.array(np.meshgrid(self.x_spacing, self.y_spacing)).T.reshape(self.nx+1, self.ny+1, 2)
        # self.pseudo_grid = np.flip(self.pseudo_grid, axis=)
        self.obstacles = obstacles
        self.create_map()
        self.set_goal()
        self.set_initial_position()
        pass
    
    def set_goal(self):
        # X: [-65.32, 44.12]
        # Y: [-53.20, 56.75]
        self.goal = np.array([np.random.choice([-55, 34], 1), np.random.choice([-43, 46], 1)]).flatten()
        # for n, g in enumerate([[-55, -43], [-55, 46], [34, -43], [34, 46]]):
        self.goal_ix = np.array(self.get_occupancy(*self.goal, tol=self.cell_size/2).nonzero()).reshape(1,2)[0]
        self.goal_grid = self.get_occupancy(*self.goal, tol=self.cell_size/2).reshape(12,12,1)
        self.goal_reached = False
        self.goal_attempts = 0
        print("Goal: {}\n".format(self.goal))
        
    def set_initial_position(self):
        candidate = np.array([np.random.uniform(-55, 34), np.random.uniform(-43, 46)])
        while np.isclose(candidate, self.obstacles, atol=5).all(axis=1).any():
            # print(candidate)
            candidate = np.array([np.random.uniform(-55, 34), np.random.uniform(-43, 46)])
        self.position = candidate
        # for n, g in enumerate([[-55, -43], [-55, 46], [34, -43], [34, 46]]):
        self.position_grid = self.get_occupancy(*self.position, tol=self.cell_size/2).reshape(12,12)
        self.position_ix = np.array(self.get_occupancy(*self.position, tol=self.cell_size/2).nonzero()).reshape(1,2)[0]
        
        
    def update_grid(self, x, y, tol):
        # print(self.pseudo_grid)
        ix = np.array(np.isclose(self.pseudo_grid, np.array([x, y]), atol=tol).all(axis=2).nonzero()).T
        # print(ix)
        for index in ix.tolist():
            self.map[index[1], index[0]] = 1 
        # pass
    
    def create_map(self):
        self.map = np.zeros((self.nx+1, self.ny+1))
        
        for obstacle in self.obstacles:
            self.update_grid(obstacle[0], obstacle[1], 5)
        pass
    
    def get_occupancy(self, x, y, tol):
        grid = np.zeros((self.nx+1, self.ny+1))
        ix = np.array(np.isclose(self.pseudo_grid, np.array([x, y]), atol=tol).all(axis=2).nonzero()).T
        # print(ix.tolist())
        
        for index in ix.tolist():
            # print(self.pseudo_grid[index[0], index[1]])
            grid[index[1], index[0]] = 1/len(ix.tolist())
        return grid
        
    def get_orientation(self, x, y, tol, orientation):
        grid = np.zeros((self.nx+1, self.ny+1))
        ix = np.array(np.isclose(self.pseudo_grid, np.array([x, y]), atol=tol).all(axis=2).nonzero()).T
        # print(ix.tolist())
        
        for index in ix.tolist():
            # print(self.pseudo_grid[index[0], index[1]])
            grid[index[1], index[0]] = orientation+2*np.pi
        return grid
    
    def propagate_belief(self, belief, yaw, step, distance=0, eps=1e-3):
    # print(belief)
        
    
        # print(c)
        # yaw = np.arctan2(pose.orientation.z_val, pose.orientation.w_val)*2
        
        angles = np.linspace(-np.pi, np.pi, num=9, endpoint=True)
        vectors = [[np.rint(np.cos(theta)), np.rint(np.sin(theta))] for theta in angles]
        direction = np.array(vectors[np.isclose(angles, yaw, atol=np.pi/8).nonzero()[0].min()])
        # print(direction)
        # print(step)
        # print(direction)
        if step:
            new_belief = np.roll(belief, np.int32(direction), axis=(1,0))
        else:
            new_belief = belief
        
        
        new_belief = (1-self.map)*new_belief
        if np.max(new_belief):
            new_belief = new_belief/np.sum(new_belief)
        return new_belief
    
    def get_error(belief, true_x, true_y):
        error = 0
        for i in range(12):
            for j in range(12):
                dist = np.linalg.norm(np.array([i,j]) - np.array([true_x, true_y]))
                error += belief[i,j]*dist
        return error
    
    def compute_reward(self, belief, yaw, step):
        new_belief = self.propagate_belief(belief, yaw, step)
        
        current_error = get_error(belief)
        if current_error < 3:
            destination_reward = 100
            done = True
        else:
            destination_reward = 0
            done = False
        
        if step:
            error_cost = get_error(new_belief) - current_error
        else:
            error_cost = -get_error(belief)
            
            
        new_position = self.propagate_belief(self.position_grid, yaw, step)
        if np.sum(new_position) == 0:
            collision_cost = -100
            done = True
        else:
            collision_cost= 0
            
            
        return error_cost + collision_cost + destination_reward, done
    
        
    def step(self, belief, yaw, step):
        next_state = self.propagate_belief(belief, yaw, step)
        self.position_grid = self.propagate_belief(self.position_grid, yaw, step)
        reward, done = self.compute_reward(belief, yaw, step)
        if done:
            self.reset()
            
        return next_state, reward, done
        
    def reset(self):
        self.create_map()
        self.set_goal()
        self.set_initial_position()