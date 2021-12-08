# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 17:46:31 2021

@author: user
"""

import tensorflow as tf
import setup_path
import airsim
import numpy as np
import math
import time
import pandas as pd
import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from helper import Net, Grid, parse_lidarData, propagate_belief, get_error

class AirSimLidarCarEnv(AirSimEnv):
    def __init__(self, lidar_range=10):
        self.lidar_range = lidar_range
        self.observation_space = spaces.Box(0, self.lidar_range, shape=(720,1), dtype=np.float32)
        self.car = airsim.CarClient()#ip=ip_address)
        self.action_space = spaces.Discrete(6)
        
        map_size=10
        map_path = "map1-new/"        
        self.net = Net(map_path, "NN2", map_size)
        
        belief = self.net.map - 1
        self.belief = belief/np.sum(belief)
        self.T = 0.5
        self.speed = 0
        
        self.old_res = np.zeros(self.net.map.shape)
        self.old_error = False
        self.grid = Grid(10, 10, 10, -65.32, -53.20)
        self.action_history = np.array([0]*5, np.int32)
        self.history_len = 5
        self.car_controls = airsim.CarControls()
        self.goal = np.array([34.0, -43.0], np.float32)
        self.goal_ix = np.array(self.grid.get_occupancy(*self.goal, tol=self.grid.cell_size/2).nonzero()).reshape(1,2)[0]
        
    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.01)
        
    def __del__(self):
        self.car.reset()
        
    def _do_action(self, action):
        self.car_controls.brake = 0
        if self.speed <10:
            self.car_controls.throttle = 1
        else:
            self.car_controls.throttle = 0

        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1:
            self.car_controls.steering = 0
        elif action == 2:
            self.car_controls.steering = 0.5
        elif action == 3:
            self.car_controls.steering = -0.5
        elif action == 4:
            self.car_controls.steering = 0.25
        else:
            self.car_controls.steering = -0.25

        self.car.setCarControls(self.car_controls)
        self.action_history = (self.action_history+ [action])[-self.history_len:]
        time.sleep(1)
        
    def get_and_process_lidar_data(self):
        lidarData = self.car.getLidarData("LidarSensor1", "Car1")
        data = parse_lidarData(lidarData)
        
        arr_1 = -1*np.ones((720, 1)) #for belief network
        arr_2 = self.lidar_range*np.ones((720, 1)) #for policy network
        
        # df = pd.DataFrame(lidar_data[i])
        if len(data):
            df = pd.DataFrame()
            df["angles"] = np.rad2deg(np.arctan2(data[:, 0], data[:, 1]))
            
            df["distance"] = np.linalg.norm(data, axis=1)
            angles = np.linspace(-179.5, 180, 720, endpoint=True)
            f = lambda x: np.where(angles>x)[0].min()
            df["angles"] = df["angles"].apply(f)
            df = df.groupby("angles").mean()
            
            arr_1[df.index] = df.values
            arr_2[df.index] = df.values
        
        return arr_1, arr_2
            
    def update_belief(self, arr):
        inputs = [arr, self.net.map]
        res  = self.net.evaluate(inputs)
        res = res[0].reshape(self.net.map.shape)
        
        self.speed = self.state.speed
        
        self.belief = propagate_belief(self.belief, self.speed, self.pose, self.net.map, self.T)
        self.belief = 0.2*self.belief + 0.8*self.belief*np.absolute(res - 0.8*self.old_res) #works ok
        self.belief = self.belief/np.sum(self.belief)
        self.old_res = res.copy()
    
        time.sleep(self.T)
            
    def _get_obs(self):
        belief_arr, policy_arr = self.get_and_process_lidar_data()
        self.state = self.car.getCarState("Car1")
        self.pose = self.car.simGetVehiclePose("Car1")
        self.update_belief(belief_arr)

        # self.state["prev_pose"] = self.state["pose"]
        # self.state["pose"] = self.car_state.kinematics_estimated
        # self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return policy_arr
    
    def _compute_reward(self):
        
        error = get_error(self.belief, *self.goal_ix)
        has_collided = self.car.simGetCollisionInfo().has_collided
        # certainty_goal_reward = 10000*self.belief.max()/len(self.belief.nonzero()[0])*-error
        if self.old_error:
            certainty_goal_reward = 10*(self.old_error - error)
        else:
            self.old_error = error
            certainty_goal_reward = 0
            # certainty_goal_reward = 10*(self.old_error - error)
        # print(certainty_goal_reward)
        collision_cost = -100*has_collided
        done=0
        destination_reward=0
        if error<1:
            done=1
            destination_reward=50
        if has_collided:
            done=1
        return certainty_goal_reward + collision_cost+destination_reward, done
            
    
    def step(self, action):
        self._do_action(action)
        policy_arr = self._get_obs()
        reward, done = self._compute_reward()

        belief = self.belief.reshape(12,12,1)
        map_data = self.net.map.reshape(12,12,1)
        stacked = np.dstack([belief, map_data])
        
        return  np.array(stacked, np.float32), \
                np.array(policy_arr.flatten(), np.float32), \
                np.array(self.action_history, np.int32), \
                np.array(self.goal_ix, np.int32), \
                np.array(reward, np.float32), \
                np.array(done, np.int32)#, {} #self.state

    def reset(self):
        self._setup_car()
        self._do_action(1)
        self.old_error = False
        belief = self.net.map - 1
        self.belief = belief/np.sum(belief)
        self.action_history = [0]*5
        policy_arr = self._get_obs()
        
        belief = self.belief.reshape(12,12,1)
        map_data = self.net.map.reshape(12,12,1)
        stacked = np.dstack([belief, map_data])
        

        # return np.hstack((self.belief.flatten(), self.net.map.flatten(), policy_arr.flatten(), self.action_history))
        return  np.array(stacked, np.float32), \
                np.array(policy_arr.flatten(), np.float32), \
                np.array(self.action_history, np.int32), \
                np.array(self.goal_ix, np.int32)
    
