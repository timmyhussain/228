# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 07:45:06 2021

@author: user
"""
from helper import Net, Grid, parse_lidarData, propagate_belief, get_error
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

obstacles = np.load("map1-new/obstacles.npy", allow_pickle=True)
grid = Grid(10, 10, 10, -65.32, -53.20, obstacles)
# obstacles = [[]]
# grid.create_map()
# 

#%%
# grid.create_map(obstacles)

#%%
belief_history = []

belief = grid.map - 1
belief = belief/np.sum(belief)

for i in range(10):
    belief_history.append(belief-0.5*grid.map)
    belief = grid.propagate_belief(belief, -np.pi/2, step=True)
    
    


fig, ax = plt.subplots()#figsize=[10, 10])
# plt.title(factor+ " over time")
sns.heatmap(belief_history[0], vmax=1, square=True)

def init():
      sns.heatmap(belief_history[0], vmax=1, square=True, cbar=False)
      # plt.xlabel("Drone")
      # plt.ylabel("Drone")

def animate(i):
    sns.heatmap(belief_history[i], vmax=1, square=True, cbar=False)
    # plt.xlabel("Drone")
    # plt.ylabel("Drone")

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(belief_history), repeat = False)

