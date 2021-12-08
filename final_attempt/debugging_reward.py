# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:56:11 2021

@author: user
"""
from helper import Grid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(8)
action_dict = dict(zip(list(range(9)), [(0, False)] + list(zip(np.linspace(-np.pi, np.pi, num=9, endpoint=True).tolist()[:-1], [True]*8))))

obstacles = np.load("map1-new/obstacles.npy", allow_pickle=True)
grid = Grid(10, 10, 10, -65.32, -53.20, obstacles)
belief, _, _ = grid.reset()

fig, ax = plt.subplots(1, 2, figsize=[25, 10])
sns.heatmap(grid.position_grid - 0.5*grid.map, ax = ax[0])
ax[0].set_title("Position")

# plt.figure()
sns.heatmap(grid.goal_grid.reshape(12,12) - 0.5*grid.map, ax=ax[1])
plt.title("Goal")

rewards = []
for a in range(9):
    reward = grid.compute_reward(belief, *action_dict[a], obstacles=True)[0]
    rewards.append(reward)
    print(a, reward)
plt.figure()
plt.scatter(list(range(9)), rewards)
    
