# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:53:49 2021

@author: user
"""
from simulation import Simulation, TestSimulation
from generator import TrafficGenerator
from model import Model, TestModel
from memory import Memory
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_test_path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


#%%
if __name__ == "__main__":
    config = import_train_configuration(config_file="training_settings.ini")

    model = Model(
        config['num_layers'], 
        config['width_layers'],
        config['num_states'],
        config['num_actions'],
        config['batch_size'],
        config['learning_rate'])

    memory = Memory(config['memory_size_max'])
    
    # viz = Visualization(
    #     config['models_path_name'], 
    #     dpi=96
    # )
        
    sim = Simulation(
        TrafficGen, 
        model,
        memory,
        config['gamma'],
        config['training_epochs'],
        sumo_cmd, config['max_steps'],
        config['green_duration'],
        config['yellow_duration'], 
        config['num_states'],
        config['num_actions']
        )
    
    
    
    episode = 0
    while episode < config['total_episodes']:
        print("\n ------ Episode: ", episode)
        epsilon = 1.0 - (episode / config['total_episodes'])
        simulation_time, avg_queue_length, total_wait_times = sim.run(episode, epsilon, baseline=False)
        episode += 1

    path = config['models_path_name']
    model.save_model(path)
    

    # viz.save_data_and_plot(data=sim._reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    # viz.save_data_and_plot(data=sim._cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    # viz.save_data_and_plot(data=sim._avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')
    
#%% Testing Simulation

    config = import_train_configuration(config_file="training_settings.ini")
    sumo_cmd = set_sumo(True, config['sumocfg_file_name'], config['max_steps'])
    TrafficGen = TrafficGenerator(config['max_steps'], config['n_cars_generated'])

    model = TestModel(
        config['num_states'],
        config['num_actions'],
        config['models_path_name']
        )
    
    testsim = TestSimulation(
        TrafficGen, 
        model,
        sumo_cmd, config['max_steps'],
        config['green_duration'],
        config['yellow_duration'], 
        config['num_states'],
        config['num_actions']
        )
    
    queues_episodes = []
    waits_episodes = []
    simulation_times_episodes = []
    episode = 5
    
    simulation_time, avg_queue_length, total_wait_times = testsim.run(episode, baseline=True)
    
    # while episode < config['total_episodes']:
    #     model.n_a = 1
    #     model.n_b = 1
    #     sim._cumulative_wait_store = []
    #     sim._total_wait_times = []
    #     sim._avg_queue_length_store = []
    #     epsilon = 1.0 - (episode / config['total_episodes'])
    #     simulation_time, avg_queue_length, total_wait_times = sim.run(episode, baseline=False)
    #     waits_episodes.append(total_wait_times)
    #     queues_episodes.append(avg_queue_length)
    #     simulation_times_episodes.append(simulation_time)
    #     episode += 1

    # with open(os.path.join("data", "high_density", str(4) + 'total_wait_times.txt'), "w") as file:
    #     for value in total_wait_times:
    #             file.write("%s\n" % value)
    # with open(os.path.join("data", "high_density", str(4) + 'avg_queue_length.txt'), "w") as file:
    #     for value in total_wait_times:
    #             file.write("%s\n" % value)
 #%%
    
    plt.plot(range(len(avg_queue_length)), total_wait_times)
    plt.title("Total wait over time")
    plt.ylabel("Total delay /t")
    
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

    matplotlib.rc('font', **font)
    
    # x = np.arange(5)
    # y = [i[-1] for i in waits_episodes]
    # p = np.poly1d(np.polyfit(x, y, 1))
    # # plt.figure(figsize=[9, 16])
    # plt.scatter(x, y)
    # plt.plot(x, p(np.arange(5)), 'r')
    # plt.xlabel("Training episodes")
    # plt.ylabel("Final Total Wait Time")
    # plt.figure()
    # plt.plot(range(len(avg_queue_length)), avg_queue_length)
    # plt.title("Average queue length over time")
    # plt.ylabel("Number of cars in queue")