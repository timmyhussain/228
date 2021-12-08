# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:45:26 2021

@author: user
"""
import numpy as np
from tensorflow.keras import backend as K
import traci
import timeit
from enum import Enum

# self._Model = Model
# self._Memory = Memory

class LightPhase(Enum):
    PHASE_NS_GREEN = 0
    PHASE_NSL_GREEN = 2
    PHASE_EW_GREEN = 4
    PHASE_EWL_GREEN = 6
    
class Simulation:
    def __init__(self, TrafficGen, Model, Memory, gamma, training_epochs, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        '''Initialisation of Simulation object
        Inputs
         - TrafficGen: TrafficGen object, for route generation
         - Model: Model object, for training and polling model
         - Memory: Memory object, for storing and retrieving experiences
         - training_epochs: integer, number of training epochs to be run per simulation
         - sumo_cmd: list, parameters to establish connection with SUMO environment
         - max_steps: integer, number of steps per simulation
         - green_duration: integer, duration of green light
         - yellow_duration: integer, duration of yellow light
         - num_states: integer, size of array used to represent simulation state
         - num_actions: integer, size of action space
        '''
        self._TrafficGen = TrafficGen
        self._Model = Model
        self._Memory = Memory
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._total_wait_times = []
        self._avg_queue_length_store = []
        self._action_mapping = dict(zip([0, 1, 2, 3], [0, 2, 4, 6]))
        self._training_epochs = training_epochs
    
    def run(self, episode, epsilon, baseline):
        '''Main function that runs a simulation of max_steps 
        Inputs
         - episode: integer, acts as a seed for repeatable training
         - epsilon: float, for epsilon greedy exploration
         - baseline: boolean, True if running baseline policy
        Outputs
         - simulation_time: duration of simulation
         - avg_queue_length_store: average number of cars in queue for simulation
         - total_wait_times: summed total of all cars' wait time in simulation
         '''
        #start simulation with right seed
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd, label="new3"+"{:.1f}".format(np.random.randn()))
        print("Simulating...") #:>
        
        #initialize simulation parameters
        self._step = 0
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._waiting_times = {}
        old_state = -1
        old_action = -1
        
        #for each step in the simulation
        while self._step < self._max_steps:
            #get current state 
            current_state = self._get_state()
            current_total_wait = self._collect_waiting_times()
            
            #update and poll model_a 50% of the time
            model_a = np.random.random() < 1
            action = self._choose_action(current_state, old_action, epsilon, model_a = model_a, baseline=baseline)


            #we only trigger a yellow light if we aren't doing the same action again
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)  
            
            self._set_green_phase(action)
            self._simulate(self._green_duration)
            
            #store current state, action, reward, next state as experience tuple
            new_state = self._get_state()
            reward = -(self._collect_waiting_times() - current_total_wait)
            current_experience = (current_state, action, reward, new_state)
            self._Memory.add_sample(current_experience)
            
            
            old_action = action
            
            
            #train on example
            # if np.random.random() < 0.7:
            # self._train(model_a, current_experience)
            
            #store negative reward
            if reward < 0:
                self._sum_neg_reward += reward
            # self._avg_queue_length_store.append(self._get_queue_length())
            # self._total_wait_times.append(self._sum_waiting_time)
            
        #store metrics from entire simulation episode
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps) 
        
        #close simulation
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 2)
        print("\n Total reward: ", self._sum_neg_reward)
        
        print("\n Training...")
        
        #batch train for training_epochs number of times
        if not baseline:
            for j in range(self._training_epochs):
                model_a = np.random.random() < 1
                self._replay(model_a)
        
        #return simulation metrics
        return simulation_time, self._avg_queue_length_store, self._total_wait_times
            
            
    def _choose_action(self, state, old_action, epsilon, model_a = True, baseline=True):
        '''Returns action in [0, 1, 2, ..., n] where n is num_actions-1
        Inputs
         - state: numpy array of shape (num_states, 1), discretized representation of simulation state
         - old_action: integer, previous action used for determining next action in periodic policy
         - epsilon: float, for epsilon-greedy policy
         - model_a: boolean, True if polling model A
         - baseline: boolean, True if executing baseline policy
        Outputs
         - action: integer, as described above
        '''
        #simple periodic action sequence
        if baseline:
            action = (old_action + 1) % 4
            return action
        #execute epsilon-greedy policy
        else:
            #random exploration with probability epsilon
            if np.random.random() < epsilon:
                # print("Here")
                return np.random.randint(0, self._num_actions)
            #poll model (a or b depending on value of boolean model_a)
            else:
                # if model_a:
                return np.argmax(self._Model._predict_state(model_a, state=state))
                # else:
                #     return np.argmax(self._Model._predict_state(model_a=False, state=state))
            # return np.argmax()

    def _train(self, model_a, current_experience):
        '''Trains model specified by model_a on singe experience
        Inputs
         - model_a: boolean, True if training model A
         - current_experience: tuple, (current_state, action, reward, next_state)
        '''
        
        #decompose current_experience tuple (s, a, r, s')
        current_state, action, reward, next_state = current_experience
        #optimal next action from model being updated, a*
        a_star = np.argmax(self._Model._predict_state(model_a, next_state))
        #learning rate as defined in Double Q-Leaning (2010) alpha = 1/n
        alpha = 1/self._Model._get_n(model_a, action)
        #increment n for model being updated
        self._Model._set_n(model_a, action)
        
        #current Q(s, .) for model being updated
        q_s = self._Model._predict_state(model_a, current_state)
        #current Q(s, a) for model being updated
        q_s_a = self._Model._predict_state(model_a, current_state)[0, action]
        #current Q(s', a*) for model not being updated
        q_s_astar = self._Model._predict_state(not model_a, next_state)[0, a_star]
        
        # q_s[0, action] = q_s_a + alpha*(reward + self._gamma*q_s_astar - q_s_a)
        
        #target Q(s, a) for model being updated 
        q_s[0, action] = (reward + self._gamma*q_s_astar)
        
        #set learning rate according to alpha determined above
        #train model on single experience
        if model_a:
            K.set_value(self._Model.model_a.optimizer.learning_rate, alpha)
            self._Model.model_a.fit(current_state.reshape(1, len(current_state)),
                                    q_s, verbose=0)
        else:
            K.set_value(self._Model.model_a.optimizer.learning_rate, alpha)
            self._Model.model_b.fit(current_state.reshape(1, len(current_state)),
                        q_s, verbose=0)
        
    
    def _replay(self, model_a):
        '''Same idea as train but for a batch of size batch_size
        Input
         - model_a: boolean, True if training model A
        '''
        #Pull batch of size batch_size from Memory module
        batch = self._Memory.get_batch(self._Model.batch_size)
        
        #initialize lists to store data from batch
        current_states = []
        actions = []
        rewards = []
        next_states = []
        # alpha = 1/self._Model._get_n(model_a)
        
        #split experiences in batch and append to appropriate lists
        for exp in batch:
            current_states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            next_states.append(exp[3])
            
        #turn current and next states into numpy arrays of size (batch_size, num_states) 
        current_states = np.vstack(current_states)
        next_states = np.vstack(next_states)
        
        #actions predicted by model for next state
        a_star = np.argmax(self._Model._predict(model_a, next_states), axis=1)
        
        #Q values predicted by model for current state (the argmax of these are already in actions list)
        q_s = self._Model._predict(model_a, current_states)
        
        #Q values predicted by not model being updated for next state
        q_s_astar = self._Model._predict(model_a, next_states)
        
        #creating target array; basically batch version of function in train
        for ex in range(self._Model.batch_size):
            # n = self._Model._get_n(model_a, actions[ex])
            # alpha = 1/n
            # q_s[ex, actions[ex]] = q_s[ex, actions[ex]] + alpha*(rewards[ex] + \
            #                      self._gamma*q_s_astar[ex, a_star[ex]] - \
            #                      q_s[ex, actions[ex]])
            q_s[ex, actions[ex]] = rewards[ex] + \
                                 self._gamma*q_s_astar[ex, a_star[ex]]
           
            #at the end of this we will have trained the model on batch_size examples
            #so increment n for model being updated batch_size times by calling this once within each loop 
            
            # self._Model._set_n(model_a, actions[ex])
        
        # n = self._Model._get_n(model_a, actions[ex])
        
        #Double Q-Learning (2010)
        
        # alpha = 4/n
        if model_a:
            # K.set_value(self._Model.model_a.optimizer.learning_rate, alpha**0.8)
            self._Model.model_a.fit(current_states, q_s, verbose=0)
        else:
            # K.set_value(self._Model.model_b.optimizer.learning_rate, alpha**0.8)
            self._Model.model_b.fit(current_states, q_s, verbose=0)
        
    def _set_yellow_phase(self, old_action):
        '''Set appropriate lights yellow'''
        yellow_phase_code = old_action * 2 +1
        traci.trafficlight.setPhase("TL", yellow_phase_code)
        
    def _set_green_phase(self, action):
        '''Set appropriate lights green as determined by action'''
        
        traci.trafficlight.setPhase("TL", self._action_mapping[action])
        
    def _simulate(self, steps_todo):
        '''Take steps_todo steps in simulation
        Inputs
         - steps_todo: integer, number of simulation steps to take
        '''
        #we can only take as many steps as we have left in the simulation 
        steps_todo = min(steps_todo, self._max_steps - self._step)
        
        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            
            #queue length and waiting times are analagous because one simulation step is equal to one second
            #a car in a queue for 1 step means a waiting time of 1 second
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length
            
            
    def _get_queue_length(self):
        '''Get total queue length, summed total of queue on each road
        Outputs
         - queue_length: integer, total queue length
        '''
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _collect_waiting_times(self):
        '''Get total waiting times for cars currently in queues
        Outputs
         - tot_waiting_times: float, total waiting times'''
        incoming_roads = "E2TL N2TL W2TL S2TL".split()
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            if traci.vehicle.getRoadID(car_id) in incoming_roads:
                self._waiting_times[car_id] = traci.vehicle.getAccumulatedWaitingTime(car_id)
                # print(self._waiting_times[car_id])
            elif car_id in self._waiting_times:
                del self._waiting_times[car_id]
        tot_waiting_times = sum(self._waiting_times.values())
        # print(tot_waiting_times)
        return tot_waiting_times
    
    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located 
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state

class TestSimulation(Simulation):
    '''Simulation subclass adjusted to facilitate testing and not training. 
    Removed training related lines and functions
    '''
    def __init__(self, TrafficGen, Model, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self._TrafficGen = TrafficGen
        self._Model = Model
        # self._Memory = Memory
        # self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._total_wait_times = []
        self._avg_queue_length_store = []
        self._action_mapping = dict(zip([0, 1, 2, 3], [0, 2, 4, 6]))
        # self._training_epochs = training_epochs
        
    def run(self, episode, baseline):
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd, label="new2"+"{:.3f}".format(np.random.randn()))
        print("Simulating...") #:>
        
        self._step = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._waiting_times = {}
        old_action = -1
        
        
        while self._step < self._max_steps:
            current_state = self._get_state()
            # print(current_state.shape)
            
            model_a = np.random.random() < 1
            action = self._choose_action(current_state, old_action, model_a = model_a, baseline=baseline)


            #we only trigger a yellow light if we aren't doing the same action again
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)  
            
            self._set_green_phase(action)
            self._simulate(self._green_duration)
            
            
            # old_state = current_state
            old_action = action
            # old_total_wait = current_total_wait
            
            
            self._avg_queue_length_store.append(self._get_queue_length())
            self._total_wait_times.append(self._sum_waiting_time)
            
            
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 2)
        print("total wait time: ", self._sum_waiting_time)
        
        
        return simulation_time, self._avg_queue_length_store, self._total_wait_times
    
    def _choose_action(self, state, old_action, model_a = True, baseline=True):
        #simple periodic action sequence
        if baseline:
            action = (old_action + 1) % 4
            return action
        else:
            return np.argmax(self._Model._predict_state(model_a, state=state))

    