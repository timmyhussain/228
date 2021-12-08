# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 13:19:27 2021

@author: user
"""
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model

class Model:
    def __init__(self, num_layers, width, input_dim, output_dim, batch_size, learning_rate):
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_a = self._build_model(num_layers, width)
        self.model_b = self._build_model(num_layers, width)
        self.n_a = 1 #dict(zip(list(range(output_dim)), [1]*output_dim))
        self.n_b = 1 #dict(zip(list(range(output_dim)), [1]*output_dim))
    
    def _build_model(self, num_layers, width):
        '''Create model with num_layers hidden layes with width units
        Inputs
         - num_layers: integer, number of hidden layers
         - width: integer, number of units per layer
        Outputs
         - model: Keras model object
        '''
        
        #Input layer
        inputs = Input((self.input_dim,))
        
        #num_layers fully connected layers
        X = Dense(width, activation="relu")(inputs)
        for i in range(num_layers-1):
            X = Dense(width, activation='relu')(X)
        
        #Output layer with output_dim units
        outputs = Dense(self.output_dim, activation='linear')(X)
        
        #Adam optimizer 
        opt = Adam(lr=self.learning_rate)
        
        #model, with mean squared loss
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss = MeanSquaredError(), optimizer=opt)
        return model
    
    def _predict_state(self, model_a, state):
        ''' Outdated, but used intially for testing getting predictions from model
        Inputs
         - model_a: Boolean
         - state: state representation
        '''
        
        #reshaping necessary for Keras to deal with (1, num_states) instead of (num_states, )
        state = state.reshape((1, self.input_dim))
        
        if model_a:
            # self.n_a += 1
            return self.model_a.predict(state)
        else:
            # self.n_b += 1
            return self.model_b.predict(state)
    
    def _predict(self, model_a, states):
        ''' Obtain prediction from model specified by model_a on sequence of states
        Inputs
         - model_a: Boolean
         - state: array of states of size (num_examples, num_states) 
        Outputs
         - prediction: numpy array of shape (num_examples, num_actions)
        '''
        # state = state.reshape((1, self.input_dim))
        if model_a:
            # self.n_a += 1
            return self.model_a.predict(states)
        else:
            # self.n_b += 1
            return self.model_b.predict(states)
        
    def _get_n(self, model_a, action):
        '''Get n(s, a) for action for model specified by model_a
        Input
         - model_a: boolean, True if obtaining n for model A
         - action: integer, element of [0, 1, ..., n] where n is num_actions
        Outputs
         - n(s, a): integer, n(s, a) for model - the number of times an update 
         has happened for action a in model
        '''
        if model_a:
            return self.n_a #self.n_a[action]
        else:
            return self.n_b #self.n_b[action]
        
    def _set_n(self, model_a, action):
        '''Increments n(s, a) for action by 1 for model specified by model_a
        Input
         - model_a: boolean, True if obtaining n for model A
         - action: integer, element of [0, 1, ..., n] where n is num_actions
        '''
        if model_a:
            self.n_a += 1 #self.n_a[action] = self.n_a[action] + 1
        else:
            self.n_b += 1 #self.n_b[action] = self.n_b[action] + 1
        
    def save_model(self, path):
        '''Save model parameters to be loaded for testing/actual use'''
        self.model_a.save(os.path.join(path, 'trained_model_a.h5'))
        self.model_b.save(os.path.join(path, 'trained_model_b.h5'))
        
class TestModel(Model):
    '''Model subclass adjusted to facilitate testing and not training. 
    Removed training related lines and functions
    '''
    def __init__(self, input_dim, output_dim, load_path):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_a, self.model_b = self._load_models(load_path)
        
    def _load_models(self, path):
        '''Load model'''
        return load_model(os.path.join(path, 'trained_model_a.h5')), \
                   load_model(os.path.join(path, 'trained_model_b.h5'))
                   
    def _predict_state(self, model_a, state):
        ''' Obtain prediction from model specified by model_a on sequence of states
        Inputs
         - model_a: Boolean
         - state: array of states of size (num_examples, num_states) 
        Outputs
         - prediction: numpy array of shape (num_examples, num_actions)
        '''
        state = state.reshape((1, self.input_dim))
        if model_a:
            return self.model_a.predict(state)
        else:
            return self.model_b.predict(state)