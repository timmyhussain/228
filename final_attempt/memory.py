# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 18:39:17 2021

@author: user
"""
import numpy as np
from random import sample

class Memory():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        
    def add_sample(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            
    def get_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            return sample(self.buffer, len(self.buffer))
        else:
            return sample(self.buffer, batch_size)
    