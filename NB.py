#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 01:07:34 2019

@author: tylerliu
"""

import numpy as np


class NB:
    def __init__(self, x, y):
        x = np.array(x.toarray())
        self.x = x
        self.y = y.reshape(y.shape[0],1)
        self.y_dict = dict()
        
        index = 0
        for y in self.y:
            if y[0] in self.y_dict:
                continue
            else:
               self.y_dict[y[0]] = index
               index += 1 
#        print(self.y_dict)
# y_dict: this parameter is of type dict. The possible format would be like this
# y_dict = {'funny' =1, 'nba' =2, 'movies' =3 .....} The 1 2 3 .... 20 are the indices
# We use to store the frequency in data
               
        self.data_table = np.full((len(self.y_dict), self.x.shape[1]+1) , 0, dtype='f')
        self.bernoulli
        self.counting()
#        print(self.data_table)
# After counting(), data_table would be like in this form (Example for the First ROW)
# [100, 21, 0 , 99 ............... 1000] 
# We've seen the index for 'funny' subreddit is 1. Then the first colomn stores the info
# about the 'funny' subreddit. It means that in our input data. We've seen class 'funny'
# 1000 times (last colomn). Featuer 1 appears 100 times, feature 2 times.
        self.frequency()
#        print(self.data_table)
# After requency(), the frequency for the appearence of each feature is calculated.
# For the new data_table, the format could be like this (Example for the First ROW)
# [0.1007983 = ||101/1002||, 0.021956, ..........., 0.05]
# The last colomn means the class 'funny' takes 5% of our training set 
# Laplace smoothing is also done.
        
    def bernoulli(self):
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                if self.x[i,j]>0:
                    self.x[i,j] = 1

        
    def counting(self):
        for i in range(self.x.shape[0]):
            index = self.y_dict[self.y[i,0]]
            arr = np.random.rand(1,self.x.shape[1]+1)
            arr[:,:-1] = self.x[i,:]
            arr[:, -1] = 1
            self.data_table[index, :] = self.data_table[index, :] + arr

#    def frequency(self):
#        total = sum(self.data_table[:,-1])
#        for i in range(len(self.y_dict)):
#            for j in range(self.x.shape[1]):
#                self.data_table[i,j] = (self.data_table[i,j])/(self.data_table[i,-1])
#            self.data_table[i,-1] = self.data_table[i,-1] / total
            
    def frequency(self):
        total = sum(self.data_table[:,-1])
        for i in range(len(self.y_dict)):
            for j in range(self.x.shape[1]):
                self.data_table[i,j] = (self.data_table[i,j]+1)/(self.data_table[i,-1]+2)
            self.data_table[i,-1] = self.data_table[i,-1] / total
          
        
    def predict(self, data_input):     

        print("Input shape is: ", data_input.shape)
        result = np.full( (data_input.shape[0],1), 'Hello World' )
        for i in range(data_input.shape[0]):
            max_prob = -1
            likely_class = ''  
            for j in self.y_dict.keys():
                index = self.y_dict[j]
                prob = 0
                for k in range(self.x.shape[1]):
                    prob += data_input[i,k] * np.math.log(self.data_table[index,k]) 
                    + (1 - data_input[i,k]) * np.math.log(1 - self.data_table[index,k])
                prob = prob + np.math.log(self.data_table[index,-1])
## Estimate the probability based on the formula log(P(class = ?)) * sigma->log(P(Xi=??|class=?))
                if np.math.exp(prob) > max_prob:
                    max_prob =np.math.exp(prob)
                    likely_class = j
            print(likely_class)
            result[i,0] = likely_class
# Store the most likely class in result
        return result