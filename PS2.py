# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:59:28 2019

@author: xu wang
"""
"""
replication of basic code
"""
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean

#Set up parameters
alpha = 0.35
beta = 0.99
delta = 0.025
sigma = 2

#Set up discretized state space
k_min = 0
k_max = 45
num_k = 1000 #% number of points in the grid for k

k = np.array([np.linspace(k_min, k_max, num_k)])
k_t = k.transpose()
k_mat = np.matlib.repmat(k_t, 1, num_k) #% this will be useful in a bit
k_mat_t = k_mat.transpose()

#Set up consumption and return function
#1st dim(rows): k today, 2nd dim (cols): k' chosen for tomorrow
cons = np.power(k_mat, alpha) + (1 - delta) * k_mat - k_mat_t 

ret = np.power(cons, 1-sigma)/(1-sigma)
#negative consumption is not possible -> make it irrelevant by assigning
#it very large negative utility
ret[cons<0]=-100000000000000

#Iteration
dis = 1 
tol = 1e-06 #% tolerance for stopping 
v_guess = np.zeros((1, num_k))

while dis > tol:
    #compute the utility value for all possible combinations of k and k':
    value_mat = ret + beta * np.matlib.repmat(v_guess, num_k, 1)
    
    #find the optimal k' for every k:
    vfn = np.array([value_mat.max(1)])
    pol_indx = np.array([np.argmax(value_mat,1)])
    #vfn = vfn.transpose()
    
    #what is the distance between current guess and value function
    dis = np.amax(abs(vfn - v_guess))
    
    #if distance is larger than tolerance, update current guess and
    #continue, otherwise exit the loop
    v_guess = vfn

g= k[0,pol_indx]
g_t = g.transpose()
vfn_t =vfn.transpose() 

plt.plot(k_t[:,0],g_t[:,0])
plt.plot(k_t[:,0],vfn_t[:,0])
plt.axis([0, 50, -65, -30])
plt.show()
"""

"""
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean

#Set up parameters
alpha = 0.35
beta = 0.99
delta = 0.025
sigma = 2
A_h=1.1
A_l=(1-0.76289*A_h)/0.23711
#Set up discretized state space
k_min = 0
k_max = 45
num_k = 1000 #% number of points in the grid for k

k = np.array([np.linspace(k_min, k_max, num_k)])
k_t = k.transpose()
k_mat = np.matlib.repmat(k_t, 1, num_k) #% this will be useful in a bit
k_mat_t = k_mat.transpose()

#Set up consumption and return function
#1st dim(rows): k today, 2nd dim (cols): k' chosen for tomorrow
cons = np.power(k_mat, alpha) + (1 - delta) * k_mat - k_mat_t 
cons_h = A_h*np.power(k_mat, alpha) + (1 - delta) * k_mat - k_mat_t
cons_l = A_l*np.power(k_mat, alpha) + (1 - delta) * k_mat - k_mat_t

ret = np.power(cons, 1-sigma)/(1-sigma)











