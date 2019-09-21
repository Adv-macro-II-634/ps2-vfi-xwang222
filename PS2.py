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
import random

#create transition matix
PI=np.array([[0.977, 0.023],
    [0.074, 0.926]])

#Set up parameters
alpha = 0.35
beta = 0.99
delta = 0.025
sigma = 2
A_h=1.00000008
A_l=(1-0.76289*A_h)/0.23711
A_mat = [A_h,A_l]
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

ret_h = np.power(cons_h, 1-sigma)/(1-sigma)
ret_l = np.power(cons_l, 1-sigma)/(1-sigma)

ret_h[cons_h<0]=-100000000000000
ret_l[cons_l<0]=-100000000000000

#Iteration
dis = 1
#dis_l = 1
 
Prob_h =np.array([PI.transpose()[:,0]])
Prob_l =np.array([PI.transpose()[:,1]])


tol = 1e-06 #% tolerance for stopping 
v_guess = np.zeros((2, num_k))

while dis > tol:
    #compute the utility value for all possible combinations of k and k':
    value_mat_h = ret_h + beta * np.matlib.repmat(np.matmul(Prob_h,v_guess), num_k, 1)
    value_mat_l = ret_l + beta * np.matlib.repmat(np.matmul(Prob_l,v_guess), num_k, 1)
    
    #find the optimal k' for every k:
    vfn = np.array([value_mat_h.max(1),value_mat_l.max(1)])
    pol_indx = np.array([np.argmax(value_mat_h,1),np.argmax(value_mat_l,1)])
    
    #vfn = vfn.transpose()
    
    #what is the distance between current guess and value function
    dis = np.amax(abs(vfn - v_guess))
    
    #if distance is larger than tolerance, update current guess and
    #continue, otherwise exit the loop
    v_guess = vfn



g_h= np.array([k[0,pol_indx[0,:]]])
g_l= np.array([k[0,pol_indx[1,:]]])

Saving_h =g_h - (1 - delta) * k
Saving_l =g_l - (1 - delta) * k

g_h_t = g_h.transpose()
g_l_t = g_l.transpose()


vfn_t =vfn.transpose() 

#plot policy function
plt.plot(k_t[:,0],g_h_t[:,0])
plt.plot(k_t[:,0],g_l_t[:,0])
plt.axis([0, 50, 0, 50])
plt.show()

#plot value function
plt.plot(k_t[:,0],vfn_t[:,0])
plt.plot(k_t[:,0],vfn_t[:,1])
plt.axis([0, 50, -100, -30])
plt.show()

#plot saving
plt.plot(k_t[:,0],Saving_h.transpose())
plt.plot(k_t[:,0],Saving_l.transpose())
plt.axis([0, 50, -1, 2])
plt.show()

#question 4
T = 5000
A_h=1.0008

Trans = np.matrix([[0.977, 0.023], [0.074, 0.926]])
np.random.seed(1)
A_sim_seq = []
for i in range(1,T):
    A_t = np.random.random_sample()
    A_sim_seq.append(A_t)    

#suppose A_0 = high
A_state = [0]
for i in range(1,T):
    if A_state[i-1] == 0:
        if A_sim_seq[i-1]<=Trans[0,0]:
            A_state_t = 0
        else:
            A_state_t = 1
    elif   A_state[i-1] == 1:
        if A_sim_seq[i-1]<=Trans[1,1]:
            A_state_t = 1
        else:
            A_state_t = 0
    A_state.append(A_state_t)

# simulate state
A_state_rel = []
for i in A_state:
    A_temp = A_mat[i]
    A_state_rel.append(A_temp)

# find K_h
K_index = [60]
for i in range(1,T):
    K_temp = pol_indx[A_state[i-1], K_index[i-1]]
    K_index.append(K_temp)
    
Y_sim = np.array([A_state_rel*np.power(k[0,K_index], alpha)])
Y_list = list(Y_sim)
Y_sim_f = Y_sim[:,100:5000]
np.std(Y_sim_f)

    