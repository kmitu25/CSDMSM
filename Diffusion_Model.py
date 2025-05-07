#!/usr/bin/env python
# coding: utf-8

# # A 1D Diffusion Model

# Here we develop a one-dimensional model of diffusion.
# It assumes a constant diffusivity
# It uses a regular grid
# It has a step function for an initial condition.
# It has  fixed boundary conditions.

# Here is the diffusion equation:

# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$
# 
# 

# Here is the  discretized version of the diffusion equation we will solve with our model

# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2} (C^t_{x+1} - 2C^t_x + C^t_{x-1}) $$

# This is the FTCS scheme as described by Slingerland and kump (2011)

# We will use two libraries, Numpy (for arrays) and matplotlib (for plotting), that are not part of the core Python distribution.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# Start by setting two fixed model parameters, the diffusivity and the size of the model domain.

# In[2]:


D=100
Lx=300


# Next, set up the model using a numpy array

# In[3]:


dx=0.5
x=np.arange(start=0, stop=Lx, step=dx)
nx=len(x)


# In[4]:


nx


# In[5]:


x


# In[6]:


#x[0]


# In[7]:


#x[1]


# To get last element its negative -1 and for the second last it is -2

# In[8]:


#x[-1]


# In[9]:


#x[-2]


# To get first 5 elements

# In[10]:


#x[0:4]


# Python is inclusive on the left side but exclusive on the right side, so if I want to get the first 5, I need to use o:5

# In[11]:


#x[0:5]


# In[12]:


#x[-5:]


# Now back to model. Set the initial conditions for the model.
# The cake 'C' is a step function with a high value of the left, a low value on the right, and a step at the center of the domain.

# In[13]:


C = np.zeros_like(x)
C_left = 500
C_right = 0
C[x<=Lx/2] = C_left
C[x>Lx/2] = C_right


# In[14]:


C


# Plot the initial profile

# In[15]:


plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("initial Profile")


# Set the number of the time steps in the model.
# calculate a stable time step using a stability criterion.

# In[16]:


nt=5000
dt=0.5*dx**2/D


# In[17]:


dt


# Loop over the time steps of the model. solving the difusion equation using the FTC scheme shown above
# Note the use of array operations on the variables 'C'. the boundary conditions remain fixed in each time step

# In[18]:


for t in range(0, nt):
	C[1:-1] += D * dt / dx ** 2 * (C[:-2] - 2*C[1:-1] + C[2:])


# In[19]:


#z=list(range(5))


# In[20]:


#z


# In[21]:


#z[1:-1]


# In[22]:


#z[:-2]


# In[23]:


#z[2:]


# plot the result

# In[24]:


plt.plot(x, C, "b")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Final Profile")


# In[ ]:




