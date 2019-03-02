
# coding: utf-8

# First I take a dataset where I have to build a network. The input layer has 4 nodes, and the output layer shall have 1 node. I have 3 records. Thus my input array shall be 3 * 4 in dimension.

# In[ ]:


X = [
    [1,1,1,0],
    [1,0,1,1],
    [0,1,0,1]
]
y = [[1.0],[1.0],[0.0]]


# I shall build a 3 layer network. With one hidden layer having 3 nodes and one output layer having 1 node. Activation function used shall be sigmoid.

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')


# In[ ]:


#We declare our network design initially.
Layer1_Nodes = len(X[0])
Layer2_Nodes = 3
Layer3_Nodes = len(y[0])


# In[ ]:


Weight_Layer1 = np.ones([Layer1_Nodes, Layer2_Nodes])
Bias_Layer1 = np.ones([1, Layer2_Nodes])
Weight_Layer2 = np.ones([Layer2_Nodes, Layer3_Nodes])
Bias_Layer2 = np.ones([1, Layer3_Nodes])


# In[ ]:


def activation(X):
    return sigmoid(X)
def sigmoid(X):
    return 1/(1+np.exp(-X))


# In[ ]:


def activation_derivative(X):
    return sigmoid_derivative(X)
def sigmoid_derivative(X):
    return X * (1-X)


# In[ ]:


def Error_Func(output, y):
    return y - output


# In[ ]:


epochs = 1000
learning_rate = 0.1
progress = []


# In[ ]:


for i in range(epochs):
    #Forward Propogation
    Layer1 = np.array(X)
    Buffer2 = np.dot(Layer1, Weight_Layer1) + Bias_Layer1
    Layer2 = activation(Buffer2)
    Buffer3 = np.dot(Layer2, Weight_Layer2) + Bias_Layer2
    Layer3 = activation(Buffer3)

    #Backpropogation
    Error_Layer3 = Error_Func(Layer3, y)

    Act_Slope_Layer3 = activation_derivative(Layer3)
    Delta_Layer3 = Act_Slope_Layer3 * Error_Layer3
    Error_Layer2 = np.dot(Delta_Layer3, Weight_Layer2.T)
    Weight_Layer2 = Weight_Layer2 + learning_rate * np.dot(Layer2.T, Delta_Layer3)
    Bias_Layer2 = Bias_Layer2 + learning_rate * np.sum(Delta_Layer3, axis = 0)

    Act_Slope_Layer2 = activation_derivative(Layer2)
    Delta_Layer2 = Act_Slope_Layer2 * Error_Layer2
    Error_Layer1 = np.dot(Delta_Layer2, Weight_Layer1.T) #Useless
    Weight_Layer1 = Weight_Layer1 + learning_rate * np.dot(Layer1.T, Delta_Layer2)
    Bias_Layer1 = Bias_Layer1 + learning_rate * np.sum(Delta_Layer2, axis = 0)
    
    progress.extend(sum(abs(Layer3-y)))


# In[ ]:


plt.plot(np.array(progress))


# In[ ]:


#Forward Propogation
Layer1 = np.array(X)
Buffer2 = np.dot(Layer1, Weight_Layer1) + Bias_Layer1
Layer2 = activation(Buffer2)
Buffer3 = np.dot(Layer2, Weight_Layer2) + Bias_Layer2
Layer3 = activation(Buffer3)


# In[ ]:


Layer3

