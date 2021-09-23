#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle  


# In[113]:


df = pd.read_excel('SFEW.xlsx').dropna()
df.columns = ['Image', 'label', 'LPQ 1', 'LPQ 2', 'LPQ 3','LPQ 4','LPQ 5','PHOG 1','PHOG 2','PHOG 3','PHOG 4','PHOG 5']


# In[114]:


#This is the part of compression
hidden_size = 9 # size of hidden layer of neurons
data_size = 11
learning_rate = 1e-1
data = df.drop(["Image"],axis = 1)
train_data,test_data = train_test_split(data,test_size = 0.1)
print(data.corr()["label"])
train_data = torch.tensor(train_data.values).float()
test_data = torch.tensor(test_data.values).float()


# In[115]:


#The compression network
class Compression(torch.nn.Module):
    def __init__(self):
        super(Compression, self).__init__()
        self.line1 = nn.Linear(data_size, hidden_size)
        self.line2 = nn.Linear(hidden_size, data_size)

    def forward(self, x):
        x = F.sigmoid(self.line1(x))
        tmp = x
        x = self.line2(x)
        return x,tmp


# In[116]:


#train compression network
Comp = Compression()
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(Comp.parameters(), lr=learning_rate)
for epoch in range(1000):
    out,activation = Comp(train_data)
    loss = loss_func(out, train_data)  
    optimizer.zero_grad()  
    loss.backward()
    optimizer.step()
    
print(loss)
tmp,tmp1 = Comp(test_data)
print("Test Loss is",loss_func(test_data,tmp))


# In[117]:


#function which is used to combine two hidden neurons
def hiddenCombine(Net,i,j):
    para = []
    tmp=Net.line1.weight
    tmp[i] = tmp[i]+tmp[j]
    tmp = tmp[torch.arange(tmp.size(0))!=j]
    para.append(tmp)
    tmp=Net.line1.bias
    tmp[i] = tmp[i]+tmp[j]
    tmp = tmp[torch.arange(tmp.size(0))!=j]
    para.append(tmp)
    tmp=Net.line2.weight
    tmp[:,i] = tmp[:,i]+tmp[:,j]
    tmp = tmp[:,torch.arange(tmp.size(1))!=j]
    para.append(tmp)
    tmp=Net.line2.bias
    para.append(tmp)
    return para
#function which is used to delete a hidden neurons
def hiddenDelete(Net,i):
    para = []
    tmp=Net.line1.weight
    tmp = tmp[torch.arange(tmp.size(0))!=i]
    para.append(tmp)
    tmp=Net.line1.bias
    tmp = tmp[torch.arange(tmp.size(0))!=i]
    para.append(tmp)
    tmp=Net.line2.weight
    tmp = tmp[:,torch.arange(tmp.size(1))!=i]
    para.append(tmp)
    tmp=Net.line2.bias
    para.append(tmp)
    return para


# In[118]:


vectors = activation.detach().numpy()
L1 = []
Min = []
#try to combine every two pairs, choose most similar pair to combine
#and also delete the zero output units
#test when we dropout different number of units
print(loss_func(tmp, test_data))
for epoches in range(8):
    out,activation = Comp(train_data)
    loss = loss_func(out, train_data)
    L1.append([hidden_size,loss])
    print("The loss of",hidden_size,"units is",loss)
    minAngel = 181
    index = []
    zeroFlag = False
    #calculate the vectors angle and find the minimum angle
    for i in range(hidden_size):
        for j in range(hidden_size):
            if i != j and vectors[:,i].all()!=0 and vectors[:,j].all()!=0:
                cos = torch.cosine_similarity(activation[:,i].view(1,train_data.shape[0]),activation[:,j].view(1,train_data.shape[0])) 
                ang = np.arccos(cos.detach().numpy())*180/np.pi
                if ang < minAngel:
                    minAngel = ang
                    index = np.array([i,j])
            elif vectors[:,i].all()==0:
                zeroFlag = True
                index = np.array([i])
                break
            elif vectors[:,j].all()==0:
                zeroFlag = True
                index = np.array([j])
                break
    #if there is no 0 output
    if not zeroFlag:
        para = hiddenCombine(Comp,index[0],index[1])
        hidden_size = hidden_size - 1
        Comp = Compression()
        Comp.line1.weight = torch.nn.Parameter(para[0])
        Comp.line1.bias = torch.nn.Parameter(para[1])
        Comp.line2.weight = torch.nn.Parameter(para[2])
        Comp.line2.bias = torch.nn.Parameter(para[3])
        Min.append([hidden_size+1,minAngel])
    #if there is 0 output then dropout the unit
    else:
        para = hiddenDelete(Comp,index[0])
        hidden_size = hidden_size - 1
        Comp = Compression()
        Comp.line1.weight = torch.nn.Parameter(para[0])
        Comp.line1.bias = torch.nn.Parameter(para[1])
        Comp.line2.weight = torch.nn.Parameter(para[2])
        Comp.line2.bias = torch.nn.Parameter(para[3])
        Min.append([hidden_size+1,minAngel])
    print("The minimum angel of",hidden_size+1,"units is",minAngel)
    
    
    

ax1 = plt.subplot(1,2,1)
ax1.plot(np.array(L1)[:,1],np.array(L1)[:,0],label = "PIC method loss")
ax2 = plt.subplot(1,2,2)
ax2.plot(np.array(Min)[:,1],np.array(Min)[:,0],label = "PIC method min ang")
ax1.set_ylabel("Hidden Size")
ax1.set_xlabel("Loss")
ax2.set_ylabel("Hidden Size")
ax2.set_xlabel("Minimum Angel")
ax1.legend()
ax2.legend()

tmp,tmp1 = Comp(test_data) 
print("Test Loss is",loss_func(tmp, test_data))





