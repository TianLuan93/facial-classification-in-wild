# -*- coding: utf-8 -*-
"""
Created on Sun May 24 19:56:41 2020

@author: Dev1ce
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import Normalizer

learning_rate = 0.0001
EPOCH=701
batch_size = 50
path = r"Subset For Assignment SFEW/"
#%%
##1 Angry   2 Disgust   3 Fear   4 Happy    5 Neutral   6 Sad   7 Surprise
data = []
labels = np.zeros(675)
index = 0
normalizer=Normalizer(norm='l2')
for dirpath,dirnames,filenames in os.walk(path):
    for filename in filenames:
        Img = cv2.imread(os.path.join(dirpath,filename),0)
        #resize the image of dataset to let data be smaller 90*72
        Img = cv2.resize(Img,(90,72))  
        width,hight=Img.shape
        if dirpath[27:len(dirpath)] == "Angry":
            label = 0
        elif dirpath[27:len(dirpath)] == "Disgust":
            label = 1 
        elif dirpath[27:len(dirpath)] == "Fear":
            label = 2 
        elif dirpath[27:len(dirpath)] == "Happy":
            label = 3
        elif dirpath[27:len(dirpath)] == "Neutral":
            label = 4
        elif dirpath[27:len(dirpath)] == "Sad":
            label = 5
        elif dirpath[27:len(dirpath)] == "Surprise":
            label = 6
        Img = normalizer.transform(Img).reshape(1,90,72).astype(np.double)
        #Img = Img.reshape(1,90,72).astype(np.double)
        labels[index] = label
        index = index + 1
        data.append(Img)
data = np.asarray(data,dtype = np.double)
train_x,test_x,train_y,test_y = train_test_split(data,labels)
#%%
class CNN(nn.Module): 
    def __init__(self): 
        super(CNN, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,stride=1, padding=2) 
        self.relu1 = nn.ReLU() 
        self.pool1 = nn.MaxPool2d(kernel_size=2) # (16,45,36) 
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)  
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3)
        self.out = nn.Linear(32*15*12, 7) # (16,15,12) 
    def forward(self, x): 
        x = self.conv1(x) 
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.shape[0],32*15*12) # flat the (len,16,15,12) to (len,16*15*12) 
        self.feature = x
        output = self.out(x) 
        return output 
  
cnn = CNN()
train = torch.tensor(train_x)
labels = torch.tensor(train_y).long()
cnn.double()
#%%
print(train.shape)
#%%
#optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
#loss_fun
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    # calculate the output
    output = cnn(train)
    # calculate teh loss
    loss = loss_func(output, labels)
    # zero last grad
    optimizer.zero_grad()
    # back propgation
    loss.backward()
    # update the parameters
    optimizer.step()
    if epoch%100 == 0:
        prediction = torch.max(F.softmax(output), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = labels.data.numpy()
        accuracy = sum(pred_y == target_y)/(labels.shape[0])
        print("Train loss is",loss,"\nTrain accuracy is",100*accuracy,"%")
    
#%%
test = torch.tensor(test_x)
labels1 = torch.tensor(test_y).long()
output = cnn(test)
prediction = torch.max(F.softmax(output), 1)[1]
print(prediction)
pred_y = prediction.data.numpy().squeeze()
target_y = labels1.data.numpy()
accuracy = sum(pred_y == target_y)/(labels1.shape[0])
print("Test loss is",loss,"\nTest accuracy is",100*accuracy,"%")
#%%
#We have a model before, if want to use
#cnn = torch.load('net.pkl')
#torch.save(cnn, 'net.pkl') 
#save the model we have
#myNet = torch.load('net.pkl')
#print(myNet)
#%%
#refresh the accuracy and loss are the train accuracy and train loss 
output = cnn(train)
loss = loss_func(output, labels)
prediction = torch.max(F.softmax(output), 1)[1]
pred_y = prediction.data.numpy().squeeze()
target_y = labels.data.numpy()
accuracy = sum(pred_y == target_y)/(labels.shape[0])
print("Train loss is",loss,"\nTrain accuracy is",100*accuracy,"%")
#%%
plot = []
#prune the neurons
def prune(idx):
    weight = cnn.conv1.weight.detach().numpy()
    importance = []
    for i in range(weight.shape[0]):
        imp = np.sqrt(np.sum(np.square(weight[i])))
        if imp != 0:
            importance.append(imp)
        else:
            importance.append(np.inf)
    importance = np.array(importance)
    del_idx = np.argmin(importance)
    output = cnn(train)
    loss = loss_func(output, labels)
    prediction = torch.max(F.softmax(output), 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    target_y = labels.data.numpy()
    accuracy = sum(pred_y == target_y)/(labels.shape[0])
    print("Train loss is",loss,"\nTrain accuracy is",100*accuracy,"%")
    plot.append(np.array([idx,importance[del_idx]]))
    w = cnn.conv1.weight
    b = cnn.conv1.bias
    b[del_idx] = 0
    w[del_idx] = torch.zeros(w[del_idx].shape)
    cnn.conv1.weight = torch.nn.Parameter(w)
    cnn.conv1.bias = torch.nn.Parameter(b)
#%%
#prune the neurons
#may cause need memory problem because it is too big
#for i in range(5):
#    prune(i)
#%%
## plot part
#plt.plot(plot[:,0],plot[:,1],c='black')
#plt.xlabel("The number of pruned neurons")
#plt.ylabel("Minimum Importance")
#plt.title("The minimum importance changed with the number of pruned neurons")

