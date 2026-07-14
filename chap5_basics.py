
conv_layer = nn.Conv2d(in_channels = 1, out_channels=32, kernel_size = 3, stride =1, padding=1)
# out_channels rep. the number of filters this layer will learn
relu = nn.ReLU
# or u can also do
self.activation = nn.ReLU
# and do
x = self.activation(x)
# maxpool, reduces the spatial dimenions, decreasing the computational load of the netwrok
pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
# pooling helps in increasing the receptive field
# fully connected layer
fc = nn.Linear(in_features = 32*14*14, out_features = 10)
# dropout randomly drops out a proportion of the neurons during training, which helps
# model from relying too much on one feature
dropout = nn.Dropout(p=0.5) # used to prevent overfitting

"""Layers of CNN play imp. role in extracting features."""

import torch
import torch.nn as nn
conv_layer = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1)
# CHOOSING THE RIGHT FILTER SIZE IS CRUCIAL for caputring right features from data
# no. of filters tells no. of unique features network can extract
conv_layer = nn.Connv2d(in_channels = 1, out_channels=32,kernel_size =3, stride=1, padding=1)

""" CHOOSING THE RIGHT FILTER SIZE IS CRUCIAL for caputring right features from data
 no. of filters tells no. of unique features network can extract
"""

conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,
stride=1, padding=1)
conv_layer = nn.Conv2d(in_channels=1, out_channels=32,
kernel_size=3, stride=1, padding=1)

#using filters in real model
class MyModel(nn.Module):
  def __init__(self):
    super(MyModel,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1)
    self.conv2= nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, stride=1, padding=1)
    self.fc = nn.Linear(64*28*28,10)
  def forward(self,x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0),-1)
    x = self.fc(x)
    return x

