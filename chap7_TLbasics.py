# -*- coding: utf-8 -*-

# while applying TL consider the similarity between the your task and the
# task which it was originally trained on

"""Before youtilising a model, data must be formatted and normalized to meet the specifications of the model."""

from torchvision import transforms
# define std image net transform
preprocess = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrops(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
0.225]),
])
# this transformation should be applied both in training and validation

''' freeze the weights of all the layers excpet for the final classification'''
from torchvision.models import resnet50
from torch import nn
model = resnet50(pretrained=True)
for param in model.parameters():
  param.requires_grad=False

# replace the last layer to match CIFAR-10's classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features,10)

"""Now in this fine tuning phase , we not only replace the last layer but wealso fine tune it ( when your model is enough large and different from the dataset model)"""

# unfreeze the last layers
for param in list(model.layer4.parameters())+list(model.fc.parameters()):
  param.requires_grad() = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
for epoch in range(num_epochs):
  # training and upadte steps over here
  pass
# we had unfreezed the last block of the resnet layer4 along with fully connected layer

