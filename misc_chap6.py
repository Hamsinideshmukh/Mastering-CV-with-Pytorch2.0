# -*- coding: utf-8 -*-

# Additional augmentations for CIFAR-10 training data
transform_augmented = transforms.Compose([
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.ColorJitter(brightness=0.1, contrast=0.1,
saturation=0.1, hue=0.1),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset_augumented = torchvision.datasets.CIFAR10(root='./data',
train=True, download=True,transform = transform_augumented)
trainloader_augumented = torch.utils.data.DataLoader(trainset_augumented,batch_size =4,shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
download=True, transform=transform)
testloader = torch.utils.DataLoader(testset,batch_size=4,shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F
class SimpleCNN(nn.Module):
def __init__(self):
  super(SimpleCNN, self).__init__()
  self.conv1 = nn.Conv2d(3, 6, 5)
  self.pool = nn.MaxPool2d(2, 2)
  self.conv2 = nn.Conv2d(6, 16, 5)
  self.fc1 = nn.Linear(16 * 5 * 5, 120)
  self.fc2 = nn.Linear(120, 84)
  self.fc3 = nn.Linear(84, 10)
def forward(self, x):
  x = self.pool(F.relu(self.conv1(x)))
  x = self.pool(F.relu(self.conv2(x)))
  x = x.view(-1, 16 * 5 * 5)
  x = F.relu(self.fc1(x))
  x = F.relu(self.fc2(x))
  x = self.fc3(x)
  return x
net = SimpleCNN()
compiled_net = torch.compile(net)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(compiled_net.parameters(),lr=0.001,momentum=0.9)
for epoch in range(5):
  running_loss = 0.0
  for i,data in enumerate(trainloader,0):
    inputs,labels = data # data tuple has a batch of inputs and labels
    # which means inputs and labels still have the batch dimension
    if i % 2000 == 1999:
    # Print every 2000 mini-batches
    print('[%d, %5d] loss: %.3f' %
    (epoch + 1, i + 1, running_loss / 2000))
    running_loss = 0.0
    print('Finished Training')
# Test the network on the test data
correct = 0
total = 0
with torch.no_grad():
for data in testloader:
images, labels = data
outputs = compiled_net(images)
_, predicted = torch.max(outputs.data, 1)
total += labels.size(0)
correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' %
(100 * correct / total))

"""The accuracy is 66p and we got 6 percent improvement

When to use data aug:
insufficient data
imbalanced classes
model generalisation
real-world variability
domain specific augumentations- eg medical images may use augs.  that mimic common artifacts found in medical scans

Begin with simple augs and then go to complex
Ensure they are relaventto problem
-> they should reflet the the type of variation expected during the testing
avoid if they dont help the model in generalizing
"""

