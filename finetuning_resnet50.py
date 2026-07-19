# -*- coding: utf-8 -*-


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch import nn

#Data prep
transform = transforms.Compose([
transforms.Resize(224),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
shuffle=False, num_workers=2)

model = resnet50(weights="IMAGENET1K_V2")
# Freeze all the layers 1st
for param in model.parameters():
  param.requires_grad = False
# unfreeze some layers
for param in list(model.layer4.parameters())+ list(model.fc.parameters()):
  param.requires_grad= True
model.fc = nn.Linear(model.fc.in_features,10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad,model.parameters()), lr = 0.001, momentum 0.9)
num_epochs = 10
for epoch in range(num_epochs):
  model.train()
  running_loss = 0.0
  for i,data in enumerate(trainloader,0):
    inputs , labels = data
    inputs , labels = inputs.to(device),labels.to(device)
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

print(f'Epoch {epoch + 1}, Loss: {running_loss /len(trainloader)}')
# save the model
torch.save(model.state_dict(),'finetuned_resnet50_cifar10.pth')

def calculate_accuracy(loader, model):
  correct = 0
  total = 0
  model.eval() # Set the model to evaluation mode
  with torch.no_grad(): # No need to track the gradients
    for data in loader:
      images, labels = data
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

val_accuracy = calculate_accuracy(testloader, model)
print(f'Accuracy of the network on the test images:{val_accuracy:.2f}%')

# Accuracy of the test images - 89.34%

# training from scratch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
# Set up the transforms for the CIFAR-10 dataset
transform = transforms.Compose([
transforms.Resize((224, 224)), # Resize images to the size that
ResNet expects
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Load the CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
shuffle=True, num_workers=2)
# Load the CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
shuffle=False, num_workers=2)
# Define the classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
'horse', 'ship', 'truck')
# Initialize the ResNet-50 model
model = models.resnet50(pretrained=False) # No pretrained weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes)) # Replace the last
layer
# Use a GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
model.to(device)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Training loop
for epoch in range(10): # Loop over the dataset multiple times
running_loss = 0.0
for i, data in enumerate(trainloader, 0):
# Get the inputs; data is a list of [inputs, labels]
inputs, labels = data
inputs, labels = inputs.to(device), labels.to(device)
# Zero the parameter gradients
optimizer.zero_grad()
# Forward pass
outputs = model(inputs)
loss = criterion(outputs, labels)
# Backward pass and optimize
loss.backward()
optimizer.step()
# Print statistics
running_loss += loss.item()
if i % 2000 == 1999:
# Print every 2000 mini-batches
print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss /2000:.3f}')
running_loss = 0.0
print('Finished Training')
# Save the trained model
PATH = './cifar_resnet50.pth'
torch.save(model.state_dict(), PATH)
# Load the model for evaluation
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load(PATH))
model.to(device)

ef calculate_accuracy(loader, model):
correct = 0
total = 0
model.eval() # Set the model to evaluation mode
with torch.no_grad(): # No need to track the gradients
for data in loader:
  images, labels = data
  images, labels = images.to(device), labels.to(device)
  outputs = model(images)
  _, predicted = torch.max(outputs.data, 1)
  total += labels.size(0)
  correct += (predicted == labels).sum().item()
  accuracy = 100 * correct / total
  return accuracy
# Calculate accuracy on the test set
test_accuracy = calculate_accuracy(testloader, model)
print(f'Accuracy of the network on the test images:{test_accuracy:.2f}%')

import torch
import torchvision.transforms as transforms
import torchvision
from sklearn.metrics import precision_score, recall_score, f1_score,roc_curve, auc, confusion_matrix, accuracy_scoreimport numpy as np
import matplotlib.pyplot as plt
from itertools import product
from torchvision import models
from torch.utils.data import DataLoader
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
# Load the trained ResNet-50 model
model_path = 'resnet50_cifar10.pth'
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10) # CIFAR-10 has 10 classes
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
# CIFAR-10 validation dataset
transform = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
0.225]),
])

validation_set = torchvision.datasets.CIFAR10(root='./data',train=False, download=True, transform=transform)
validation_loader = DataLoader(validation_set, batch_size=64,shuffle=False)

def get_predictions_and_targets(loader, model):
all_preds = []
all_targets = []
with torch.no_grad():
for images, targets in loader:
images, targets = images.to(device), targets.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)
all_preds.extend(preds.cpu().numpy())
all_targets.extend(targets.cpu().numpy())return all_preds, all_targets
# Get predictions and targets from the model
predictions, targets =
get_predictions_and_targets(validation_loader, model)
# Calculate metrics
accuracy = accuracy_score(targets, predictions)
precision = precision_score(targets, predictions, average='macro')
recall = recall_score(targets, predictions, average='macro')
f1 = f1_score(targets, predictions, average='macro')
# Output the metrics
print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall:
{recall}\nF1 Score: {f1}')
# ROC Curve and AUC
# For ROC and AUC, we need to get the probability scores of classes
probabilities = []
with torch.no_grad():
for images, _ in validation_loader:
images = images.to(device)
outputs = model(images)
probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())
probabilities = np.array(probabilities)
# Calculate ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
fpr[i], tpr[i], _ = roc_curve(targets, probabilities[:, i],
pos_label=i)
roc_auc[i] = auc(fpr[i], tpr[i])
# Plot all ROC curves
plt.figure()
colors = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
for i, color in zip(range(10), colors):
plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of
class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for CIFAR-10 classes')
plt.legend(loc="lower right")
plt.show()
# Confusion Matrix
conf_matrix = confusion_matrix(targets, predictions)
# Plot the confusion matrix as an image
fig, ax = plt.subplots(figsize=(10,10))
cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for CIFAR-10 Classification')
# Print the confusion matrix
tick_marks = np.arange(len(validation_set.classes))
plt.xticks(tick_marks, validation_set.classes, rotation=45)
plt.yticks(tick_marks, validation_set.classes)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
thresh = conf_matrix.max() / 2.
for i, j in product(range(conf_matrix.shape[0]),
range(conf_matrix.shape[1])):
plt.text(j, i, format(conf_matrix[i, j], 'd'),
horizontalalignment="center", color="white" if conf_matrix[i, j] >
thresh else "black")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
an image
plt.show()

# Assuming that 'model_ft' is the fine-tuned model and 'model' is
trained from scratch
from sklearn.metrics import precision_score, recall_score,
accuracy_score
# Load the datasets and model
# … (omitted for brevity)
# Function to get predictions
def get_predictions(loader, model):
all_predictions = []
all_targets = []
model.eval()
with torch.no_grad():
  for inputs, targets in loader:
  inputs, targets = inputs.to(device), targets.to(device)
  outputs = model(inputs)
  _, predictions = torch.max(outputs, 1)
  all_predictions.extend(predictions.cpu().numpy())
  all_targets.extend(targets.cpu().numpy())
  return all_predictions, all_targets
  # Get predictions
  predictions_ft, targets_ft = get_predictions(testloader, model_ft)
  predictions, targets = get_predictions(testloader, model)
  # Calculate metrics
  accuracy_ft = accuracy_score(targets_ft, predictions_ft)
  precision_ft = precision_score(targets_ft, predictions_ft,
  average='macro')

  recall_ft = recall_score(targets_ft, predictions_ft,
  average='macro')
  accuracy = accuracy_score(targets, predictions)
  precision = precision_score(targets, predictions, average='macro')
  recall = recall_score(targets, predictions, average='macro')
  # Output the metrics
  print(f"Fine-tuned Model: Accuracy: {accuracy_ft}, Precision:{precision_ft}, Recall: {recall_ft}")
  print(f"From Scratch Model: Accuracy: {accuracy}, Precision:{precision}, Recall: {recall}")
