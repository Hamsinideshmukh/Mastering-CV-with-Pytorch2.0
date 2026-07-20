# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


# Defining a basic inception module
class BasicInceptionModule(nn.Module):

    # Initialization for the inception module
    def __init__(self):
        super(BasicInceptionModule, self).__init__()

        # Defining branches with different filter types for the same input

        # 1×1 branch
        self.branch1x1 = nn.Conv2d(192, 64, kernel_size=1)

        # 3×3 branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, padding=1)
        )

        # 5×5 branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(192, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2)
        )

        # Pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(192, 32, kernel_size=1)
        )

    # Forward pass that applies the branches and concatenates results
    def forward(self, x):

        # Apply convolutions on the same input and concatenate

        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        return torch.cat(
            [branch1x1, branch3x3, branch5x5, branch_pool],
            dim=1
        )


# Simple auxiliary classifier
class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class InceptionV3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True):
        super(InceptionV3, self).__init__()

        # Defining the initial convolution and pooling layers
        self.Conv2d_1a_3x3 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            stride=2
        )

        # … (Other layers )

        self.conv = nn.Conv2d(
            32,
            192,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Stacking inception modules
        self.Mixed_5b = BasicInceptionModule()   # An example inception module

        # (Other inception modules not shown for brevity)

        self.conv_after = nn.Conv2d(
            256,
            2048,
            kernel_size=1
        )

        # Auxiliary classifier used in training
        if aux_logits:
            self.AuxLogits = InceptionAux(192, num_classes)
        else:
            self.AuxLogits = None

        # Final classifier replacing the fully connected layer in other architectures
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):

        # Forward pass through each layer up to a certain point

        x = self.Conv2d_1a_3x3(x)
        x = F.relu(x)

        x = self.conv(x)
        x = self.pool(x)

        # In training mode, InceptionV3 uses an auxiliary output for the middle of
        # the network to combat the vanishing gradient problem

        if self.training and self.AuxLogits is not None:
          aux = self.AuxLogits(x)

          x = self.Mixed_5b(x)

          # Final forward pass through later inception modules
          x = self.conv_after(x)

          # Adaptive average pooling and classifier

          x = F.adaptive_avg_pool2d(x, (1, 1))
          x = F.dropout(x, training=self.training)

          x = x.view(x.size(0), -1)

          x = self.fc(x)

        # If auxiliary classifier is used, return both main and auxiliary outputs

        if self.training and self.AuxLogits is not None:
            return x, aux

        return x

# OR
import torchvision
import torch.nn as nn
inceptionv3_pretrained =
torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
inceptionv3_pretrained.fc = nn.Linear(2048, 100) # Change the last layer to fit CIFAR100

import torch
import torch.nn as nn
from torchvision import models
# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inceptionv3_pretrained.to(device)
# Assuming we have a model 'inceptionv3_pretrained' and a dataloader 'train_loader'
optimizer = torch.optim.SGD(inceptionv3_pretrained.parameters(),
lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
# Create data loaders
train_dataset = torchvision.datasets.CIFAR100(root='./data',
train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data',
train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
batch_size=32, shuffle=True)test_loader = torch.utils.data.DataLoader(test_dataset,
batch_size=32, shuffle=False)
# Set number of epochs to train
number_of_epochs = 10
# Train the model
for epoch in range(number_of_epochs):
inceptionv3_pretrained.train() # Set the model to training mode
running_loss = 0.0
for i, (inputs, labels) in enumerate(train_loader):
inputs, labels = inputs.to(device), labels.to(device)
optimizer.zero_grad() # Zero the parameter gradients
# Forward pass
outputs, aux_outputs = inceptionv3_pretrained(inputs)
loss1 = criterion(outputs, labels)
loss2 = criterion(aux_outputs, labels)
loss = loss1 + 0.4 * loss2 # Combine losses from the main and
auxiliary outputs
loss.backward() # Backpropagate the losses
optimizer.step() # Update weights
running_loss += loss.item()
print(f'Epoch {epoch + 1}: Loss: {running_loss / len(train_loader)}')
# Save the model's state dictionary
torch.save(inceptionv3_pretrained.state_dict(),'inceptionv3_cifar100.pth')
