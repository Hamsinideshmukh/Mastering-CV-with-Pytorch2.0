# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
class VGG16(nn.Module):
  def __init__(self, num_classes=1000): # VGG16 has 1000 classes by default
  super(VGG16, self).__init__()
  self.features = self._create_features()
  self.classifier = self._create_classifier(num_classes)
  def _create_features(self):
    layers = [
    # Block 1
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # Block 2
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # Block 3
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # Block 4
    nn.Conv2d(256, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # Block 5
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2)
    ]
    return nn.Sequential(*layers)
  def _create_classifier(self, num_classes):
    layers = [
    nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096), nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, num_classes)
    ]
    return nn.Sequential(*layers)
  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, start_dim=1)
    #classifier intake
    x = self.classifier(x)
    return x

# sequential layout
import torch.nn as nn# Defining each block of the VGG16 architecture
def make_layers(cfg):
  layers = []
  in_channels = 3
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
      nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      in_channels = v
    return nn.Sequential(*layers)
# Configuration for VGG16
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
512, 'M', 512, 512, 512, 'M']
class VGG16(nn.Module):
  def __init__(self, num_classes=1000):
    super(VGG16, self).__init__()
    # Feature extraction part
    self.features = make_layers(cfg)
    # Classifier part
    self.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
    )
  # The VGG16 forward pass
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

# from pytorch model zoo
from torchvision import models
vgg16_pretrained = models.vgg16(weights=torchvision.models.BGG16_Weights.DEFAULT)
vgg16_pretrained.classifier[6] = nn.Linear(4096,100) # change the last layer to fit CIFAR100

from torchvision.transforms import transforms
transform = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),transforms.ToTensor(),
  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2009, 0.1984,
  0.2023)),
  ])

device = torch.device("cude:0" if torch.cuda.is_availabe() else "cpu")
optimizer = torch.optim.SGD(vgg16_pretrained.parameters(), lr=0.01,
momentum=0.9)
criterion = nn.CrossEntropyLoss()

train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data',train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=64, shuffle=False)

number_of_epochs = 5

for epoch in range(number_of_epochs):
  running_loss = 0.0
  for i , data in enumerate(train_loader,0):
    inputs , labels = data[0].to(device), data[1].to(device)
    optimizer.zero_grad()
    outputs = vgg16_pretrained(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  print(f'poch {epoch + 1}: Loss: {running_loss /len(train_loader)}')

  # Save the model's state dictionary
torch.save(vgg16_pretrained.state_dict(), 'vgg16_cifar100.pth')

correct, total = 0, 0
with torch.no_grad():
  for data in test_loader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = vgg16_pretrained(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  print(f'Accuracy on test images: {100 * correct // total}%')

# Implement ResNet in PyTorch
import torch
import torch.nn as nn
from torchvision import models

class BasicBlock(nn.Module):
  expansion = 1
  def __init__(self, in_channels, out_channels, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    # 1x1 convolutions are used as bottlenecks for reducing and then increasing dimensions
    # without changing width/height size
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
    stride=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    # A 3x3 convolution is used in the middle
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
    stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    # Another 1x1 convolution with the expansion factor applied
    self.conv3 = nn.Conv2d(out_channels, out_channels *
    self.expansion, kernel_size=1, stride=1, bias=False)
    self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample

def forward(self, x):
  identity = x if self.downsample is None else self.downsample(x)
  out = self.relu(self.bn1(self.conv1(x)))
  out = self.relu(self.bn2(self.conv2(out)))
  out = self.bn3(self.conv3(out))
  out += identity
  out = self.relu(out)
  return out

class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    super(ResNet, self).__init__()self.in_channels = 64
    self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7,
    stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(self.in_channels)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # Each _make_layer() creates one stage of blocks in ResNet
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Adaptive pooling to
    fit the final layer
    self.fc = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, out_channels, blocks, stride=1):downsample = None
      if stride != 1 or self.in_channels != out_channels *block.expansion:
        downsample = nn.Sequential(
        nn.Conv2d(self.in_channels, out_channels * block.expansion,
        kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels * block.expansion),
        )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride,
        downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
          layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.maxpool(x)
    # ResNet layers
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x
def resnet50(num_classes=1000):
# Constructs a ResNet-50 model
  return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
# Instantiate a ResNet-50 model
model = resnet50(num_classes=1000)

# basic building block
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * BasicBlock.expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels * BasicBlock.expansion)

        # Optional downsampling
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Passing input through conv1 -> bn1 -> relu stack
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Passing input through conv2 -> bn2 stack
        out = self.conv2(out)
        out = self.bn2(out)

        # Applying downsampling if needed
        if self.downsample is not None:
            residual = self.downsample(x)

        # Adding the residual connection
        out += residual
        out = self.relu(out)

        return out

import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.in_channels = 64

        # Initial convolutional layer with max-pooling
        self.conv1 = nn.Conv2d(
            3,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Adaptive average pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            512 * block.expansion,
            num_classes
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # Helper function to create each ResNet layer
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        if (
            stride != 1
            or self.in_channels != out_channels * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(
                    out_channels * block.expansion
                ),
            )

        # First block in the layer
        layers = [
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample
            )
        ]

        self.in_channels = out_channels * block.expansion

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels
                )
            )

        return nn.Sequential(*layers)

    # Forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # First 1×1 convolution (reduces channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second 3×3 convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Third 1×1 convolution (expands channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    # Forward function is the same as BasicBlock
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Instantiate ResNet-50
resnet50_model = ResNet(
    Bottleneck,
    [3, 4, 6, 3],
    num_classes=1000
)

def resnet50(num_classes=1000):
# Constructs a ResNet-50 model
return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
# Initialize a ResNet-50 model
resnet50_custom = resnet50(num_classes=1000)

import torchvision
resnet50_pretrained =
torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weig
hts.IMAGENET1K_V1)
resnet50_pretrained.fc =
nn.Linear(resnet50_pretrained.fc.in_features, 100) # Change the last layer to fit CIFAR100

import torch
import torch.nn as nn
import torchvision
from torchvision import models

# Send model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50_pretrained.to(device)

# Optimizer and loss function
optimizer = torch.optim.SGD(
    resnet50_pretrained.parameters(),
    lr=0.001,
    momentum=0.9
)
criterion = nn.CrossEntropyLoss()

# Create data loaders
train_dataset = torchvision.datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# Number of epochs
number_of_epochs = 10

# Training loop
for epoch in range(number_of_epochs):

    resnet50_pretrained.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader):

        # Get inputs and labels
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = resnet50_pretrained(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}: Loss = {running_loss / len(train_loader):.4f}")

# Save trained model
torch.save(
    resnet50_pretrained.state_dict(),
    "resnet50_cifar100.pth"
)

# Evaluation
resnet50_pretrained.eval()

correct = 0
total = 0

with torch.no_grad():

    for data in test_loader:

        images, labels = data[0].to(device), data[1].to(device)

        outputs = resnet50_pretrained(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total

print(f"Accuracy of the network on test images: {100 * accuracy:.2f}%")

