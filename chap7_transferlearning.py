# -*- coding: utf-8 -*-

"""

# leveraging previously learned features
''' 1.feature ectraction
    2. fine tuning -
    are two parts of TL'''

import torch
import torch.nn as nn
class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    # Convolutional layer block 1
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
    kernel_size=3, padding=1)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(kernel_size=2)
    # Convolutional layer block 2
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
    kernel_size=3, padding=1)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(kernel_size=2)
    # Fully connected layers
    self.fc1 = nn.Linear(in_features=64 * 16 * 16, out_features=512)
    self.relu3 = nn.ReLU()
    self.fc2 = nn.Linear(in_features=512, out_features=10)
    def forward(self, x):
    # Block 1
    x = self.pool1(self.relu1(self.conv1(x)))
    # Block 2
    x = self.pool2(self.relu2(self.conv2(x)))
    # Flatten and pass through the fully connected layers
    x = x.view(x.size(0), -1)
    x = self.relu3(self.fc1(x))
    x = self.fc2(x)
    return x
# Instantiate the model
simple_cnn = SimpleCNN()

"""w and b are the parameters of a neural network, in a pretrained model its a result of extensive training."""

# loading a pretrained resnet model
import torchvision.models as models
resnet = torchvision.models.resnet18(pretrained=True)

import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

model_classification = models.resnet50(pretrained=True)
model_classification.eval()  # generally pair it with no_grad, model goes into eval state ( batch norm and drop out)

transform_classification = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
0.225]),
])

Image.open('test.png').convert('RGB')
input_tensor = transform_classification(image).unsqueeze(0) # make a minibatch as expeced by the model

with torch.no_grad():
  output_classification = model_classification(input_tensor)
_,predicted_class = torch.max(output_classification, 1)
print(f'Predicted class (Classification): {predicted_class.item()}')

# Segmentation with DeepLabv3
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models.segmentation as segmentation_models

# Load pretrained DeepLabV3 model
model_segmentation = segmentation_models.deeplabv3_resnet101(pretrained=True)
model_segmentation.eval()

transform_segmentation = transforms.Compose([
transforms.Resize(256),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
0.225]),
])
image = Image.open('test.png').convert('RGB')
input_tensor = transform_segmentation(image).unsqueeze(0)
# inference
with torch.no_grad():
  output_segmentation = output_segmentation(input_tensor)['out'][0]
segmentation = output_segmentation.argmax(0)
to_pil = transforms.ToPILImage()
segmentation_pil = to_pil(segmentation.byte())
segmentation_pil.save('segmentation_output.png')
print('Segmentation mask saved as segmentation_output.png')