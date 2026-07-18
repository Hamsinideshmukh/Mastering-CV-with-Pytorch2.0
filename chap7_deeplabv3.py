# -*- coding: utf-8 -*-

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

