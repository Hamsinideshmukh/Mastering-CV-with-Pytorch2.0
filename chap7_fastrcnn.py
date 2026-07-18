# -*- coding: utf-8 -*-


import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import torchvision.models.detection as detection_models
# Load the pre-trained Faster R-CNN model
model_detection = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)
model_detection.eval()
image=Image.open('test.png')
transform_detection = transforms.Compose([transforms.ToTensor()])
input_tensor = transform_detection(image).unsqueeze(0)

with torch.no_grad():
  predictions = model_detection(input_tensor)
draw = ImageDraw.Draw(image)
for element in range(len(predictions[0]['boxes'])):
  boxes = predictions[0]['boxes'][element].cpu().numpy()
  score = predictions[0]['scores'][element].cpu().numpy()
  label = predictions[0]['labels'][element].cpu().numpy()
# draw the box with a threshold for scores to filter weak detections
if score > 0.5:
  draw.rectangle([(boxes[0],boxes[1]),(boxes[2],boxea[3])],outline="red",width=3)
  draw.text((boxes[0],boxes[1]), f'Label: {label}', fill="red")
#save
image.save('detection_output.png')
print('Detection output saved as detection_output.png')

import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
# Load the pre-trained Keypoint R-CNN model

model_keypoint = detection_models.keypointrcnn_resnet50_fpn(pretrained=True)
model_keypoint.eval()

image = Image.open('test.png')
transform_keypoint = transforms.Compose([
    transforms.ToTensor()])
input_tensor = transform_keypointrcnn_resnet50_fpn(pretrained=True)
model_keypoint.eval()
# Preprocess the image
image = Image.open('test.png')
transform_keypoint = transforms.Compose([transforms.ToTensor()])
input_tensor = transform_keypoint(image).unsqueeze(0)
# Inference
with torch.no_grad():
prediction_keypoint = model_keypoint(input_tensor)
draw = ImageDraw.Draw(image)
for person in prediction_keypoint[0]['keypoints']:
  keypoints = person.cpu().numpy()
  scores = person[:2]
  for kp , s in zip(keypoints,scores):
  if s>2:  # the model outputs a score of 0,1,2, for visibility
  draw.ellipse([(kp[0] - 3, kp[1] - 3), (kp[0] + 3, kp[1] + 3)],
fill="blue")
# Save the image with keypoints
image.save('keypoint_output.png')
print('Keypoint output saved as keypoint_output.png')