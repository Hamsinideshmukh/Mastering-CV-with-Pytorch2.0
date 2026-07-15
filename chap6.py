# -*- coding: utf-8 -*-

import pandas as pd
# Load dataset
data = pd.read_csv('dataset.csv')
# Handle missing values
data.fillna(0, inplace=True)
# Handle outliers
data = data[(data['value'] > lower_bound) & (data['value'] <upper_bound)]

from sklearn.preprocessing import StandardScaler
data = pd.read_csv('dataset.csv')
# standarsize  features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# handling missing data
data.fillna(method='ffill', inplace = True)
 # create polynomial features
from sklearn.preprocessing impor tPolynomialFeatures
poly = PolynomialFeatures(degrees = 2)
data_poly = poly.fit_transform(data)

from torchvision import transforms
# Define transformations
transform = transforms.Compose([
transforms.Resize((256, 256)),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# Load dataset and apply transformations
dataset = datasets.ImageFolder(root='data', transform=transform)

from torchvision import transforms
transform_pipeline = transforms.Compose([
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
])

# random horizantal flip
transforms.RandomHorizantalFlip(p=0.5)
# random vertical flip
transforms.RandomVerticalFlip(p=0.5)
#color jitter
transform.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0)
# random rotation
transforms.RandomRotaton(degrees, resample=False, expland= False, center = None, fill=None)
# random resized crop
transforms.RandomResizedCrop(size,scale=(0.08,1.0),ratio = (3./4.,4./3.),interpolation=2)
# random affine
transforms.RandomAffine(degrees, tranlanslate = None, scale = None, shear = None, resample = False, fillcolor = 0)
# gaussian-blur
transform.GaussianBlur(kernel_size, sigma =(0.1,2.0))
# random crop
transforms.RandomCrop(size, padding=None, pad_if_needed=False,
fill=0, padding_mode='constant')
# random erasing
transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3,
3.3), value=0, inplace=False)
# random perspective
transforms.RandomPerspective(distortion_scale=0.5, p=0.5,
interpolation=3)
# center crop
transfroms.CenterCrop(size)

"""resample is same as interpolation
also,
affine - warps image like geometric transformation, rotation, scale, shear,translate, shear, but perspective means changing the view of the image, main difference affine keeps parallel lines parallel, but perspective makes them look like they are converging at afar
"""

