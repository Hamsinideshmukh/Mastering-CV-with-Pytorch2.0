# -*- coding: utf-8 -*-


# Dataloader is module in pytorch effiecient to load data in batches
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder #looks at te folder and makes dataset
from torch.utils.data import DataLoader

# Define the data preprocessing transformations
transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225]),
])
dataset = ImageFolder(root='pathtodataset',transform = transform)
dataloader = DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4)

# Flips the image horizontally with a given probability (p=1 in thisexample).
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
# Load image
image = Image.open('test.png')
# RandomHorizontalFlip
random_horizontal_flip = transforms.RandomHorizontalFlip(p=1)
h_flipped_image = random_horizontal_flip(image)
# Display
plt.imshow(h_flipped_image)
plt.title('RandomHorizontalFlip')
plt.show()

# Flips the image vertically with a given probability (p=1 in this example).
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
# Load image
image = Image.open('test.png')
# RandomVerticalFlip
random_vertical_flip = transforms.RandomVerticalFlip(p=1)
flipped_image = random_vertical_flip(image)
# Display
plt.imshow(flipped_image)
plt.title('RandomVerticalFlip')
plt.axis('off')
plt.show()

# Randomly changes the brightness, contrast, saturation, and hue of an image.
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
# Load image
image = Image.open('test.png')
# ColorJitter
color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5,
hue=0.5)
jittered_image = color_jitter(image)
# Display
plt.imshow(jittered_image)
plt.title('ColorJitter')
plt.show()

# Rotates the image by a random angle.
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
# Load image
image = Image.open('test.png')
# RandomRotation
random_rotation = transforms.RandomRotation(degrees=(0, 180))
rotated_image = random_rotation(image)
# Display
plt.imshow(rotated_image)
plt.title('RandomRotation')
plt.axis('off')
plt.show()

image = Image.open('test.png')
random_rotation = transforms.RandomRotation(degrees=(0,180))
rotated_image = random_image = random_rotation(image)
# Display
plt.imshow(rotated_image)
plt.title('RandomRotation')
plt.axis('off')
plt.show()

# Crops the given image to a random size and aspect ratio.
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
# Load image
image = Image.open('test.png')
# RandomResizedCrop
random_resized_crop = transforms.RandomResizedCrop(size=(224, 224))
resized_cropped_image = random_resized_crop(image)
# Display
plt.imshow(resized_cropped_image)
plt.title('RandomResizedCrop')
plt.axis('off')
plt.show()

# Applies random affine transformations to the image.
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
# Load image
image = Image.open('test.png')
# RandomAffine
random_affine = transforms.RandomAffine(degrees=(0, 360), translate=
(0.1, 0.1), scale=(0.7, 1.3), shear=(0, 20))
affine_image = random_affine(image)
# Display
plt.imshow(affine_image)
plt.title('RandomAffine')
plt.axis('off')
plt.show()

# Blurs the image with a Gaussian filter.
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
# Load image
image = Image.open('test.png')
# GaussianBlur
gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=
(0.1, 5))
blurred_image = gaussian_blur(image)
# Display
plt.imshow(blurred_image)
plt.title('GaussianBlur')
plt.axis('off')
plt.show()

# Crops the given image at a random location.
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
# Load image
image = Image.open('test.png')
# RandomCrop
random_crop = transforms.RandomCrop(size=(100, 100))
cropped_image = random_crop(image)
# Display
plt.imshow(cropped_image)
plt.title('RandomCrop')
plt.axis('off')
plt.show()


# Randomly erases a rectangle region in an image.
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
# Load image
image = Image.open('test.png')
# Convert to tensor for RandomErasing
to_tensor = transforms.ToTensor()
image_tensor = to_tensor(image)
# RandomErasing
random_erasing = transforms.RandomErasing(p=1, scale=(0.1, 0.33),
ratio=(0.3, 3.3), value='random')
erased_image_tensor = random_erasing(image_tensor)
# Convert back to image to display
to_pil = transforms.ToPILImage()
erased_image = to_pil(erased_image_tensor)
plt.imshow(erased_image)
plt.title('RandomErasing')
plt.axis('off')
plt.show()

# random perspective
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
image = Image.open('test.png')
random_perspective = transform.Randomperspective(distortion_scale=0.6),p=1,interpolation=3
perspective_image=random_perspective(image)

plt.imshow(perspective_image)
plt.title('Randomperspective')
plt.axis('off')
plt.show()

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
# Load image
image = Image.open('test.png')
all_augumentations = transforms.Compose([
    transforms.CenterCrop(size=(100,100)),
    transforms.RandomPerspective(distortion_scale=0.6,p=1,interpolation=3)
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))],p=0.5)
    transforms.RandomAffine(degrees=(0,360),translate=(0.1,0.1),scale=(0.7,1.3),shear=(0,20)),
    transforms.RandomResizedCrop(size=(224,224))
    transforms.RandomRotation(degrees=(0,180)),
    transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue = 0.5),
    transforms.RandomVerticalFlip(p=0.5)
    transfroms.RandomHorizantalFlip(p=0.5)
    transforms.ToTensor(),
    transform.RandomErasing(p=0.5,scale=(0.1,0.33),ratio=(0.3,3.3),value='random')
])

# function to display the image
def show_augmented_images(image_path, num_images=5):
    image = Image.open(image_path)
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
    augmented_image = all_augmentations(image)
    ax = plt.subplot(1, num_images, i + 1)
    ax.imshow(transforms.ToPILImage()(augmented_image))
    ax.axis('off')
    plt.show()
show_augmented_images('test.png')

