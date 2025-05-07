#1. By using open cv library read an image and apply the following noises:
# Gaussian Noise
# Salt and Paper Noise
# Random Noise

#1.1 Gaussian Noise
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image
image = cv2.imread('pi.jpg')  # replace with your image path

# Step 2: Add Gaussian Noise
mean = 5
std = 25  # standard deviation (higher = more noise)
gaussian = np.random.normal(mean, std, image.shape).astype(np.uint8)

noisy_image = image + gaussian

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Image with Gaussian Noise")
plt.imshow(noisy_image)
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#1.2 Salt and Paper Noise

import cv2
import numpy as np
import matplotlib.pyplot as plt
#
# Load and convert image for proper color display in matplotlib
image = cv2.imread('orange.png')

def add_salt_and_pepper_noise(image, noise_ratio=0.5):
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    num_noisy_pixels = int(h * w * noise_ratio)

    for _ in range(num_noisy_pixels):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)

        if np.random.rand() < 0.5:
            noisy_image[y, x] = [0, 0, 0]
        else:
            noisy_image[y, x] = [255, 255, 255]

    return noisy_image

noisy = add_salt_and_pepper_noise(image)

# Display
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Salt & Pepper Noise")
plt.imshow(noisy)
plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#1.3 Random Noise
import numpy as np

image = cv2.imread('orange.png')

def add_random_noise(image, intensity=25):
    noisy_image = image.copy()

    # Generate uniform noise in the range [-intensity, +intensity]
    noise = np.random.randint(-intensity, intensity + 1, noisy_image.shape)

    # Add noise and clip to keep valid pixel range
    noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)

    return noisy_image
noisy = add_random_noise(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Random Noise")
plt.imshow(noisy)
plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#2. using pillow library read an image and compress it.
from PIL import Image

import os

image = Image.open('strawberry.png')
image = image.convert('RGB')  # Converts to RGB mode


width, height = image.size

new_size = (width//2, height//2)

resized_image = image.resize(new_size)

resized_image.save('compressed_image.jpg', optimize=True, quality=50)

original_size = os.path.getsize('strawberry.png')

compressed_size = os.path.getsize('compressed_image.jpg')

print("Original Size: ", original_size)

print("Compressed Size: ", compressed_size)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title(f"Original Image, Size: {original_size} Bytes")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title(f"Compressed image, Size: {compressed_size} Bytes")
plt.imshow(resized_image)
plt.show()

