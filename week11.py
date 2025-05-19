#1-Read an image using open-cv in RGB mode then split it into it's channels
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("pi.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(image_rgb)
plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.imshow(r, cmap='Reds')
plt.title("Red")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(g, cmap='Greens')
plt.title("Green")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(b, cmap='Blues')
plt.title("Blue")
plt.axis('off')

plt.show()
#-----------------------------------------------------------------------------------------------------------------------

#2-Convert from RGB into CMY.
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("pi.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb = image_rgb / 255.0

c = 1 - image_rgb[:, :, 0]  # take all rows , take all columns, take first channel (RED)
m = 1 - image_rgb[:, :, 1]  # take all rows , take all columns, take second channel (GREEN)
y = 1 - image_rgb[:, :, 2]  # take all rows , take all columns, take third channel (BLUE)

# Combine the CMY channels
cmy_image = np.stack((c, m, y), axis=-1)

plt.figure(figsize=(10, 10))

# Cyan
plt.subplot(1, 3, 1)
plt.imshow(c, cmap='Blues')
plt.title("Cyan")
plt.axis('off')

# Magenta
plt.subplot(1, 3, 2)
plt.imshow(m, cmap='Purples')
plt.title("Magenta")
plt.axis('off')

# Yellow
plt.subplot(1, 3, 3)
plt.imshow(y, cmap='YlOrBr')
plt.title("Yellow")
plt.axis('off')

plt.show()
#-----------------------------------------------------------------------------------------------------------------------

# 3-Convert from RGB to HSV.
import cv2
import matplotlib.pyplot as plt

image_rgb = cv2.cvtColor(cv2.imread("strawberry.png"), cv2.COLOR_BGR2RGB)

# Convert RGB to HSV
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image (RGB)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_hsv)
plt.title("HSV Image")
plt.axis('off')
plt.show()
#-----------------------------------------------------------------------------------------------------------------------


#4-Apply region based pseudocolor on an image.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image and convert to grayscale
image = cv2.imread("pi.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Create an empty color image
pseudo = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

# Step 3: Apply region-based color mapping
pseudo[gray < 50]        = [255, 0, 0]     # Red
pseudo[(gray >= 50) & (gray < 100)]  = [0, 255, 0]     # Green
pseudo[(gray >= 100) & (gray < 150)] = [0, 0, 255]     # Blue
pseudo[(gray >= 150) & (gray < 200)] = [255, 255, 0]   # Cyan
pseudo[gray >= 200]      = [255, 255, 255] # White

# Step 4: Show results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(pseudo, cv2.COLOR_BGR2RGB))
plt.title("Region-Based Pseudocolor")
plt.axis("off")
plt.show()