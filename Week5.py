#****************************************************** Histogram ******************************************************
# A Histogram is a graphical representation of the distribution of pixel intensities in an image.
# It shows how many pixels in an image have a specific intensity value, typically ranging from 0 (black) to 255 (white)
# for grayscale images.

# Here is an example that shows the Histogram Graph of a given image in grey scale:
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("example.jpg").convert("L")  # Convert to grayscale

histogram = image.histogram()

plt.figure(figsize=(10, 5))
plt.bar(range(256), histogram, color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')
plt.title('Grayscale Image Histogram')
plt.show()

# Applications of Histogram:
# 1- The brightness of an image can be adjusted using information from its histogram.
# 2- By comparing input and output histograms of image, we can identify type of transformation applied in the algorithm.
# 3- Motion detection Algorithms.
# 4- Compare images by analyzing their intensity distributions.

# Histogram Processing Techniques:
# 1- Sliding
# 2- Stretching
# 3- Equalization
#***********************************************************************************************************************


#************************************************** Histogram Sliding **************************************************
# Histogram Sliding is a technique used to modify the brightness of an image by shifting the intensity values
# of all pixels by a fixed amount.

# To apply this technique, a shift value (constant) is selected and then either added to or subtracted
# from each pixel's intensity in the image.

# There are two types of Histogram Sliding:
# 1- Right Sliding --> Increasing brightness by adding a constant value to pixel intensities.
# 2- Left Sliding  --> Decreasing brightness by subtracting a constant value from pixel intensities.

# Here is a simple function to apply Histogram Sliding on an image :
import cv2

def slide_hist(img,shift):
    fill = np.ones(img.shape,dtype = np.uint8) * shift
    return cv2.add(img,fill)
# This function takes two parameters: an image matrix (containing pixel values) and a shift amount constant.
# First, we need to create another matrix of the same size as the image, where all values are equal to the shift amount.
# To achieve this, we use the np.ones() function to generate a matrix filled with ones, then set its dimensions
# using the "img.shape" property to match the image matrix.
# Next, we convert the ones matrix to an unsigned 8-bit integer format and perform element-wise multiplication
# with the shift amount constant.
# Now, we have two matrices of equal size, and ready to add or subtract the shifting matrix from the image matrix
# to apply right or left Histogram Sliding.
#***********************************************************************************************************************


#************************************************* Histogram Stretching ************************************************
# is a technique used to enhance the contrast of an image by expanding its intensity range. It redistributes pixel
# intensities across a wider range, making dark areas darker and bright areas brighter.
#"MEN EL A5ER BETWAZA3 QEYAM EL PIXELS ZAY MA BENFUNAT EL KOTSHENA"

# Here is a simple code applying the Histogram Stretching :
import numpy as np
import cv2

pi_grey = cv2.imread("pi.jpg", 0)
constant = (255-0)/(pi_grey.max()-pi_grey.min())  # --First step--

pi_grey_stretch = (pi_grey-pi_grey.min()) * constant # --Second step--
pi_grey_stretch = np.clip(pi_grey_stretch,0,255).astype(np.uint8) # --Third step--

# First step: We first calculated the constant value, This constant ensures that the pixel intensity values are stretched from their
# original range [I_min, I_max] to the full range [0, 255].

# Second step: This is the core histogram stretching transformation, which adjusts the pixel intensity values,
# we subtracted the pixels from min. value to shift the intensity range so that
# the lowest value in the image is now 0 instead of some unknown value.

# Third Step: cliping values to make sure they are in range [0,255],
# any value less than 0 will be zero & any value greater than 255 will be 255, the "astype()" function ensures that
# pixels values are not float(converted to unsigned integer 8-bits)
#***********************************************************************************************************************


#************************************************ Histogram Equalization ***********************************************
# Histogram Equalization is a technique used to enhance the contrast of an image by redistributing pixel intensity
# values to achieve a more uniform histogram. It equalizes all pixel values by applying a transformation that produces
# a flattened, more evenly distributed histogram, making details more visible, especially in low-contrast images.

# Applying it is very easy, we just call the function "equalizeHist()" and pass the image as a parameter to the function
# this function is built in cv2 library.

import cv2
pi_grey = cv2.imread("pi.jpg", 0)
pi_grey_equalized = cv2.equalizeHist(pi_grey)
#***********************************************************************************************************************





#******************************************************* Sheet 5 *******************************************************
#1.  Read an image and convert it to gray scale, then by using matplotlib library then show the image and its histogram.
import matplotlib.pyplot as plt
from PIL import Image

pi_grey = Image.open("pi.jpg").convert("L")
pi_grey_histogram = pi_grey.histogram()

rows = 1
columns = 2
figure_all_in_one = plt.figure(figsize=(6,8))

plt.subplot(rows,columns,1)
plt.imshow(pi_grey)
plt.title("Pi Image")
plt.axis("off")

plt.subplot(rows,columns,2)
plt.plot(pi_grey_histogram)
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')
plt.title("Histogram")

plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#2. Read an image and convert it to gray scale using pillow library, then show the image and its histogram.
from PIL import Image
import matplotlib.pyplot as plt

pi_grey = Image.open("pi.jpg").convert("L")  # Convert to grayscale
pi_grey.show()

pi_grey_histogram = pi_grey.histogram()

plt.figure(figsize=(8, 5))
plt.plot(pi_grey_histogram)
plt.title("Grayscale Image Histogram")
plt.show()

# Another application in question 2 is to obtain the histogram of a colored image, then extract the intensity ranges
# for the Red, Green, and Blue (RGB) channels and visualize them separately.from PIL import Image
from PIL import Image
import matplotlib.pyplot as plt

pi = Image.open("pi.jpg")
pi.show()
var = pi.histogram()

pi_red   = var[0:255]
pi_green = var[256:511]
pi_blue  = var[512:767]

plt.plot(pi_red,color = "red")g
plt.plot(pi_green,color = "green")
plt.plot(pi_blue,color = "blue")
plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#3.Read an image and convert it to gray scale using open CV library, then show the image and its histogram.
import cv2
import matplotlib.pyplot as plt

pi_grey = cv2.imread("pi.jpg",flags = 0)

var  = cv2.calcHist([pi_grey],[0],None,[256],[0, 255])
# [pi_grey] : the image that we will calculate its histogram, this parameter can take more than one image
# [0]       : the channel number targeted (in case the image is grey scale so channel = 0)
# None      : disable masking, it means we want to calculate the histogram of the entire image not part of it
# [256]     : the number of bins we are using for the graph, decreasing it decreases graph resolution
# [0, 255]   : the range of pixel values

plt.plot(var)
plt.show()

cv2.imshow("Pi grey image",pi_grey)
cv2.waitKey()
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------
#4. Apply histogram stretching on gray image and show the images after histogram stretching and its histogram.
import cv2
import matplotlib.pyplot as plt
import numpy as np

pi_grey = cv2.imread("strawberry.png", 0)
constant = (255-0)/(pi_grey.max()-pi_grey.min())

pi_grey_stretch = (pi_grey-pi_grey.min()) * constant
pi_grey_stretch = np.clip(pi_grey_stretch,0,255).astype(np.uint8)

rows = 2
columns = 2
figure_all_in_one = plt.figure(figsize=(6,8))

plt.subplot(rows,columns,1)
plt.imshow(pi_grey)
plt.title("Original Pi")
plt.axis("off")

plt.subplot(rows,columns,2)
plt.hist(pi_grey.ravel(),256)
plt.title("Original Pi Histogram")

plt.subplot(rows,columns,3)
plt.imshow(pi_grey_stretch)
plt.title("Stretched Pi")
plt.axis("off")

plt.subplot(rows,columns,4)
plt.hist(pi_grey_stretch.ravel(),256)
plt.title("Stretched Pi Histogram")

plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#5-  Apply left and right histogram Sliding on gray image and show the images after sliding and their histogram.
import cv2
import matplotlib.pyplot as plt
import numpy as np

def slide_hist(img,shift):
    fill = np.ones(img.shape,dtype = np.uint8) * shift
    return cv2.add(img,fill)

pi_grey = cv2.imread("pi.jpg", 0)
pi_grey_histogram = cv2.calcHist([pi_grey],[0],None,[256],[0,255])

pi_grey_shift = slide_hist(pi_grey,20)
pi_grey_shift_histogram = cv2.calcHist([pi_grey_shift],[0],None,[256],[0,255])

rows = 2
columns = 2
plt.subplot(rows,columns,1)
plt.imshow(pi_grey)
plt.title("Original Pi")
plt.axis("off")

plt.subplot(rows,columns,2)
plt.plot(pi_grey_histogram)
plt.title("Original Pi Histogram")

plt.subplot(rows,columns,3)
plt.imshow(pi_grey_shift)
plt.title("Shifted Pi")
plt.axis("off")

plt.subplot(rows,columns,4)
plt.plot(pi_grey_shift_histogram)
plt.title("Shifted Pi Histogram")

plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------
#6. Apply histogram equalization on gray image. Show the original image ,the histogram of the original image, the image
# after equalization and its histogram in the same figure.
import cv2
import matplotlib.pyplot as plt

pi_grey = cv2.imread("pi.jpg", 0)
pi_grey_histogram = cv2.calcHist([pi_grey],[0],None,[256],[0,256])
pi_grey_equalized = cv2.equalizeHist(pi_grey)
pi_grey_equalized_histogram = cv2.calcHist([pi_grey_equalized],[0],None,[256],[0,256])

rows = 2
columns = 2
plt.subplot(rows,columns,1)
plt.imshow(pi_grey)
plt.title("Original Pi")
plt.axis("off")

plt.subplot(rows,columns,2)
plt.plot(pi_grey_histogram)
plt.title("Original Pi Histogram")

plt.subplot(rows,columns,3)
plt.imshow(pi_grey_equalized)
plt.title("Equalized Pi")
plt.axis("off")

plt.subplot(rows,columns,4)
plt.plot(pi_grey_equalized_histogram)
plt.title("Equalized Pi Histogram")

plt.show()
cv2.waitKey()
cv2.destroyAllWindows()