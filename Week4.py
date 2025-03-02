#************************************************ Linear Transformation ************************************************
# S = L - 1 - R
# S : Transformed (new) pixel value
# L : Maximum Intensity possible value [0-255] --> L = 256
# R : original pixel value
#***********************************************************************************************************************


#********************************************** Logarithmic Transformation *********************************************
# S = C log(1 + R)
# S : Transformed (new) pixel value
# R : Original pixel value
# C : A scaling constant (to adjust brightness)

#Law of C:
#       C = (L-1)/log(1+max(R))
#the value of L = 256 for 8-bit image, so the nominator = 255
#for the denominator, we examine each pixel in the image, select the highest value, add 1 to it,
#and then apply the logarithm.

import cv2
import numpy as np

pi_grey = cv2.imread("pi.jpg",flags = 0) #we used grey photo for simpler calculations(flag = 0)

c = 255 / np.log(1+np.max(pi_grey))
pi_grey_logarithmic = c * (np.log(1+pi_grey))
pi_grey_logarithmic = np.array(pi_grey_logarithmic,dtype = np.uint8) #the logarithmic generate floating point values
                                                                     #while images works with integer values so we
                                                                     #need to convert float to unsigned int 8-bit
cv2.imshow("pi grey original",pi_grey)
cv2.imshow("pi grey logarithmic", pi_grey_logarithmic)
cv2.waitKey()
cv2.destroyAllWindows()
#***********************************************************************************************************************


#******************************************** Power Law (Gamma)Transformation ******************************************
#S = C x R ^ γ
# S : Transformed (new) pixel value
# R : Original pixel value (normalized: 0 to 1)
# C : Scaling constant (optional, usually = 1)
# γ (Gamma) : The exponent that controls brightness & contrast

import cv2
import numpy as np
pi_grey = cv2.imread("pi.jpg",flags = 0) #we used grey photo for simpler calculations(flag = 0)
C = 1
gamma = 1.5 #random chosen value
gamma_corrected = np.array(255 * C * (pi_grey/255) ** gamma, dtype = np.uint8) #we made multiplication by 255 to return
                                                                               #to 8-bit scale, because C = 1 , R is
                                                                               #normalized [0,1] so we need to return
                                                                               #back to [0,255] scale

cv2.imshow("pi grey original",gamma_corrected)
cv2.waitKey()
cv2.destroyAllWindows()
#***********************************************************************************************************************


#******************************************************** Sheet 4 ******************************************************
#1. By using open cv library read colored image, convert it to gray image and save it.

import cv2

pi_cv2 = cv2.imread(r"pi.jpg", flags = 0)
cv2.imshow("pi",pi_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------
#2. By using pillow library read colored image, convert it to gray image and save it.

from PIL import Image

pi_PIL = Image.open("pi.jpg")
pi_PIL_grey = pi_PIL.convert("L")
pi_PIL_grey.show()
pi_PIL_grey.save(f"pi_grey.{pi_PIL.format}")
#-----------------------------------------------------------------------------------------------------------------------
#3. Read a gray image and apply the log transformation on the image. show the original and transformation images.

import numpy as np
import cv2

pi_original = cv2.imread("pi_grey.JPEG")

c = 255 / np.log(1+np.max(pi_original))
pi_log = c * (np.log(1+pi_original))
pi_log = np.array(pi_log,dtype = np.uint8) #to convert it to integer


cv2.imshow("original image",pi_original)
cv2.imshow("log image",pi_log)
cv2.waitKey(0)
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------
#4. Read an image in gray scale level and apply the Power-law (gamma) transformation with different gamma [0.1, 0.5, 1.2
# ,2.2] on the image. show the original and transformation images in the same figure.
import numpy as np
import matplotlib.pyplot as plt

figure_all_in_one = plt.figure(figsize=(6,8))
rows = 2
columns = 2

pi_grey_original = cv2.imread("pi_grey.JPEG")
gamma_values = [0.1, 0.5, 1.2, 2.2]

i = 1

for gamma in gamma_values:
    gamma_corrected = np.array(255 * (pi_grey_original/255) ** gamma, dtype = np.uint8)
    plt.subplot(rows, columns, i)
    i = i + 1
    plt.imshow(gamma_corrected,cmap="grey")
    plt.title(f"Gamma : {gamma}")
    plt.axis("off")

plt.show()