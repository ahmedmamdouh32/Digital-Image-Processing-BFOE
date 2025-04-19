# 1.By using open cv library read an image and apply the following filters:
#1)bilateralFilter, 2)medianBlur, 3)GaussianBlur, 4)blur, 5)cv2.boxFilter, 6)laplacian
import cv2

image = cv2.imread('orange.png')
Bilateral_image = cv2.bilateralFilter(image,20,75,75)
median_blurred_image = cv2.medianBlur(image, ksize=9)
gaussian_image = cv2.GaussianBlur(image, (5, 5), sigmaX=10)
blurred_image = cv2.blur(image, (5, 5))
box_filter_image = cv2.boxFilter(image, ddepth=-1, ksize=(5, 5), normalize=True)
laplacian_image = cv2.Laplacian(image, ddepth=cv2.CV_64F)


# Show the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Bilateral Filtered Image', Bilateral_image)
cv2.imshow('Median Blured Filtered Image', median_blurred_image)
cv2.imshow('Gaussian Filtered Image', gaussian_image)
cv2.imshow('Blurred Filtered Image', blurred_image)
cv2.imshow('Box Filtered Image', box_filter_image)
cv2.imshow('Laplacian Filtered Image', laplacian_image)

# Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------
# 2-By Using cv2.filter2D() and numpy apply any kernel to the image.
import cv2
import numpy as np

image = cv2.imread('orange.png')

kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0,-1, 0]], dtype=np.float32)

filtered_image = cv2.filter2D(src=image , ddepth=-1 , kernel=kernel)

cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------
# 3- using the cv2.getGaussianKernel().apply the output gaussian kernel to the image.
import cv2
import numpy as np

image = cv2.imread('orange.png')

gaussian_kernel_1d = cv2.getGaussianKernel(5, 3)

gaussian_kernel_2d = gaussian_kernel_1d @ gaussian_kernel_1d.T

blurred_image = cv2.filter2D(image, -1, gaussian_kernel_2d)

cv2.imshow('Original Image', image)
cv2.imshow('Gaussian Blurred Image', blurred_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()