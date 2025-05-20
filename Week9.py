#**************************************************shift on grayscale image****************************************************
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread(r'orange.png', cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(image)
f_magnitude = np.log(np.abs(f) + 1)

fshift = np.fft.fftshift(f)
fshift_magnitude = np.log(np.abs(fshift) + 1)

plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(f_magnitude, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(fshift_magnitude, cmap='gray')
plt.tight_layout()
plt.show()

#******************************************************HPF************************************************
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("orange.png", cv2.IMREAD_GRAYSCALE)
M, N = img.shape
F = np.fft.fft2(img)
F_shifted = np.fft.fftshift(F)
D0 = 20
H_ilpf = np.zeros((M, N), dtype=np.float32)

for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        if D > D0:
            H_ilpf[u, v] = 1

        else:
            H_ilpf[u, v] = 0
F_filtered = F_shifted * H_ilpf
F_ishift = np.fft.ifftshift(F_filtered)
img_back = np.fft.ifft2(F_ishift)
img_back = np.abs(img_back)
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(H_ilpf, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(img_back, cmap='gray')
plt.tight_layout()
plt.show()

#***************************************************LPF***************************************************
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"orange.png", cv2.IMREAD_GRAYSCALE)
M, N = img.shape
F = np.fft.fft2(img)
F_shifted = np.fft.fftshift(F)
D0 = 20
H_ilpf = np.zeros((M, N), dtype=np.float32)
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        if D <= D0:
            H_ilpf[u, v] = 1

        else:
            H_ilpf[u, v] = 0
F_filtered = F_shifted * H_ilpf
F_ishift = np.fft.ifftshift(F_filtered)
img_back = np.fft.ifft2(F_ishift)
img_back = np.abs(img_back)
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(H_ilpf, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(img_back, cmap='gray')
plt.tight_layout()
plt.show()

#************************************************Gaussian High-Pass Filter (GHPF)******************************************************
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread(r"orange.jpeg", cv2.IMREAD_GRAYSCALE)
M, N = img.shape
F = np.fft.fft2(img)
F_shifted = np.fft.fftshift(F)
D0 = 20
H_ghpf = np.zeros((M, N), dtype=np.float32)
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
        H_ghpf[u, v] = 1 - np.exp(-(D*2) / (2 * (D0*2)))
F_filtered = F_shifted * H_ghpf
F_ishift = np.fft.ifftshift(F_filtered)
img_back = np.fft.ifft2(F_ishift)
img_back = np.abs(img_back)
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(H_ghpf, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(img_back, cmap='gray')
plt.tight_layout()
plt.show()
#*************************************************Gaussian Low-Pass Filter (GLPF)،*****************************************************
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"orange.jpeg", cv2.IMREAD_GRAYSCALE)
M, N = img.shape

F = np.fft.fft2(img)
F_shifted = np.fft.fftshift(F)

D0 = 20
H_glpf = np.zeros((M, N), dtype=np.float32)

for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
        H_glpf[u, v] = np.exp(-(D*2) / (2 * (D0*2)))

F_filtered = F_shifted * H_glpf

F_ishift = np.fft.ifftshift(F_filtered)
img_back = np.fft.ifft2(F_ishift)
img_back = np.abs(img_back)
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(H_glpf, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(img_back, cmap='gray')
plt.tight_layout()
plt.show()
#******************************************************************************************************