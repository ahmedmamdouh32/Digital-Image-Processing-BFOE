from PIL import Image
strawberry = Image.open("strawberry.png") #to load image into object 'strawberry'

strawberry.show() #to show image

#***************************************************** Image Pixels ****************************************************
strawberry_pixels = strawberry.load() #to load pixels values of strawberry image
strawberry_pixels[30,50] #access pixels values at coordinates (x = 30, y = 50)
print(strawberry_pixels[30,50]) #--> (229, 236, 173, 0) it prints 4 values as image is in RGBA mode
#***********************************************************************************************************************


#***************************************************** Image Format ****************************************************
strawberry.format #returns the format of image (PNG,JPG,JPEG,...)
print(strawberry.format) #-->PNG
#***********************************************************************************************************************


#****************************************************** Image Size *****************************************************
strawberry.size #returns width and height of image
print(strawberry.size)   #-->(600, 346)
print(strawberry.width)  #-->600
print(strawberry.height) #-->346
#***********************************************************************************************************************


#***************************************************** Image Modes *****************************************************
strawberry.mode #the mode where the image in (RGB,RGBA,L,1,CMYK,...)
print(strawberry.mode) #-->RGBA
#***********************************************************************************************************************


#***************************************************** Image Crop ******************************************************
strawberry.crop((0,0,200,200)) #crops the image, takes the upper left coordinates(0,0) and lower right
                               #coordinates(200,200) in a single tuple
cropped_strawberry = strawberry.crop((0,0,300,346)) #this nearly crops to half the image
#***********************************************************************************************************************


#**************************************************** Image Resize *****************************************************
resized_strawberry = strawberry.resize((100,100)) #to give the image specific width and height
resized_strawberry.show() #image shown here is not as original, because the ratio between width and height has changed
resized_strawberry = strawberry.resize((strawberry.width//2, strawberry.height//2)) #this keeps the ratio w/h constant
resized_strawberry.show() #we used '//' not '/' in [previous line] so we can floor division result to integer value
resized_strawberry.reduce(2) #this code makes the same resize with keeping the ratio w/h unchanged
#***********************************************************************************************************************


#***************************************************** Save Image ******************************************************
resized_strawberry.save("resized_strawberry.png") #taking the full path of the image and its format to save it
#***********************************************************************************************************************


#***************************************************** Transpose *******************************************************
trans_strawberry = strawberry.transpose(Image.FLIP_TOP_BOTTOM)
trans_strawberry = strawberry.transpose(Image.FLIP_LEFT_RIGHT)
trans_strawberry = strawberry.transpose(Image.ROTATE_90)
trans_strawberry = strawberry.transpose(Image.ROTATE_180)
trans_strawberry = strawberry.transpose(Image.ROTATE_270)
trans_strawberry = strawberry.transpose(Image.TRANSPOSE)
trans_strawberry = strawberry.transpose(Image.TRANSVERSE)

#difference between transpose and transverse
#Transpose :
# A B C            A D G
# D E F    --->    B E H
# G H I            C F I

#Transverse :
# A B C            I F C
# D E F    --->    H E B
# G H I            G D A
#***********************************************************************************************************************


#*********************************************** Some useful function **************************************************
strawberry.rotate(90,expand=True) #to rotate the image, expand to rotate with the original size
strawberry.getbands() #returns a tuple contains the separate layers of colors of the image
print(strawberry.getbands()) #-->('R', 'G', 'B', 'A')
strawberry.convert('RGB') #converting image mood
R,G,B,A = strawberry.split()  #splits all the layers
reconstructed_strawberry = Image.merge("RGBA",(R,G,B,A)) #recreate the image again after splitting it
#***********************************************************************************************************************



#****************************************************** Sheet 3 ********************************************************
# 1. By using pillow library read an image and get the
# following information:
# a)Type of image  b)Name of image  c)Mode of image  d)Size of image  e)Format of image  f) Pixel’s values of image

from PIL import Image
import numpy as np
strawberry = Image.open("strawberry.png")
#a)
type(strawberry)
#b)
strawberry.filename
#c)
strawberry.mode
#d)
strawberry.size
#e)
strawberry.format
#f)
strawberry.getpixel((0,0)) #values of a single pixel
np.array(strawberry) #values of all pixels
#-----------------------------------------------------------------------------------------------------------------------
# 2. By Using functions of pillow library perform the following manipulations on the image, and plot all
# images in the same figure:
# a)Crop the image.  b)Resample the image using resize function and reduce function.

from PIL import Image
import matplotlib.pyplot as plt

strawberry = Image.open("strawberry.png")

#a)
cropped_strawberry = strawberry.crop((0,0,350,360))
#b)
resampled_strawberry = strawberry.resize((strawberry.width//2,strawberry.height//2))
reduced_strawberry = strawberry.reduce(2)

figure_all_in_one = plt.figure(figsize=(6,8)) #image that will contain all images
rows = 2
columns = 2

plt.subplot(rows,columns,1)
plt.imshow(strawberry)
plt.title("Original strawberry")
plt.axis("off")

plt.subplot(rows,columns,2)
plt.imshow(cropped_strawberry)
plt.title("Cropped strawberry ")
plt.axis("off")

plt.subplot(rows,columns,3)
plt.imshow(resampled_strawberry)
plt.title("Resampled strawberry")
plt.axis("off")

plt.subplot(rows,columns,4)
plt.imshow(reduced_strawberry)
plt.title("Reduced strawberry")
plt.axis("off")
plt.show()
#-----------------------------------------------------------------------------------------------------------------------
# 3. Save the cropped image.

from PIL import Image

strawberry = Image.open("strawberry.png")
cropped_strawberry = strawberry.crop((0,0,300,300))
cropped_strawberry.save(f"cropped_strawberry.{strawberry.format}") #saves the image according to format of original one
#-----------------------------------------------------------------------------------------------------------------------
# 4. Perform the following transformations using transpose function and plot all images in the same figure:
# a)Flip the image left to right.  b)Flip the image top to bottom.  c)rotate the image 90,180 and 270.
# d)Transposes the rows and columns using the top-left pixel as the origin.
# e)Transposes the rows and columns using the bottom-left pixel as the origin.

from PIL import Image
import matplotlib.pyplot as plt

strawberry = Image.open("strawberry.png")
#a)
left_to_right_strawberry = strawberry.transpose(Image.FLIP_LEFT_RIGHT)
#b)
top_to_bottom_strawberry = strawberry.transpose(Image.FLIP_LEFT_RIGHT)
#c)
rotate_90_strawberry = strawberry.transpose(Image.ROTATE_90)
rotate_180_strawberry = strawberry.transpose(Image.ROTATE_180)
rotate_270_strawberry = strawberry.transpose(Image.ROTATE_270)
#d)
transpose_strawberry = strawberry.transpose(Image.TRANSPOSE)
#e)
transverse_strawberry = strawberry.transpose(Image.TRANSVERSE)

figure_all_in_one = plt.figure(figsize=(6,8)) #image that will contain all images
rows = 2
columns = 4

plt.subplot(rows,columns,1)
plt.imshow(strawberry)
plt.title("Original strawberry")
plt.axis("off")

plt.subplot(rows,columns,2)
plt.imshow(left_to_right_strawberry)
plt.title("left_to_right_strawberry")
plt.axis("off")

plt.subplot(rows,columns,3)
plt.imshow(top_to_bottom_strawberry)
plt.title("top_to_bottom_strawberry")
plt.axis("off")

plt.subplot(rows,columns,4)
plt.imshow(rotate_90_strawberry)
plt.title("rotate_90_strawberry")
plt.axis("off")

plt.subplot(rows,columns,5)
plt.imshow(rotate_180_strawberry)
plt.title("rotate_180_strawberry")
plt.axis("off")

plt.subplot(rows,columns,6)
plt.imshow(rotate_270_strawberry)
plt.title("rotate_270_strawberry")
plt.axis("off")

plt.subplot(rows,columns,7)
plt.imshow(transpose_strawberry)
plt.title("transpose_strawberry")
plt.axis("off")

plt.subplot(rows,columns,8)
plt.imshow(transverse_strawberry)
plt.title("transverse_strawberry")
plt.axis("off")

plt.show()
#-----------------------------------------------------------------------------------------------------------------------
# 5. Rotate image using rotate function.

from PIL import Image
strawberry = Image.open("strawberry.png")
strawberry.rotate(45,expand=False).show()
strawberry.rotate(45,expand=True).show() #prevents cropping the image while rotating
#-----------------------------------------------------------------------------------------------------------------------
# 6. Print the bands of image.

from PIL import Image
strawberry = Image.open("strawberry.png")
print(strawberry.getbands()) #-->('R', 'G', 'B', 'A')
#-----------------------------------------------------------------------------------------------------------------------
# 7. Convert image to another modes.

from PIL import Image

strawberry = Image.open("strawberry.png")
RGB_strawberry = strawberry.convert("RGB")
L_strawberry = strawberry.convert("L")
#-----------------------------------------------------------------------------------------------------------------------
# 8. separate an image into its bands and plot each band individually.

from PIL import Image
import matplotlib.pyplot as plt

pi = Image.open("pi.jpg")
R,G,B = pi.split()

figure_all_in_one = plt.figure(figsize=(6,8)) #image that will contain all images
rows = 2
columns = 2

plt.subplot(rows,columns,1)
plt.imshow(pi)
plt.title("Original pi")
plt.axis("off")

plt.subplot(rows,columns,2)
plt.imshow(R)
plt.title("Red pi")
plt.axis("off")

plt.subplot(rows,columns,3)
plt.imshow(G)
plt.title("Green pi")
plt.axis("off")

plt.subplot(rows,columns,4)
plt.imshow(B)
plt.title("Blue pi")
plt.axis("off")

plt.show()