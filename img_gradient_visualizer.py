import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

#Reading the image as a 3D numpy array, since it is a colored image
x = cv2.imread("nf.tif")
#Converting 3D array to 2D, i.e, RBG image --> Grayscale image
img = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
#Changing the datatype from uint8 to a bigger value
img=img.astype(np.float32)
#Creating a copy of image to do operations without modifying original
gradient_magnitude = img.copy()
gradient_direction = img.copy()

#Finding the dimensions of the image
print(img.shape)
#Above command returns a tuple of two values (height,width)
#Now we are assigning height and width to seperate variables
height = img.shape[0]
width = img.shape[1]

#Kernel convolution via sobel operator
"""Kernel convolution is the process of taking a small matrix (say 3x3)
 and placing it on all of image's pixels one by one.
 This small matrix is called the kernel.
 The kernel contains the weight we want to give to surrounding pixels and center pixel,
 while performing operations on the center pixel.
 
 Kernels can be used for a variety of operations and thus come in various arguments.
 There are normalization kernels, identity kernel, mean calculator kernel, etc.
 
 In our example, we use the sobel operator.
 This kernel helps in edge detection in images."""
 
#Kernel for detecting horizontal edge
weight_x = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])

"""The array of zeros in the sobel operator denotes the orientation of suspected edge."""

#Kernel for detecting vertical edge
weight_y = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

#The two Outer loops, loop over each pixel in the image. (Except boundary pixels)
#The two Outer loops fix the kernel over that pixel, such that the pixel is in center of the kernel. 
for i in np.arange(1, height-1):
    for j in np.arange(1, width-1):
        sum1 = 0.0; sum2 = 0.0;
        #Inner two loops, loop over each weight value in the weight matrices...
        #...and multiply surrounding pixel values to these weights
        #Then two sums are calculated-->(one for each weight matrix)
        """These sums are equal to x and y components of the gradient vector."""
        for s in np.arange(0,3):
            for t in np.arange(0,3):

                p1 = img[i+s-1, j+t-1]
                p2 = weight_x[s,t]
                p3 = weight_y[s,t]
                
                sum1 = sum1 + p1*p2
                sum2 = sum2 + p1*p3
        dx = sum1
        dy = sum2

        magnitude = ((dx*dx)+(dy*dy))**0.5
        angle = math.atan(dy/dx) if dx != 0 else math.atan(10000)
        gradient_magnitude[i, j]= magnitude
        gradient_direction[i,j] = angle
        
#plotting the vectors on the image
img=np.flip(img,0)
dy, dx = np.gradient(img)
skip = (slice(None, None, 3), slice(None, None, 3))
y, x = np.mgrid[0:height:512j, 0:width:512j]
fig, ax = plt.subplots()
im = ax.imshow(gradient_direction, extent=[x.min(), x.max(), y.min(), y.max()])
ax.quiver(x[skip], y[skip], dx[skip], dy[skip])

ax.set(aspect=1, title='Quiver Plot')
plt.show()
"""#Showing the edges in the image by sir
cv2.imwrite('gradient.tif', gradient_magnitude)
cv2.imshow('Gradient', gradient_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()"""