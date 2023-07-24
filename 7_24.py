import cv2 as cv
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
mpl.rc('image', cmap='gray')
import math

def imshow(img):
    plt.imshow(img)
    plt.colorbar()
    # plt.axis('off')
    plt.show()

def time_step(n = 25):
    # image size
    size = 100
    delta = size / n
    return delta

def heat_kernel(kernel_size, delta):
    """Create a kernel using heat equation with input size and deviation"""
    # Create kernel maxtrix with input size
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = np.exp(-(i**2 + j**2)/4)/(4*np.pi*delta**2)
    return kernel/np.sum(kernel)

def heat_kernel_convolution(image, kernel):
    """Compute heat kernel convolution on input image"""
    # Normalize the image
    image = 1-image/255
    imshow(image)
    # Apply kernel convolution to image
    i,j = kernel.shape
    padding = i//2
    inputs = np.pad(image, (padding,padding), constant_values=0)
    heat_convoluted = np.zeros(shape=image.shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):  
            heat_convoluted[x, y] = np.sum(np.multiply(inputs[x: x + i, y: y + j], kernel))

    # heat_convoluted = cv.filter2D(image, -1, kernel=kernel)
    heat_convoluted = heat_convoluted.astype(float)

    # Set values above 0.5 to 1 and below to 0
    heat_convoluted[heat_convoluted > 0.5] = 1
    heat_convoluted[heat_convoluted <= 0.5] = 0

    return heat_convoluted

def heat_diffusion(image, kernel, lapse):
  """Apply heat diffusion on image over a period"""
  images = [image]
  # imshow(image)
  for i in range(1,lapse):
    conv_img =  heat_kernel_convolution(images[i-1], kernel)
    #append the new convoluted image 
    images.append(conv_img)
  imshow(images[lapse-1])
  return images

#TEST
img = cv.imread("disk.jpeg", 0)

IMG_HEIGHT, IMG_WIDTH = 100,100
img = cv.resize(img,(IMG_HEIGHT, IMG_WIDTH))
print(img.shape)

delta = time_step(400)
kernel = heat_kernel(50, delta)
print(kernel)
imshow(img)
img2=heat_kernel_convolution(img,kernel)
# imshow(img2)
# print(img2)
images = heat_diffusion(img, kernel,2)
