"""
Course Number: ENGR 13300
Semester: Fall 2025

Description:
    Implements a sobel filter for edge detection

Assignment Information:
    Assignment:     11.2.2 tp2 team 2
    Team ID:        LC4 - 18 
    Author:         Arav Srivastava, sriva222@purdue.edu
                    Justin Chen, chen5731@purdue.edu
                    Gina Dhaliwal dhaliwag@purdue.edu
                    Ayona Kuriakose, akuriak@purdue.edu
    Date:           10/10/2025

Contributors:
    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

# takes in blurred grayscale image
# returns uint8 numpy array of sobel-filtered image
def sobel_filter(image_array):
    # pad edges with zeros and convert to float
    image_array = np.pad(image_array, pad_width=1, mode='constant', constant_values=0).astype(float)
    # create working copy of image as float array
    data_array = np.array(image_array, dtype=float)
    # duplicate arrays for x and y gradient calculations
    data_x = np.array(data_array)
    data_y = np.array(data_array)

    # define sobel operator kernels for x and y gradients
    sobel_x = [[-1, 0, 1], 
               [-2, 0, 2], 
               [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], 
               [0, 0, 0], 
               [1, 2, 1]]

    # apply sobel filter to each pixel (excluding padded border)
    for row in range(1, len(data_array) - 1):
        for col in range(1, len(data_array[0]) - 1):
            # extract 3x3 neighborhood around current pixel
            kernel = image_array[row-1:row+2, col-1:col+2]
            # compute gradient in x and y directions
            data_x[row][col] = np.sum(sobel_x * kernel)
            data_y[row][col] = np.sum(sobel_y * kernel)

    # compute gradient magnitude
    data_array = np.sqrt(data_x**2 + data_y**2)
    # clip values to 0–255 range
    data_array = np.clip(data_array, 0, 255)
    # remove padding
    data_array = data_array[1:-1, 1:-1]

    # apply threshold to highlight edges
    for row in range(len(data_array)):
        for col in range(len(data_array[0])):
            threshold = 50
            if data_array[row, col] >= threshold:
                data_array[row, col] = 255 
            else:
                data_array[row, col] = 0

    # return final edge-detected image as uint8 array
    return data_array.astype(np.uint8)
            
 # creating UDF for loading the image
def load_img(path):
    # opening the image selected by user
    img=Image.open(path)
    # making image RGB into array, and normalizes  
    img_array=np.array(img).astype(np.float64)/255.0
    #Linearlize the pixel values
    lin_array=linearize(img_array)
    # if statement for if the lineraized array has 4 dimensions, then convert to 3 (drops the alpha dimension)
    if lin_array.shape[-1]==4:
        lin_array=lin_array[:,:,:3]
    # denormalizes and returns the array 
    lin_array=(lin_array*255.0).astype(np.uint8)
    return lin_array

# UDF for linearization process. uses nested for loop to get row and column indexing
def linearize(img_array):
    #Linearize the pixels for greyscale 
    if img_array.ndim==2:
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                c_prime=img_array[i,j]
                if c_prime<=0.04045:
                    img_array[i,j]=c_prime/12.92
                else:
                    img_array[i,j]=((c_prime+0.055)/1.055)**2.4
    
    #Linearize the pixels for color
    else:
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                for k in range(img_array.shape[2]):
                    c_prime=img_array[i,j,k]
                    if c_prime<=0.04045:
                        img_array[i,j,k]=c_prime/12.92
                    else:
                        img_array[i,j,k]=((c_prime+0.055)/1.055)**2.4
    return img_array

#UDF to convert a color image to a grayscale image
def rgb_to_grayscale(lin_array):
    # making RGB into grayscale
    gray_array=(0.2126*lin_array[:,:,0])+(0.7152*lin_array[:,:,1])+(0.0722*lin_array[:,:,2])
    return gray_array.astype(np.uint8)

#UDF to blur a grayscale image using a gaussian filter
def gaussian_filter(gray_array,sigma):
    #Makes sure sigma is > 0
    sigma=math.ceil(sigma)
    #Finds kernel size using given formula
    kernel_size=int(6*sigma+1)
    #Finds the size for the grid
    k=kernel_size//2
    #x values from -k to k from the center
    x=np.linspace(-k,k,kernel_size)
    #y vlaues from -k to k from the center
    y=np.linspace(-k,k,kernel_size)
    # relative x and y values for each position in the kernel
    X,Y=np.meshgrid(x,y)
    #Gaussian fucntion
    gaus_xy=(1/(2*np.pi*sigma**2))*(np.exp(-(((X**2)+(Y**2))/(2*sigma**2))))
    #Normalize kernel
    kernel=gaus_xy/np.sum(gaus_xy)
    #gets the height and width of the array
    height,width=np.shape(gray_array)
    #creates a zero array same size as gray array
    blurred=np.zeros_like(gray_array)
    #pads the edges of gray array
    padded=np.pad(gray_array,k)
    #loop to blur each pixel
    #loops through all the rows
    for x in range(gray_array.shape[0]):
        #loops through all the columns
        for y in range(gray_array.shape[1]):
            #finds the 3x3 region surounding the pixel
            region=padded[x:x+kernel_size,y:y+kernel_size]
            #blurred value for each pixel
            blurred[x,y]=np.sum(region*kernel)
    #returns blurred array
    return blurred.astype(np.uint8)

def main():
    #user input for path to image
    path = input("Enter the path to the image file: ")
    #loads image and linearlizes and normalizes the image
    image=load_img(path)
    #converts color image to a grayscale image
    gray_array=rgb_to_grayscale(image)
    #applies the gaussian filter to the grayscale image
    blurred_img=gaussian_filter(gray_array, sigma=1)

    #applies the sobel filter to the blurred image
    sobel_img=sobel_filter(blurred_img)

    #shows the blurred image and applies gray cmap to the image
    plt.imshow(sobel_img, cmap='gray')
    #makes sure no axes are showing in the output
    plt.axis("off")
    #shows the final gray blurred image 
    plt.show()

if __name__ == "__main__":
    main()


