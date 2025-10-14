"""
Course Number: ENGR 13300
Semester: Fall 2025

Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     11.2.1 Tp1 team 1
    Team ID:        LC4 - 18 
    Author:         Justin Chen, chen5731@purdue.edu
                    Arav Srivastava, sriva222@dhaliwag@purdue.edu
                    Gina Dhaliwal, dhaliwag@purdue.edu
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
# importing necessary libraries for use in code
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# creating UDF for loading the image
def load_img(path):
    # opening the image selected by user
    img=Image.open(path)
    # making image RGB into array
    img_array=np.array(img)
    # if statement for if the image array has 4 dimensions, then convert to 3
    if img_array.shape[-1]==4:
        img_array=img_array[:,:,:3]
    # if statement to make sure the img_array is uint8
    if img_array.dtype!=np.uint8:
        img_array=img_array.astype(np.uint8)
    # returns img_array now following our goal plans
    return img_array
# creating another UDF for converting image into grayscale image

def rgb_to_grayscale(img_array):
    # normalizing array
    norm_array=img_array/255.0
    # making RGB into grayscale
    gray_array=(0.2126*norm_array[:,:,0])+(0.7152*norm_array[:,:,1])+(0.0722*norm_array[:,:,2])
    # denormalizing, brings back to being array values between 0 and 255
    gray_array=(gray_array*255.0).astype(np.uint8)
    return gray_array
    
# main UDF
def main():
    # user input for path of image
    path = input("Enter the path of the image you want to load: ")
    # loading image 
    image = load_img(path)
    
    if image.ndim==2:
        plt.imshow(image,cmap="gray")
        plt.axis("off")
        plt.show()
    else:
        ask_user=input("Would you like to convert to grayscale?\n")
        if ask_user=="yes":
            gray_image=rgb_to_grayscale(image)
            plt.imshow(gray_image,cmap="gray")
            plt.axis("off")
            plt.show()
        else:
            plt.imshow(image)
            plt.axis("off")
            plt.show()

if __name__ == "__main__":
    main()
