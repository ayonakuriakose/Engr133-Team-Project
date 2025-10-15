"""
Course Number: ENGR 13300
Semester: Fall 2025

Description:
    Cleans an image by resizing + adding padding to the sides to prepare it for feature extraction and classification.

Assignment Information:
    Assignment:     11.2.2 Tp1 team 2
    Team ID:        LC4 - 18 
    Author:         Justin Chen, chen5731@purdue.edu
                    Arav Srivastava, sriva222@dhaliwag@purdue.edu
                    Gina Dhaliwal, dhaliwag@purdue.edu
                    Ayona Kuriakose, akuriak@purdue.edu
    Date:           10/10/2025

Contributors:
    Name, login@purdue [repeat for each]

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

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# takes a numpy image array and cleans/resizes it
def clean_image(array):
    print(f"Image shape before cleaning: {array.shape}")  # display original shape

    aspect_ratio = len(array[0]) / len(array)  # calculate width-to-height ratio
    new_width = 100  # base width for resizing
    new_height = 100  # base height for resizing

    # adjust dimensions based on aspect ratio
    if aspect_ratio < 1:  
        new_width *= aspect_ratio  # narrow image: scale width down
    elif aspect_ratio > 1:
        new_height /= aspect_ratio  # wide image: scale height down

    # convert dimensions to integers
    new_height = int(new_height)
    new_width = int(new_width)

    # resize image 
    image = Image.fromarray(array).resize(size=[new_width, new_height], resample=2)
    print(f"Resized image to: ({new_height}, {new_width})")

    # pad image to 100×100 with black borders, centered at middle
    image = ImageOps.pad(image=image, size=[100, 100], color="black", centering=(0.5, 0.5))

    # convert processed image back to numpy array
    output_array = np.array(image)
    print(f"Image shape after cleaning: {output_array.shape}")

    return output_array  # return cleaned 100×100 image array


# loads an image file and converts it to a numpy array
def load_img(path):
    img = Image.open(path)  # open image from given path
    img_array = np.array(img)  # convert to numpy array

    # reshapes array to only have 3 values (removes alpha value, if exists)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # ensure pixel values are in uint8 format
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    return img_array  # return 3-channel uint8 array


# main function for running the image cleaning process
def main():
    path = input("Enter the path of the image you want to clean: ")  # ask user for image path
    img_array = load_img(path)  # load image into array
    output_img = clean_image(img_array)  # clean and resize image
    plt.imshow(output_img)  # display cleaned image
    plt.axis("off")  # hide axis labels
    plt.show()  # render image display

if __name__ == "__main__":
    main()
