"""
Course Number: ENGR 13300
Semester: Fall 2025

Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     11.2.2 Tp1 team 2
    Team ID:        LC4 - 18 
    Author:         Justin Chen, chen5731@purdue.edu
                    Arav, 
                    Gina,
                    Ayona
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

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#intakes an np array with the image
def clean_image(array):
    print(f"Image shape before cleaning: {array.shape}")
    
    aspect_ratio = len(array[0]) / len(array)
    new_width = 100
    new_height = 100
    if aspect_ratio < 1:
        new_height /= aspect_ratio
    elif aspect_ratio > 1:
        new_width /= 100
    image = Image.fromarray(array).resize(size=[new_height, new_width], resample=2)

    print(f"Resized image to: ({new_height}, {new_width})")
    Image.ImageOps.pad(image=image, size=[100, 100], color="black", centering=(0.5, 0.5))

    output_array = np.array(image)
    print(f"Image shape after cleaning: {output_array.shape}")
    return output_array

def load_img(path):
    img=Image.open(path)
    img_array=np.array(img)
    if img_array.ndim==4:
        img_array=img_array[:,:,3]
    if img_array.dtype!=np.uint8:
        img_array=img_array.astype(np.uint8)
    return img_array

def main():
    path = input("Enter the path of the image you want to clean: ")
    img_array = load_img(path)
    output_img = clean_image(img_array)
    plt.imshow(output_img)

if __name__ == "__main__":
    main()
