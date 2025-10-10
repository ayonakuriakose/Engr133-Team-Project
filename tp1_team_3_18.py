"""
Course Number: ENGR 13300
Semester: Fall 2025

Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     11.2.2 tp1 team 3
    Team ID:        LC4 - 18 
    Author:         Arav Srivastava, sriva222@purdue.edu
                    Justin, 
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


def rgb_to_hsv(red, green, blue):

    r_prime = red / 255
    g_prime = green / 255
    b_prime = blue / 255

    C_max = max(r_prime, g_prime, b_prime)
    C_min = min(r_prime, g_prime, b_prime)
    delta = C_max - C_min

    if delta == 0:
        H_prime = 0
    elif C_max == r_prime:
        H_prime = (60 * (g_prime - b_prime) / delta) % 360
    elif C_max == g_prime:
        H_prime = ((60 * (b_prime - r_prime) / delta) + 120) % 360
    else:
        H_prime = ((60 * (r_prime - g_prime) / delta) + 240) % 360


    if C_max == 0:
        S_prime = 0
    else:
        S_prime = (delta / C_max)

    V_prime = C_max

    h = (H_prime / 360) * 255
    s = S_prime * 255
    v = V_prime * 255

    return h, s, v

def convert_to_hsv(rgb_image):


    return


def cleanImage(array):
    print(f"Image shape before cleaning: {array.shape}")
    
    aspectRatio = len(array[0]) / len(array)
    newWidth = 100
    newHeight = 100
    if aspectRatio < 1:
        newHeight /= aspectRatio
    elif aspectRatio > 1:
        newWidth /= 100
    image = Image.resize(size=[newHeight, newWidth], resample=2)

    print(f"Resized image to: ({newHeight}, {newWidth})")
    Image.ImageOps.pad(image=image, size=[100, 100], color="black", centering=(0.5, 0.5))

    outputArray = np.array(image)
    print(f"Image shape after cleaning: {outputArray.shape}")
    return outputArray

def main():
    
    image_path = str(int("Enter the path of the image you want to convert to hsv: "))
    cleanImage(image_path)

if __name__ == "__main__":
    main()
