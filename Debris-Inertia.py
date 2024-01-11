import pandas as pd 
import numpy as np 
import math 
import os 
import cv2
from astropy.io import fits
import matplotlib.pyplot as plt
from PIL import Image
from math import *

def convert_fits_to_image(fits_filename, output_image_filename):
    # Open the FITS file
    with fits.open(fits_filename) as hdul:
        # Get the data from the primary HDU (Header Data Unit)
        data = hdul[0].data

        # Noise reduction using Gaussian filter
        data = cv2.GaussianBlur(data, (5, 5), 0)

        # Sharpening using Laplacian filter
        laplacian = cv2.Laplacian(data, cv2.CV_64F)
        sharpened = data - 0.8 * laplacian

        # Plot the data as an image without grid
        plt.imshow(sharpened, cmap='gray')
        plt.axis('off')  # Turn off the axis (including grid)

        # Save the preprocessed image as a PNG file
        plt.savefig(output_image_filename, bbox_inches='tight', pad_inches=0)
        plt.close()

def iterative_thresholding(image, initial_threshold=128, max_iterations=50, tolerance=1e-3):
    threshold = initial_threshold

    for iteration in range(max_iterations):
        # Segment the image into foreground and background based on the threshold
        foreground = image >= threshold
        background = image < threshold

        # Compute the mean intensity of each group
        foreground_mean = np.mean(image[foreground])
        background_mean = np.mean(image[background])

        # Compute the new threshold as the average of the means
        new_threshold = (foreground_mean + background_mean) / 2.0

        # Check for convergence
        if abs(new_threshold - threshold) < tolerance:
            break

        threshold = new_threshold

    return threshold
def momentOfInertia (xWidth, yHeight, xCG, yCG): #how to take output mn function we ahoto fe func tanya?
      """
      
2nd: calculate the moment of inertia 
    Ixx= summations (y-yCG)^2 and 
    Iyy= summation (x-xCG)^2 

3rd: calculate the moment of inertia of
    Ixy= Ixx*Iyy

        """
      Ixx =sum ((y-yCG)**2 for y in yHeight)
      Iyy= sum ((x- xCG)**2 for x in xWidth)
      Ixy=sum((x-xCG)*(y-yCG) for x,y in zip(xWidth,yHeight))
        

      return Ixx, Iyy, Ixy

def mainInteria(Ixx,Iyy,Ixy,yHeight,xWidth):
     """ 
     4th: calculate the principal interial moments 
    Imain1= 1/2 *(Ixx+Iyy+ sqrt((Ixx-Iyy)^2+ 4*(Ixy)^2)) 
    Imain2= 1/2 *(Ixx+Iyy- sqrt((Ixx-Iyy)^2+ 4*(Ixy)^2)) 

    Theta= -1/2 atan( 2*Ixy/Ixx-Iyy)
    
    """
     Imain1= 0.5 *(Ixx+Iyy+ sqrt((Ixx-Iyy)**2+ 4*(Ixy)**2))
     Imain2= 0.5 *(Ixx+Iyy- sqrt((Ixx-Iyy)**2+ 4*(Ixy)**2))
     
     epsilonn=10

     finalInteria= Imain1 / Imain2
     if finalInteria > epsilonn:
          print("This object is predicted to be debris")
     else:
          print("this object is predicted to be a Celestial objects")
     return finalInteria
# Directory containing FITS files
fits_directory = 'C:\Users\USER\Desktop\Space-Debris-Project\dataset'

# Output directory for PNG images
output_directory ='C:\Users\USER\Desktop\Space-Debris-Project\dataset\output_files'
# fits_filenames = ['NEOS_SCI_2022001030508.fits','NEOS_SCI_2022001030535.fits','NEOS_SCI_2022001030602.fits',
#                   'NEOS_SCI_2022001030629.fits','NEOS_SCI_2022001030656.fits','NEOS_SCI_2022001030723.fits',
#                   'NEOS_SCI_2022001030750.fits','NEOS_SCI_2022001030817.fits','NEOS_SCI_2022001030844.fits',
#                   'NEOS_SCI_2022001030911.fits','NEOS_SCI_2022001030938.fits','NEOS_SCI_2022001031005.fits',
#                   'NEOS_SCI_2022001031032.fits','NEOS_SCI_2022001031059.fits','NEOS_SCI_2022001031126.fits',
#                   'NEOS_SCI_2022001031153.fits','NEOS_SCI_2022001031220.fits','NEOS_SCI_2022001031247.fits',
#                   'NEOS_SCI_2022001031314.fits','NEOS_SCI_2022001031341.fits','NEOS_SCI_2022001031408.fits',
#                   'NEOS_SCI_2022001031435.fits','NEOS_SCI_2022001031502.fits','NEOS_SCI_2022001031529.fits',
#                   'NEOS_SCI_2022001031556.fits','NEOS_SCI_2022001031623.fits','NEOS_SCI_2022001031650.fits',
#                   'NEOS_SCI_2022001031717.fits','NEOS_SCI_2022001031744.fits','NEOS_SCI_2022001031811.fits',
#                   'NEOS_SCI_2022001031838.fits','NEOS_SCI_2022001031905.fits','NEOS_SCI_2022001031959.fits',
#                   'NEOS_SCI_2022001032026.fits','NEOS_SCI_2022001032053.fits']  # Add more filenames as needed

fits_filenames = ['please.fits','please4.fits','please5.fits','please6.fits','please7.fits','tria.fits']  # Add more filenames as needed
for fits_filename in fits_filenames:
    # Full path to the FITS file
    full_path_fits = os.path.join(fits_directory, fits_filename)

    # Output PNG filename (assuming the same name with a different extension)
    output_image_filename = os.path.join(output_directory, os.path.splitext(fits_filename)[0] + '_preprocessed.png')
    convert_fits_to_image(full_path_fits, output_image_filename)

    image = cv2.imread(output_image_filename)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the iterative thresholding algorithm to the image
    optimal_threshold = iterative_thresholding(img)

    # Threshold the image using the optimal threshold
    thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

    num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)
    # for label in range(1, num_labels_iterative): 
    #   center_x, center_y = centroids_iterative[label] 
    #   object_id = 1
    # # Iterate through each detected object
    #   for i in range(1, num_labels_iterative):
    # # Get the coordinates of the bounding box for the current object
    #      x, y, w, h, area = stats_iterative[i]
    #      Ixx,Iyy,Ixy=momentOfInertia (w, h, center_x, center_y)

    for label in range(1, num_labels_iterative): 
            center_x, center_y = centroids_iterative[label]
            
            # Get the coordinates of the bounding box for the current object
            x, y, w, h, area = stats_iterative[label]
            
            # Ensure xWidth and yHeight are iterable (lists)
            xWidth = list(range(w))
            yHeight = list(range(h))
            
            Ixx, Iyy, Ixy = momentOfInertia(xWidth, yHeight, center_x, center_y)
            finalint=mainInteria(Ixx,Iyy,Ixy,yHeight,xWidth)