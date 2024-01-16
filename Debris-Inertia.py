import pandas as pd
import numpy as np
import math
import os
import cv2
from astropy.io import fits
import matplotlib.pyplot as plt
from PIL import Image
from math import *
import csv
from Convert_debris import convert_fits_to_image
from threshold import iterative_thresholding

 # YA SALMA : MERGE BAA BEL CODES EL TANYA, YA SHELE DOL WE IMPORT MN HETA TANYA WE HENA TEBAA EL INERTIA BAS
 # YA SALMA: aham haga fel merge en inertia tebaa akher haga baad kol el extractions// prepocessing
def momentOfInertia(xWidth, yHeight, xCG, yCG):
    Ixx = sum((y - yCG)**2 for y in yHeight)
    Iyy = sum((x - xCG)**2 for x in xWidth)
    Ixy = sum((x - xCG)*(y - yCG) for x, y in zip(xWidth, yHeight))

    return Ixx, Iyy, Ixy

def mainInteria(Ixx, Iyy, Ixy, yHeight, xWidth):
    Imain1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4*(Ixy)**2))
    Imain2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy)**2 + 4*(Ixy)**2))

    epsilonn = 10

    finalInteria = Imain1 / Imain2
    if finalInteria > epsilonn:
        print(f"This object  is predicted to be debris")
    else:
        print(f"This object  is predicted to be a Celestial object")

    return finalInteria

# Directory containing FITS files
fits_directory = '/content/drive/MyDrive/Colab-Debris'  # YA SALMA : EZBOTY PATH

# Output directory for PNG images
output_directory = '/content/drive/MyDrive/Colab-Debris/output_images' # YA SALMA : EZBOTY PATH


#csvfile
csv_file_path = '/content/drive/MyDrive/Colab-Debris/output_images/InetriaOutPut.csv' # YA SALMA : EZBOTY PATH
fits_filenames = ['space5.fits','tria.fits','please4.fits','space8.fits','space6.fits','space3.fits']  # Add more filenames as needed
# Open the CSV file in write mode
with open(csv_file_path, 'w', newline='') as csvfile:
    # Create a CSV writer
    csvwriter = csv.writer(csvfile)

    # Write the header row
    csvwriter.writerow(['Image', 'Object ID', 'Prediction'])

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

        num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = cv2.connectedComponentsWithStats(
            thresholded_img, connectivity=8)

        # Reset object_id for each new image
        object_id = 1

        for label in range(1, num_labels_iterative):
            center_x, center_y = centroids_iterative[label]

            # Get the coordinates of the bounding box for the current object
            x, y, w, h, area = stats_iterative[label]

            # Ensure xWidth and yHeight are iterable (lists)
            xWidth = list(range(w))
            yHeight = list(range(h))

            # Print the coordinates of the bounding box
            print(f"Object {object_id} in {fits_filename}:")

            cv2.putText(image, str(object_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Increment Id
            object_id += 1

            Ixx, Iyy, Ixy = momentOfInertia(xWidth, yHeight, center_x, center_y)
            finalint = mainInteria(Ixx, Iyy, Ixy, yHeight, xWidth)

            # Write the row to the CSV file
            csvwriter.writerow([fits_filename, object_id - 1, 'Debris' if finalint > 10 else 'Celestial Object'])