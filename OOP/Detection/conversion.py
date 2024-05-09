import math
import os
import pandas as pd
import numpy as np
import astropy
import cv2
import matplotlib
from astropy.io import fits
import csv
import matplotlib.pyplot as plt




#convert fits to png whilst sharpening images and smoothing them:
def convert_fits_to_image(fits_directory, output_directory):
    """
    This function converts FITS files in the given directory to images and saves them in the output directory.
    """
    # List all files in the FITS directory
    fits_files = os.listdir(fits_directory)

    # Iterate over each FITS file
    for fits_filename in fits_files:
        # Skip if not a FITS file
        if not fits_filename.endswith('.fits'):
            continue

        # Construct full paths for input FITS file and output image file
        fits_file_path = os.path.join(fits_directory, fits_filename)
        output_image_filename = os.path.join(output_directory, os.path.splitext(fits_filename)[0] + '.png')

        # Open the FITS file
        with fits.open(fits_file_path) as hdul:
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

