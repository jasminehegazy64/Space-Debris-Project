import astropy
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
from PIL import Image
#from google.colab.patches import cv2_imshow
import os
from scipy.optimize import curve_fit
#from google.colab import drive
import Convert_debris
import threshold

# Directory containing FITS files
fits_directory = 'C:\Users\USER\Desktop\Space-Debris-Project\dataset'

# Output directory for PNG images
output_directory = '/content/drive/MyDrive/Colab-Debris/output_images/'

# List of FITS filenames
fits_filenames = ['space5.fits','tria.fits','please4.fits','space8.fits','space6.fits','space3.fits']  # Add more filenames as needed

# Define bin_edges outside the loop
bin_edges = None

# Define arrays to store parameters for each FITS file
all_amplitudes = []
all_means = []
all_stddevs = []