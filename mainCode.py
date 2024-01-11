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
output_directory = 'C:\Users\USER\Desktop\Space-Debris-Project\dataset\output_files'

# List of FITS filenames
fits_filenames = ['space5.fits','tria.fits','please4.fits','space8.fits','space6.fits','space3.fits']  # Add more filenames as needed

# Define bin_edges outside the loop
bin_edges = None

# Define arrays to store parameters for each FITS file
all_amplitudes = []
all_means = []
all_stddevs = []


for fits_filename in fits_filenames:
    # Full path to the FITS file
    full_path_fits = os.path.join(fits_directory, fits_filename)

    # Output PNG filename (assuming the same name with a different extension)
    output_image_filename = os.path.join(output_directory, os.path.splitext(fits_filename)[0] + '_preprocessed.png')

    # Convert FITS to PNG with preprocessing
    convert_fits_to_image(full_path_fits, output_image_filename)

    # Read the PNG image
    image = cv2.imread(output_image_filename)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Plot histogram for original image
    hist_original, bin_edges = np.histogram(img.flatten(), bins=256, range=[0, 256])

    # Fit a Gaussian curve to the histogram
    p0 = [1.0, np.mean(img), np.std(img)]
    params, _ = curve_fit(gaussian_curve, bin_edges[:-1], hist_original, p0=p0)

     # Append parameters to arrays
    all_amplitudes.append(params[0])
    all_means.append(params[1])
    all_stddevs.append(params[2])

    # Apply the iterative thresholding algorithm to the image
    optimal_threshold = iterative_thresholding(img)

    # Threshold the image using the optimal threshold
    thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

    # Apply Otsu's thresholding to the image
    thresholded_img_otsu, optimal_threshold_otsu = otsu_thresholding(img)

    # Print the results for the iterative method
    print(f"\nResults for Thresholding (FITS file: {fits_filename}):")
    print(f"The optimal threshold determined by the iterative algorithm: {optimal_threshold}")
    print(f"The optimal threshold determined by Otsu's method: {optimal_threshold_otsu}")

    # Display the original, thresholded (Iterative), and thresholded (Otsu) images for comparison
    cv2_imshow(np.hstack([img, thresholded_img, thresholded_img_otsu]))

     # Plot histograms
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(hist_original, color='blue')
    plt.title('Histogram for Original Image')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    hist_thresholded = cv2.calcHist([thresholded_img], [0], None, [256], [0, 256])
    plt.plot(hist_thresholded, color='black')
    plt.ylim(0,500)
    plt.title('Histogram for Thresholded Image (Iterative)')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 3)
    hist_thresholded_otsu = cv2.calcHist([thresholded_img_otsu], [0], None, [256], [0, 256])
    plt.plot(hist_thresholded_otsu, color='green')
    plt.ylim(0,500)
    plt.title('Histogram for Thresholded Image (Otsu)')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 4)
    plt.plot(bin_edges[:-1], hist_original, label='Histogram', color='blue')
    plt.plot(bin_edges[:-1], gaussian_curve(bin_edges[:-1], *params), label='Gaussian Curve', linestyle='--', color='red')
    plt.legend()
    plt.title(f'Histogram with Gaussian Curve for {fits_filename}')
    plt.ylabel('Frequency')
    plt.show()


    plt.tight_layout()
    plt.show()

    # Connected components labeling for thresholded image (Iterative method) becuase the iterative method gets better results
    num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)

    # Create a random color map for visualization
    colors_iterative = np.random.randint(0, 255, size=(num_labels_iterative, 3), dtype=np.uint8)

    # Create a colored image based on the labels
    colored_image_iterative = colors_iterative[labels_iterative]

    # Display the result
    cv2_imshow(colored_image_iterative)

    # Edge detection using the Canny edge detector
    edges = cv2.Canny(thresholded_img, 30, 100)

    # Save the processed images (Iterative method)
    cv2.imwrite(os.path.join(output_directory, f'processed_{fits_filename}_iterative.png'), colored_image_iterative)