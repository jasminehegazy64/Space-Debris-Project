import matplotlib.pyplot as plt
import cv2
from astropy.io import fits


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