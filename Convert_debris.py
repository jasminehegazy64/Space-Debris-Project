import matplotlib.pyplot as plt
import cv2
from astropy.io import fits


def convert_fits_to_image(fits_filename, output_image_filename):
    # Open the FITS file
    with fits.open(fits_filename) as hdul:
        data = hdul[0].data
        data = cv2.GaussianBlur(data, (5, 5), 0)
        laplacian = cv2.Laplacian(data, cv2.CV_64F)
        sharpened = data - 0.8 * laplacian
        plt.imshow(sharpened, cmap="gray")
        plt.axis("off")
        plt.savefig(output_image_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
