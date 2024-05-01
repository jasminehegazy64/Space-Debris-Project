import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
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


def otsu_thresholding(image):
    # Ensure the image is of type np.uint8
    image = np.uint8(image)

    # Apply Otsu's thresholding
    # minimize intraclass variance
    _, thresholded_img = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Get the optimal threshold value determined by Otsu's method
    optimal_threshold = _

    return thresholded_img, optimal_threshold


def gaussian_curve(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))


# Directory containing FITS files
fits_directory = '/content/drive/MyDrive/2024TOTAL'

# Output directory for PNG images
output_directory = '/content/drive/MyDrive/2024_output'

# List of FITS filenames
# fits_filenames = ['space5.fits','tria.fits','please4.fits','space8.fits','space6.fits','space3.fits']  # Add more filenames as needed

# Define bin_edges outside the loop
bin_edges = None

# Define arrays to store parameters for each FITS file
all_amplitudes = []
all_means = []
all_stddevs = []

# Iterate through each subdirectory in fits_directory
for root, dirs, files in os.walk(fits_directory):
    for fits_filename in files:
        if fits_filename.endswith('.fits'):
            # Full path to the FITS file
            full_path_fits = os.path.join(root, fits_filename)

            # Create separate output directories for each FITS file
            fits_folder_name = os.path.splitext(fits_filename)[0]
            fits_output_directory = os.path.join(
                output_directory, fits_folder_name)
            os.makedirs(fits_output_directory, exist_ok=True)

            # Output PNG filename
            output_image_filename = os.path.join(
                fits_output_directory, fits_folder_name + '_preprocessed.png')

            # Convert FITS to PNG with preprocessing
            convert_fits_to_image(full_path_fits, output_image_filename)

            # Read the PNG image
            image = cv2.imread(output_image_filename)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Plot histogram for original image
            hist_original, bin_edges = np.histogram(
                img.flatten(), bins=256, range=[0, 256])

            # Apply the iterative thresholding algorithm to the image
            optimal_threshold = iterative_thresholding(img)

            # Threshold the image using the optimal threshold
            thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

            # Apply Otsu's thresholding to the image
            thresholded_img_otsu, optimal_threshold_otsu = otsu_thresholding(
                img)

            # Print the results for the iterative method
            print(f"\nResults for Thresholding (FITS file: {fits_filename}):")
            print(
                f"The optimal threshold determined by the iterative algorithm: {optimal_threshold}")
            print(
                f"The optimal threshold determined by Otsu's method: {optimal_threshold_otsu}")

            # Display the original, thresholded (Iterative), and thresholded (Otsu) images for comparison
            cv2_imshow(np.hstack([img, thresholded_img, thresholded_img_otsu]))

            # Plot histograms
            plt.figure(figsize=(12, 6))

            plt.subplot(2, 2, 1)
            plt.plot(hist_original, color='blue')
            plt.title('Histogram for Original Image')
            plt.ylabel('Frequency')

            plt.subplot(2, 2, 2)
            hist_thresholded = cv2.calcHist(
                [thresholded_img], [0], None, [256], [0, 256])
            plt.plot(hist_thresholded, color='black')
            plt.ylim(0, 500)
            plt.title('Histogram for Thresholded Image (Iterative)')
            plt.ylabel('Frequency')

            plt.subplot(2, 2, 3)
            hist_thresholded_otsu = cv2.calcHist(
                [thresholded_img_otsu], [0], None, [256], [0, 256])
            plt.plot(hist_thresholded_otsu, color='green')
            plt.ylim(0, 500)
            plt.title('Histogram for Thresholded Image (Otsu)')
            plt.ylabel('Frequency')

            plt.subplot(2, 2, 4)
            plt.plot(bin_edges[:-1], hist_original,
                     label='Histogram', color='blue')
            # plt.plot(bin_edges[:-1], gaussian_curve(bin_edges[:-1], *params), label='Gaussian Curve', linestyle='--', color='red')
            plt.legend()
            plt.title(f'Histogram with Gaussian Curve for {fits_filename}')
            plt.ylabel('Frequency')
            plt.show()

            plt.tight_layout()
            plt.show()

            # Connected components labeling for thresholded image (Iterative method) becuase the iterative method gets better results
            num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = cv2.connectedComponentsWithStats(
                thresholded_img, connectivity=8)

            # Create a random color map for visualization
            colors_iterative = np.random.randint(
                0, 255, size=(num_labels_iterative, 3), dtype=np.uint8)

            # Create a colored image based on the labels
            colored_image_iterative = colors_iterative[labels_iterative]

            # Display the result
            cv2_imshow(colored_image_iterative)

            # Edge detection using the Canny edge detector
            edges = cv2.Canny(thresholded_img, 30, 100)

            # Save the processed images (Iterative method)
            cv2.imwrite(os.path.join(
                output_directory, f'processed_{fits_filename}_iterative.png'), colored_image_iterative)

            # Print the area of each component (Iterative method)
            # Skip label 0 as it corresponds to the background
            for label in range(1, num_labels_iterative):
                area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
                component_mask = (labels_iterative == label).astype(np.uint8)

                # Multiply the component mask with the edges to get edges within the component
                edges_in_component = cv2.bitwise_and(
                    edges, edges, mask=component_mask)

                # Count the number of edges in the component
                edge_count = np.count_nonzero(edges_in_component)

                # Apply Shi-Tomasi corner detection to the current component ROI
                corners = cv2.goodFeaturesToTrack(
                    thresholded_img * component_mask, maxCorners=100, qualityLevel=0.01, minDistance=0.1)
                num_corners = corners.shape[0] if corners is not None else 0

                # Connected components labeling for thresholded image (Iterative method) because the iterative method gets better results
            num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = cv2.connectedComponentsWithStats(
                thresholded_img, connectivity=8)

            # Create a random color map for visualization
            colors_iterative = np.random.randint(
                0, 255, size=(num_labels_iterative, 3), dtype=np.uint8)

            # Create a colored image based on the labels
            colored_image_iterative = colors_iterative[labels_iterative]

            # Display the result
            cv2_imshow(colored_image_iterative)

            # Print the centers of each component (Iterative method)
            # Skip label 0 as it corresponds to the background
            for label in range(1, num_labels_iterative):
                area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
                center_x, center_y = centroids_iterative[label]
                component_mask = (labels_iterative == label).astype(np.uint8)

                # Multiply the component mask with the edges to get edges within the component
                edges_in_component = cv2.bitwise_and(
                    edges, edges, mask=component_mask)

                # Count the number of edges in the component
                edge_count = np.count_nonzero(edges_in_component)

                # Apply Shi-Tomasi corner detection to the current component ROI
                corners = cv2.goodFeaturesToTrack(
                    thresholded_img * component_mask, maxCorners=100, qualityLevel=0.01, minDistance=0.1)
                num_corners = corners.shape[0] if corners is not None else 0

                print(
                    f"Component {label} (Iterative): Area = {area_iterative}, Center = ({center_x}, {center_y}), Edge count = {edge_count}, Number of Corners = {num_corners}")
                # print(f"Component {label} (Iterative): Area = {area_iterative} Edge count = {edge_count} Number of Corners = {num_corners}")

            # Print the number of white objects (excluding the background) for the iterative method
            num_white_objects_iterative = num_labels_iterative - \
                1  # Subtract 1 for the background
            print(
                f'The number of white objects (Iterative) is: {num_white_objects_iterative}')

            # COORDINATES OF THE COMPONENTS
            num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = cv2.connectedComponentsWithStats(
                thresholded_img, connectivity=8)
            object_id = 1
            # Iterate through each detected object
            for i in range(1, num_labels_iterative):
                # Get the coordinates of the bounding box for the current object
                x, y, w, h, area = stats_iterative[i]

            # Draw the bounding box on the original image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Print the coordinates of the bounding box
                print(f"Object {i}: X={x}, Y={y}, Width={w}, Height={h}")

                cv2.putText(image, str(object_id), (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Increment Id
                object_id += 1

            # Display the result
            cv2_imshow(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
