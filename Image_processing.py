import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits


def convert_fits_to_image(fits_filename, output_image_filename):
    with fits.open(fits_filename) as hdul:
        data = hdul[0].data
        data = cv2.GaussianBlur(data, (5, 5), 0)
        laplacian = cv2.Laplacian(data, cv2.CV_64F)
        sharpened = data - 0.8 * laplacian

        plt.imshow(sharpened, cmap='gray')
        plt.axis('off')  
        plt.savefig(output_image_filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def iterative_thresholding(image, initial_threshold=128, max_iterations=50, tolerance=1e-3):
    threshold = initial_threshold

    for iteration in range(max_iterations):
        foreground = image >= threshold
        background = image < threshold

        foreground_mean = np.mean(image[foreground])
        background_mean = np.mean(image[background])
        new_threshold = (foreground_mean + background_mean) / 2.0

        if abs(new_threshold - threshold) < tolerance:
            break

        threshold = new_threshold
    return threshold


def otsu_thresholding(image):
    image = np.uint8(image)
    _, thresholded_img = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    optimal_threshold = _
    return thresholded_img, optimal_threshold


def gaussian_curve(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

fits_directory = '/content/drive/MyDrive/2024TOTAL'
output_directory = '/content/drive/MyDrive/2024_output'

bin_edges = None

all_amplitudes = []
all_means = []
all_stddevs = []

for root, dirs, files in os.walk(fits_directory):
    for fits_filename in files:
        if fits_filename.endswith('.fits'):
            full_path_fits = os.path.join(root, fits_filename)

            fits_folder_name = os.path.splitext(fits_filename)[0]
            fits_output_directory = os.path.join(
                output_directory, fits_folder_name)
            os.makedirs(fits_output_directory, exist_ok=True)

            output_image_filename = os.path.join(
                fits_output_directory, fits_folder_name + '_preprocessed.png')

            convert_fits_to_image(full_path_fits, output_image_filename)

            image = cv2.imread(output_image_filename)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            hist_original, bin_edges = np.histogram(
                img.flatten(), bins=256, range=[0, 256])

            optimal_threshold = iterative_thresholding(img)

            thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

            thresholded_img_otsu, optimal_threshold_otsu = otsu_thresholding(
                img)

            print(f"\nResults for Thresholding (FITS file: {fits_filename}):")
            print(
                f"The optimal threshold determined by the iterative algorithm: {optimal_threshold}")
            print(
                f"The optimal threshold determined by Otsu's method: {optimal_threshold_otsu}")

            cv2.imshow(np.hstack([img, thresholded_img, thresholded_img_otsu]))

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
            plt.legend()
            plt.title(f'Histogram with Gaussian Curve for {fits_filename}')
            plt.ylabel('Frequency')
            plt.show()

            plt.tight_layout()
            plt.show()

            num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = cv2.connectedComponentsWithStats(
                thresholded_img, connectivity=8)

            colors_iterative = np.random.randint(
                0, 255, size=(num_labels_iterative, 3), dtype=np.uint8)

            colored_image_iterative = colors_iterative[labels_iterative]

            # Display the result
            cv2.imshow(colored_image_iterative)

            edges = cv2.Canny(thresholded_img, 30, 100)

            cv2.imwrite(os.path.join(
                output_directory, f'processed_{fits_filename}_iterative.png'), colored_image_iterative)

            for label in range(1, num_labels_iterative):
                area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
                component_mask = (labels_iterative == label).astype(np.uint8)

                edges_in_component = cv2.bitwise_and(
                    edges, edges, mask=component_mask)

                edge_count = np.count_nonzero(edges_in_component)

                corners = cv2.goodFeaturesToTrack(
                    thresholded_img * component_mask, maxCorners=100, qualityLevel=0.01, minDistance=0.1)
                num_corners = corners.shape[0] if corners is not None else 0

            num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = cv2.connectedComponentsWithStats(
                thresholded_img, connectivity=8)

            colors_iterative = np.random.randint(
                0, 255, size=(num_labels_iterative, 3), dtype=np.uint8)

            colored_image_iterative = colors_iterative[labels_iterative]

            # Display the result
            cv2.imshow(colored_image_iterative)

            
            for label in range(1, num_labels_iterative):
                area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
                center_x, center_y = centroids_iterative[label]
                component_mask = (labels_iterative == label).astype(np.uint8)

                edges_in_component = cv2.bitwise_and(
                    edges, edges, mask=component_mask)

                edge_count = np.count_nonzero(edges_in_component)

                corners = cv2.goodFeaturesToTrack(
                    thresholded_img * component_mask, maxCorners=100, qualityLevel=0.01, minDistance=0.1)
                num_corners = corners.shape[0] if corners is not None else 0

                print(
                    f"Component {label} (Iterative): Area = {area_iterative}, Center = ({center_x}, {center_y}), Edge count = {edge_count}, Number of Corners = {num_corners}")

            num_white_objects_iterative = num_labels_iterative - \
                1  # Subtract 1 for the background
            print(
                f'The number of white objects (Iterative) is: {num_white_objects_iterative}')

            num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = cv2.connectedComponentsWithStats(
                thresholded_img, connectivity=8)
            object_id = 1
            for i in range(1, num_labels_iterative):
                x, y, w, h, area = stats_iterative[i]

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                print(f"Object {i}: X={x}, Y={y}, Width={w}, Height={h}")

                cv2.putText(image, str(object_id), (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                object_id += 1

            cv2.imshow(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
