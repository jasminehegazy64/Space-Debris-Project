from google.colab import drive
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
import mahotas.features.texture as texture
from skimage.feature import local_binary_pattern
import pandas as pd
from skimage import feature
import csv
from scipy.optimize import curve_fit
import os
from google.colab.patches import cv2_imshow
from PIL import Image
import cv2
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import astropy


drive.mount("/content/drive", force_remount=True)


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
        plt.imshow(sharpened, cmap="gray")
        plt.axis("off")  # Turn off the axis (including grid)

        # Save the preprocessed image as a PNG file
        plt.savefig(output_image_filename, bbox_inches="tight", pad_inches=0)
        plt.close()


def iterative_thresholding(
    image, initial_threshold=128, max_iterations=50, tolerance=1e-3
):
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
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Get the optimal threshold value determined by Otsu's method
    optimal_threshold = _

    return thresholded_img, optimal_threshold


def gaussian_curve(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev**2))


# Directory containing FITS files
fits_directory = "/content/drive/MyDrive/2024TOTAL"

# Output directory for PNG images
output_directory = "/content/drive/MyDrive/2024_output"

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
        if fits_filename.endswith(".fits"):
            # Full path to the FITS file
            full_path_fits = os.path.join(root, fits_filename)

            # Create separate output directories for each FITS file
            fits_folder_name = os.path.splitext(fits_filename)[0]
            fits_output_directory = os.path.join(output_directory, fits_folder_name)
            os.makedirs(fits_output_directory, exist_ok=True)

            # Output PNG filename
            output_image_filename = os.path.join(
                fits_output_directory, fits_folder_name + "_preprocessed.png"
            )

            # Convert FITS to PNG with preprocessing
            convert_fits_to_image(full_path_fits, output_image_filename)

            # Read the PNG image
            image = cv2.imread(output_image_filename)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Plot histogram for original image
            hist_original, bin_edges = np.histogram(
                img.flatten(), bins=256, range=[0, 256]
            )

            # Apply the iterative thresholding algorithm to the image
            optimal_threshold = iterative_thresholding(img)

            # Threshold the image using the optimal threshold
            thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

            # Apply Otsu's thresholding to the image
            thresholded_img_otsu, optimal_threshold_otsu = otsu_thresholding(img)

            # Print the results for the iterative method
            print(f"\nResults for Thresholding (FITS file: {fits_filename}):")
            print(
                f"The optimal threshold determined by the iterative algorithm: {optimal_threshold}"
            )
            print(
                f"The optimal threshold determined by Otsu's method: {optimal_threshold_otsu}"
            )

            # Display the original, thresholded (Iterative), and thresholded (Otsu) images for comparison
            cv2.imshow(np.hstack([img, thresholded_img, thresholded_img_otsu]))

            # Plot histograms
            plt.figure(figsize=(12, 6))

            plt.subplot(2, 2, 1)
            plt.plot(hist_original, color="blue")
            plt.title("Histogram for Original Image")
            plt.ylabel("Frequency")

            plt.subplot(2, 2, 2)
            hist_thresholded = cv2.calcHist(
                [thresholded_img], [0], None, [256], [0, 256]
            )
            plt.plot(hist_thresholded, color="black")
            plt.ylim(0, 500)
            plt.title("Histogram for Thresholded Image (Iterative)")
            plt.ylabel("Frequency")

            plt.subplot(2, 2, 3)
            hist_thresholded_otsu = cv2.calcHist(
                [thresholded_img_otsu], [0], None, [256], [0, 256]
            )
            plt.plot(hist_thresholded_otsu, color="green")
            plt.ylim(0, 500)
            plt.title("Histogram for Thresholded Image (Otsu)")
            plt.ylabel("Frequency")

            plt.subplot(2, 2, 4)
            plt.plot(bin_edges[:-1], hist_original, label="Histogram", color="blue")
            # plt.plot(bin_edges[:-1], gaussian_curve(bin_edges[:-1], *params), label='Gaussian Curve', linestyle='--', color='red')
            plt.legend()
            plt.title(f"Histogram with Gaussian Curve for {fits_filename}")
            plt.ylabel("Frequency")
            plt.show()

            plt.tight_layout()
            plt.show()

            # Connected components labeling for thresholded image (Iterative method) becuase the iterative method gets better results
            (
                num_labels_iterative,
                labels_iterative,
                stats_iterative,
                centroids_iterative,
            ) = cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)

            # Create a random color map for visualization
            colors_iterative = np.random.randint(
                0, 255, size=(num_labels_iterative, 3), dtype=np.uint8
            )

            # Create a colored image based on the labels
            colored_image_iterative = colors_iterative[labels_iterative]

            # Display the result
            cv2.imshow(colored_image_iterative)

            # Edge detection using the Canny edge detector
            edges = cv2.Canny(thresholded_img, 30, 100)

            # Save the processed images (Iterative method)
            cv2.imwrite(
                os.path.join(
                    output_directory, f"processed_{fits_filename}_iterative.png"
                ),
                colored_image_iterative,
            )

            # Print the area of each component (Iterative method)
            # Skip label 0 as it corresponds to the background
            for label in range(1, num_labels_iterative):
                area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
                component_mask = (labels_iterative == label).astype(np.uint8)

                # Multiply the component mask with the edges to get edges within the component
                edges_in_component = cv2.bitwise_and(edges, edges, mask=component_mask)

                # Count the number of edges in the component
                edge_count = np.count_nonzero(edges_in_component)

                # Apply Shi-Tomasi corner detection to the current component ROI
                corners = cv2.goodFeaturesToTrack(
                    thresholded_img * component_mask,
                    maxCorners=100,
                    qualityLevel=0.01,
                    minDistance=0.1,
                )
                num_corners = corners.shape[0] if corners is not None else 0

                # Connected components labeling for thresholded image (Iterative method) because the iterative method gets better results
            (
                num_labels_iterative,
                labels_iterative,
                stats_iterative,
                centroids_iterative,
            ) = cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)

            # Create a random color map for visualization
            colors_iterative = np.random.randint(
                0, 255, size=(num_labels_iterative, 3), dtype=np.uint8
            )

            # Create a colored image based on the labels
            colored_image_iterative = colors_iterative[labels_iterative]

            # Display the result
            cv2.imshow(colored_image_iterative)

            # Print the centers of each component (Iterative method)
            # Skip label 0 as it corresponds to the background
            for label in range(1, num_labels_iterative):
                area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
                center_x, center_y = centroids_iterative[label]
                component_mask = (labels_iterative == label).astype(np.uint8)

                # Multiply the component mask with the edges to get edges within the component
                edges_in_component = cv2.bitwise_and(edges, edges, mask=component_mask)

                # Count the number of edges in the component
                edge_count = np.count_nonzero(edges_in_component)

                # Apply Shi-Tomasi corner detection to the current component ROI
                corners = cv2.goodFeaturesToTrack(
                    thresholded_img * component_mask,
                    maxCorners=100,
                    qualityLevel=0.01,
                    minDistance=0.1,
                )
                num_corners = corners.shape[0] if corners is not None else 0

                print(
                    f"Component {label} (Iterative): Area = {area_iterative}, Center = ({center_x}, {center_y}), Edge count = {edge_count}, Number of Corners = {num_corners}"
                )
                # print(f"Component {label} (Iterative): Area = {area_iterative} Edge count = {edge_count} Number of Corners = {num_corners}")

            # Print the number of white objects (excluding the background) for the iterative method
            num_white_objects_iterative = (
                num_labels_iterative - 1
            )  # Subtract 1 for the background
            print(
                f"The number of white objects (Iterative) is: {num_white_objects_iterative}"
            )

            # COORDINATES OF THE COMPONENTS
            (
                num_labels_iterative,
                labels_iterative,
                stats_iterative,
                centroids_iterative,
            ) = cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)
            object_id = 1
            # Iterate through each detected object
            for i in range(1, num_labels_iterative):
                # Get the coordinates of the bounding box for the current object
                x, y, w, h, area = stats_iterative[i]

                # Draw the bounding box on the original image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Print the coordinates of the bounding box
                print(f"Object {i}: X={x}, Y={y}, Width={w}, Height={h}")

                cv2.putText(
                    image,
                    str(object_id),
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                # Increment Id
                object_id += 1

            # Display the result
            cv2.imshow(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Define a class to represent each detected object


class DetectedObject:
    def __init__(self, id, bbox, area, edges, corners):
        self.id = id  # Object ID
        self.bbox = bbox  # Bounding box (x, y, w, h)
        self.area = area  # Area of the object
        self.edges = edges  # Number of edges within the object
        self.corners = corners  # Number of corners within the object


# Function to compute distance between two objects based on their attributes


def object_distance(prev_obj, curr_obj):
    pos_dist = np.linalg.norm(np.array(prev_obj.bbox[:2]) - np.array(curr_obj.bbox[:2]))
    area_dist = abs(prev_obj.area - curr_obj.area)
    edges_dist = abs(prev_obj.edges - curr_obj.edges)
    corners_dist = abs(prev_obj.corners - curr_obj.corners)
    return pos_dist + area_dist + edges_dist + corners_dist


# Function to match objects between frames based on their positions and attributes


def match_objects(prev_objects, curr_objects):
    matched_pairs = []  # List to store matched object pairs
    for prev_obj in prev_objects:
        min_dist = float("inf")
        matched_obj = None
        for curr_obj in curr_objects:
            dist = object_distance(prev_obj, curr_obj)
            if dist < min_dist:
                min_dist = dist
                matched_obj = curr_obj
        if matched_obj:
            matched_pairs.append((prev_obj.id, matched_obj.id))
    return matched_pairs


# Function to draw bounding boxes and IDs on the image


def draw_boxes(image, objects):
    for obj in objects:
        x, y, w, h = obj.bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            str(obj.id),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


# Initialize variables for object tracking
prev_objects = []  # List to store detected objects in the previous frame
first_frame = True  # Flag to indicate the first frame
id_mapping = {}  # Dictionary to map IDs from previous frame to current frame

# Iterate over FITS files and convert each to PNG
for fits_filename in os.listdir(fits_directory):
    if fits_filename.endswith(".fits"):
        # Full path to the FITS file
        full_path_fits = os.path.join(fits_directory, fits_filename)

        # Output PNG filename (assuming the same name with a different extension)
        output_image_filename = os.path.join(
            output_directory, os.path.splitext(fits_filename)[0] + "_preprocessed.png"
        )

        # Convert FITS to PNG with preprocessing
        convert_fits_to_image(full_path_fits, output_image_filename)

        # Read the PNG image
        image = cv2.imread(output_image_filename)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply the iterative thresholding algorithm to the image
        optimal_threshold = iterative_thresholding(img)

        # Threshold the image using the optimal threshold
        thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

        # Connected components labeling for thresholded image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresholded_img, connectivity=8
        )

        # Create DetectedObject instances for each detected object
        objects = []
        for i in range(1, num_labels):
            x, y, w, h = stats[i][:4]  # Bounding box coordinates
            area = stats[i][cv2.CC_STAT_AREA]  # Area of the object
            component_mask = (labels == i).astype(
                np.uint8
            )  # Mask for the current object
            _, edges = cv2.threshold(
                cv2.Canny(component_mask, 30, 100), 0, 255, cv2.THRESH_BINARY
            )  # Edges within the object
            corners = cv2.goodFeaturesToTrack(
                component_mask, maxCorners=100, qualityLevel=0.01, minDistance=0.1
            )  # Corners within the object
            num_corners = corners.shape[0] if corners is not None else 0

            # Check if there is a mapping for the object ID from the previous frame
            if first_frame or prev_objects == []:
                obj_id = i
            else:
                prev_obj_id = id_mapping.get(i)
                if prev_obj_id is not None:
                    obj_id = prev_obj_id
                else:
                    obj_id = max(id_mapping.values()) + 1

            objects.append(
                DetectedObject(
                    obj_id, (x, y, w, h), area, np.count_nonzero(edges), num_corners
                )
            )
            id_mapping[i] = obj_id

        # For the first frame, initialize object IDs
        if first_frame:
            first_frame = False
        else:
            # For subsequent frames, match objects between frames and update their IDs
            matched_pairs = match_objects(prev_objects, objects)
            for prev_id, curr_id in matched_pairs:
                id_mapping[curr_id] = prev_id

        # Make a copy of the thresholded image for visualization
        thresholded_img_vis = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2BGR)

        # Draw bounding boxes and IDs on the thresholded image copy
        draw_boxes(thresholded_img_vis, objects)

        # Display the result
        cv2.imshow(thresholded_img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Update previous objects for the next iteration
        prev_objects = objects[:]


# Define a class to represent each detected object


class DetectedObject:
    def __init__(self, id, bbox):
        self.id = id  # Object ID
        self.bbox = bbox  # Bounding box (x, y, w, h)


# Function to match objects between frames based on their positions


def match_objects(prev_objects, curr_objects):
    matched_pairs = []  # List to store matched object pairs
    for prev_obj in prev_objects:
        min_dist = float("inf")
        matched_obj = None
        for curr_obj in curr_objects:
            dist = np.linalg.norm(
                np.array(prev_obj.bbox[:2]) - np.array(curr_obj.bbox[:2])
            )
            if dist < min_dist:
                min_dist = dist
                matched_obj = curr_obj
        if matched_obj:
            matched_pairs.append((prev_obj.id, matched_obj.id))
    return matched_pairs


# Function to draw bounding boxes and IDs on the image


def draw_boxes(image, objects):
    for obj in objects:
        x, y, w, h = obj.bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            str(obj.id),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


# Initialize variables for object tracking
prev_objects = []  # List to store detected objects in the previous frame
first_frame = True  # Flag to indicate the first frame
obj_counter = 1  # Counter for assigning IDs to objects

# Iterate over FITS files and convert each to PNG
for fits_filename in os.listdir(fits_directory):
    if fits_filename.endswith(".fits"):
        # Full path to the FITS file
        full_path_fits = os.path.join(fits_directory, fits_filename)

        # Output PNG filename (assuming the same name with a different extension)
        output_image_filename = os.path.join(
            output_directory, os.path.splitext(fits_filename)[0] + "_preprocessed.png"
        )

        # Convert FITS to PNG with preprocessing
        convert_fits_to_image(full_path_fits, output_image_filename)

        # Read the PNG image
        image = cv2.imread(output_image_filename)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply the iterative thresholding algorithm to the image
        optimal_threshold = iterative_thresholding(img)

        # Threshold the image using the optimal threshold
        thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

        # Connected components labeling for thresholded image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresholded_img, connectivity=8
        )

        # Create DetectedObject instances for each detected object
        objects = []
        for i in range(1, num_labels):
            x, y, w, h = stats[i][:4]  # Bounding box coordinates
            objects.append(DetectedObject(i, (x, y, w, h)))

        # For the first frame, initialize object IDs
        if first_frame:
            for obj in objects:
                obj.id = obj_counter
                obj_counter += 1
            first_frame = False
        else:
            # For subsequent frames, match objects between frames and update their IDs
            matched_pairs = match_objects(prev_objects, objects)
            for prev_id, curr_id in matched_pairs:
                for obj in objects:
                    if obj.id == curr_id:
                        obj.id = prev_id

        # Make a copy of the thresholded image for visualization
        thresholded_img_vis = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2BGR)

        # Draw bounding boxes and IDs on the thresholded image copy
        draw_boxes(thresholded_img_vis, objects)

        # Display the result
        cv2.imshow(thresholded_img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Update previous objects for the next iteration
        prev_objects = objects[:]

# Create a list to store DataFrames for each FITS file
dfs = []

# Iterate over FITS files and extract header information
for fits_filename in os.listdir(fits_directory):
    if fits_filename.endswith(".fits"):
        # Full path to the FITS file
        full_path_fits = os.path.join(fits_directory, fits_filename)

        # Open the FITS file
        with fits.open(full_path_fits) as hdul:
            # Access the header of the primary HDU
            header = hdul[0].header

            # Extract relevant information
            try:
                # Extracting Date of observation from header
                dateobs = header["DATE-OBS"]
            except KeyError:
                dateobs = "Not available"

            try:
                # Extracting Number of axis from the header
                naxis = header["NAXIS"]
            except KeyError:
                naxis = "Not available"

            try:
                # Extracting Wavelength from the header
                wavelen = header["WAVELENG"]
            except KeyError:
                wavelen = "Not available"

            try:
                # Extracting Pixel Scale from the header
                pixel_scale = header["BITPIX"]
            except KeyError:
                pixel_scale = "Not available"

            try:
                # Extracting Filter from the header
                filter_used = header["FILTER"]
            except KeyError:
                filter_used = "Not available"

            try:
                # Extracting Exposure Time from the header
                exposure_time = header["EXPOSURE"]
            except KeyError:
                exposure_time = "Not available"

            try:
                # Extracting Pixels Zero Value from the header
                bzero = header["BZERO"]
            except KeyError:
                bzero = "Not available"

            # Extract the image data (pixel values)
            image_data = hdul[0].data

            # Calculate the intensity
            intensity = (image_data - bzero) * exposure_time

            # Create a DataFrame for the current FITS file
            df = pd.DataFrame(
                {
                    "FITS File": [fits_filename],
                    "Number of Axis": [naxis],
                    "Wave Length": [wavelen],
                    "Date of Observation": [dateobs],
                    "Pixel Scale": [pixel_scale],
                    "Exposure Time": [exposure_time],
                    "Physical Zero Value": [bzero],
                    "Filter Used": [filter_used],
                    "Intensity": [intensity.sum()],
                }
            )

            # Append the DataFrame to the list
            dfs.append(df)

# Concatenate the list of DataFrames into a single DataFrame
df_one = pd.concat(dfs, ignore_index=True)

# Display the resulting DataFrame
print(df_one)

# Extract Texture:
# Haralick Texture Features and Local Binary Pattern
# Haralick texture -> derived from the co-occurrence matrix[representation of the spatial relationships of pixel intensities in an image]
# LBP -> texture description that encodes the local spatial patterns of pixel intensities.

# Create lists to store extracted features
fits_files_list = []
haralick_mean_list = []
haralick_std_list = []
lbp_hist_list = []

# Iterate over FITS files and extract header information
for fits_filename in os.listdir(fits_directory):
    if fits_filename.endswith(".fits"):
        # Full path to the FITS file
        full_path_fits = os.path.join(fits_directory, fits_filename)

        # Open the FITS file
        with fits.open(full_path_fits) as hdul:
            # Access the data from the primary HDU
            data = hdul[0].data

        # Convert FITS data to a grayscale image (adjust normalization if necessary)
        image = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Calculate Haralick texture features
        haralick_features = texture.haralick(image)

        # Calculate Local Binary Patterns (LBP)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method="uniform")

        # Extract LBP histogram as features
        lbp_hist, _ = np.histogram(
            lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
        )

        # Append the features to the lists
        fits_files_list.append(fits_filename)
        haralick_mean_list.append(np.mean(haralick_features, axis=0))
        haralick_std_list.append(np.std(haralick_features, axis=0))
        lbp_hist_list.append(lbp_hist)

        # Print the information
        print(f"FITS File: {fits_filename}")
        print("Haralick Texture Features:")
        print("Mean:", np.mean(haralick_features, axis=0))
        print("Standard Deviation:", np.std(haralick_features, axis=0))
        print("Local Binary Pattern Histogram:", lbp_hist)
        print("-" * 50)


# Create a DataFrame
df_second = pd.DataFrame(
    {
        "FITS File": fits_files_list,
        "Haralick Mean": haralick_mean_list,
        "Haralick Std": haralick_std_list,
        "LBP Histogram": lbp_hist_list,
    }
)

# Save the DataFrame to a CSV file
csv_filename = "/content/drive/MyDrive/2024_output/texture_features.csv"

# Concatenate df_first and df_second
# df_combined = pd.concat([df_first, df_second], ignore_index=True)
# Merge the DataFrames based on the "FITS File" column
df_combined = pd.merge(df_one, df_second, on="FITS File", how="outer")
df_combined.to_csv(csv_filename, index=False)

# Print CSV file holding the all texture features
csv_filename = "/content/drive/MyDrive/2024_output/texture_features.csv"
df = pd.read_csv(csv_filename)

# Display the DataFrame
print(df)

# Descriptive statistics for numerical columns
print("Descriptive Statistics:\n", df.describe())


def momentOfInertia(xWidth, yHeight, xCG, yCG):
    Ixx = sum((y - yCG) ** 2 for y in yHeight)
    Iyy = sum((x - xCG) ** 2 for x in xWidth)
    Ixy = sum((x - xCG) * (y - yCG) for x, y in zip(xWidth, yHeight))

    return Ixx, Iyy, Ixy


def mainInteria(Ixx, Iyy, Ixy, yHeight, xWidth):
    Imain1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy) ** 2 + 4 * (Ixy) ** 2))
    Imain2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy) ** 2 + 4 * (Ixy) ** 2))

    epsilonn = 10

    finalInteria = Imain1 / Imain2
    if finalInteria > epsilonn:
        print(f"This object  is predicted to be debris")
    else:
        print(f"This object  is predicted to be a Celestial object")

    return finalInteria


csv_file_path = "/content/drive/MyDrive/2024_output/InetriaOutPut.csv"

# Open the CSV file in write mode
with open(csv_file_path, "w", newline="") as csvfile:
    # Create a CSV writer
    csvwriter = csv.writer(csvfile)

    # Write the header row
    csvwriter.writerow(
        [
            "Image",
            "Object ID",
            "Area",
            "Edges",
            "Center_x",
            "Center_y",
            "Width",
            "Height",
            "lbp_mean",
            "lbp_std",
            "Prediction",
        ]
    )

    for fits_filename in os.listdir(fits_directory):
        if fits_filename.endswith(".fits"):
            # Full path to the FITS file
            full_path_fits = os.path.join(fits_directory, fits_filename)

            # Output PNG filename (assuming the same name with a different extension)
            output_image_filename = os.path.join(
                output_directory,
                os.path.splitext(fits_filename)[0] + "_preprocessed.png",
            )
            convert_fits_to_image(full_path_fits, output_image_filename)

            image = cv2.imread(output_image_filename)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply the iterative thresholding algorithm to the image
            optimal_threshold = iterative_thresholding(img)

            # Threshold the image using the optimal threshold
            thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

            (
                num_labels_iterative,
                labels_iterative,
                stats_iterative,
                centroids_iterative,
            ) = cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)

            # Reset object_id for each new image
            object_id = 1

            for label in range(1, num_labels_iterative):
                area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
                component_mask = (labels_iterative == label).astype(np.uint8)
                center_x, center_y = centroids_iterative[label]
                # Multiply the component mask with the edges to get edges within the component
                edges = cv2.Canny(thresholded_img, 30, 100)
                edges_in_component = cv2.bitwise_and(edges, edges, mask=component_mask)

                # Get the coordinates of the bounding box for the current object
                x, y, w, h, area = stats_iterative[label]
                # Count the number of edges in the component
                edge_count = np.count_nonzero(edges_in_component)
                # Extract the region of interest (ROI)
                roi = img[y : min(y + h, img.shape[0]), x : min(x + w, img.shape[1])]

                # Compute Local Binary Pattern (LBP) features
                lbp_features = feature.local_binary_pattern(
                    roi, P=8, R=1, method="uniform"
                )
                lbp_mean = np.mean(lbp_features)
                lbp_std = np.std(lbp_features)

                # Ensure xWidth and yHeight are iterable (lists)
                xWidth = list(range(w))
                yHeight = list(range(h))

                # Print the coordinates of the bounding box
                print(f"Object {object_id} in {fits_filename}:")

                cv2.putText(
                    image,
                    str(object_id),
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # Increment Id
                object_id += 1

                Ixx, Iyy, Ixy = momentOfInertia(xWidth, yHeight, center_x, center_y)
                finalint = mainInteria(Ixx, Iyy, Ixy, yHeight, xWidth)

                # Write the row to the CSV file
                csvwriter.writerow(
                    [
                        fits_filename,
                        object_id - 1,
                        area_iterative,
                        edge_count,
                        center_x,
                        center_y,
                        w,
                        h,
                        lbp_mean,
                        lbp_std,
                        "Debris" if finalint > 10 else "Celestial Object",
                    ]
                )

# Print CSV file holding the all texture features and Inertia
csv_filename = "/content/drive/MyDrive/2024_output/texture_features.csv"
df = pd.read_csv(csv_filename)
print(df)

# Check for missing values
missing_values = df.isnull().sum()

# Display the count of missing values for each column
print("Missing Values:\n", missing_values)

# Models
csv_file_path = "/content/drive/MyDrive/2024_output/InetriaOutPut.csv"
df = pd.read_csv(csv_file_path)
df.info()


def outlier_percent(data):
    numeric_columns = data.select_dtypes(include=[np.number])
    Q1 = numeric_columns.quantile(0.25)
    Q3 = numeric_columns.quantile(0.75)
    IQR = Q3 - Q1
    minimum = Q1 - (1.5 * IQR)
    maximum = Q3 + (1.5 * IQR)
    num_outliers = (
        ((numeric_columns < minimum) | (numeric_columns > maximum)).sum().sum()
    )
    num_total = numeric_columns.count().sum()
    return (num_outliers / num_total) * 100


outlier_percent(df)

# CHECKING the precentage of NULL VALUES after removing outliers
nulls = df.isnull().sum()
nulls

# encoding Prediction column
df["Prediction"].replace(to_replace="Celestial Object", value=0, inplace=True)
df["Prediction"].replace(to_replace="Debris", value=1, inplace=True)

scaler = StandardScaler()
df[
    ["Area", "Edges", "Center_x", "Center_y", "Width", "Height", "lbp_mean", "lbp_std"]
] = scaler.fit_transform(
    df[
        [
            "Area",
            "Edges",
            "Center_x",
            "Center_y",
            "Width",
            "Height",
            "lbp_mean",
            "lbp_std",
        ]
    ]
)

df.head()

# Calculate correlation matrix
correlation_matrix = df.drop(["Object ID", "Image"], axis=1).corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create a heatmap of the correlation matrix
plt.figure(figsize=(14, 14))
sns.heatmap(data=correlation_matrix, annot=True, cmap="coolwarm", center=0, mask=mask)
plt.title("Correlation Matrix")
plt.show()

df = df.sample(frac=1, random_state=42)

X = df.drop(["Object ID", "Image", "Prediction"], axis=1)
y = df["Prediction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = []
acc = []

# Support Vector Machine (SVM)
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)

# Print SVM accuracy
print(f"SVM Test Accuracy: {svm_accuracy}")
print("\nSVM Classification Report:\n", classification_report(y_test, svm_predictions))

model.append("SVM")
acc.append(svm_accuracy)

svm_predictions_train = svm_model.predict(X_train)
svm_accuracy_train = accuracy_score(y_train, svm_predictions_train)

# Print SVM accuracy
print(f"SVM Train Accuracy: {svm_accuracy_train}")
print(
    "\nSVM Train Classification Report:\n",
    classification_report(y_train, svm_predictions_train),
)

# k-Nearest Neighbors (k-NN)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

# Print k-NN accuracy
print(f"\nk-NN Test Accuracy: {knn_accuracy}")
print("\nk-NN Classification Report:\n", classification_report(y_test, knn_predictions))

model.append("KNN")
acc.append(knn_accuracy)

knn_predictions_train = knn_model.predict(X_train)
knn_accuracy_train = accuracy_score(y_train, knn_predictions_train)

# Print k-NN accuracy
print(f"\nk-NN Accuracy train: {knn_accuracy_train}")
print(
    "\nk-NN Classification Report:\n",
    classification_report(y_train, knn_predictions_train),
)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)

# Print Naive Bayes accuracy
print(f"\nNaive Bayes Test Accuracy: {nb_accuracy}")
print(
    "\nNaive Bayes Classification Report:\n",
    classification_report(y_test, nb_predictions),
)

model.append("NB")
acc.append(nb_accuracy)

nb_predictions_train = nb_model.predict(X_train)
nb_accuracy_train = accuracy_score(y_train, nb_predictions_train)

# Print Naive Bayes accuracy
print(f"\nNaive Bayes Train Accuracy: {nb_accuracy_train}")
print(
    "\nNaive Bayes Train Classification Report:\n",
    classification_report(y_train, nb_predictions_train),
)

# Logistic Regression
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
logreg_predictions = logreg_model.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)

# Print Logistic Regression accuracy
print(f"\nLogistic Regression Test Accuracy: {logreg_accuracy}")
print(
    "\nLogistic Regression Classification Report:\n",
    classification_report(y_test, logreg_predictions),
)

model.append("LG")
acc.append(logreg_accuracy)

logreg_predictions_train = logreg_model.predict(X_train)
logreg_accuracy_train = accuracy_score(y_train, logreg_predictions_train)

# Print Logistic Regression accuracy
print(f"\nLogistic Regression Train Accuracy: {logreg_accuracy_train}")
print(
    "\nLogistic Regression Train Classification Report:\n",
    classification_report(y_train, logreg_predictions_train),
)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Print Random Forest accuracy
print(f"Random Forest Test Accuracy: {rf_accuracy}")
print(
    "\nRandom Forest Classification Report:\n",
    classification_report(y_test, rf_predictions),
)

model.append("RF")
acc.append(rf_accuracy)

rf_predictions_train = rf_model.predict(X_train)
rf_accuracy_train = accuracy_score(y_train, rf_predictions_train)

# Print Random Forest accuracy
print(f"Random Forest Train Accuracy: {rf_accuracy_train}")
print(
    "\nRandom Forest Classification Report:\n",
    classification_report(y_train, rf_predictions_train),
)

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

# Print Decision Tree accuracy
print(f"Decision Tree Test Accuracy: {dt_accuracy}")
print(
    "\nDecision Tree Classification Report:\n",
    classification_report(y_test, dt_predictions),
)

model.append("DT")
acc.append(dt_accuracy)

dt_predictions_train = dt_model.predict(X_train)
dt_accuracy_train = accuracy_score(y_train, dt_predictions_train)

# Print Decision Tree accuracy
print(f"Decision Tree Train Accuracy: {dt_accuracy_train}")
print(
    "\nDecision Tree Classification Report:\n",
    classification_report(y_train, dt_predictions_train),
)

# Create XGBoost classifier
xgb_model = xgb.XGBClassifier(objective="binary:logitraw", random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Test Accuracy: {accuracy}")

# Print classification report
print("\nXGBoost Classification Report:\n", classification_report(y_test, y_pred))

model.append("xgb")
acc.append(accuracy)

# Make predictions
y_pred_train = xgb_model.predict(X_train)

# Calculate accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"XGBoost Train Accuracy: {accuracy_train}")

# Print classification report
print(
    "\nXGBoost Train Classification Report:\n",
    classification_report(y_train, y_pred_train),
)

# Create LightGBM classifier
lgb_model = lgb.LGBMClassifier(objective="binary", random_state=42)
lgb_model.fit(X_train, y_train)

# Make predictions
y_pred = lgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"LightGBM Test Accuracy: {accuracy}")

# Print classification report
print("\nLightGBM Classification Report:\n", classification_report(y_test, y_pred))

model.append("LGBM")
acc.append(accuracy)

# Make predictions
y_pred_train = lgb_model.predict(X_train)

# Calculate accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"LightGBM Train Accuracy: {accuracy_train}")

# Print classification report
print(
    "\nLightGBM Train Classification Report:\n",
    classification_report(y_train, y_pred_train),
)

plt.figure(figsize=(10, 8))
plt.bar(model, acc)
plt.title("conclusion")
plt.xlabel("model")
plt.ylabel("accuracy")

plt.figure(figsize=(10, 10))
plt.plot(model, acc, "r*-")  # 'r' is the color red
plt.xlabel("model")
plt.ylabel("accuracy")
plt.title("Conclusion")

# Assuming 'best_model' is your trained model
joblib.dump(logreg_model, "best_model.pkl")

loaded_model = joblib.load("best_model.pkl")
# do preprocessing needed

# Assuming 'new_data' is a DataFrame with the same features as the training data
predictions = loaded_model.predict(
    df.drop(["Object ID", "Image", "Prediction"], axis=1)
)
predictions


def block_matching_optical_flow(
    prev_frame, next_frame, bounding_boxes, block_size, search_area
):
    flow_vectors = []
    for bbox in bounding_boxes:
        x, y, w, h = bbox

        # Define the search region
        x_min = max(0, x - search_area)
        y_min = max(0, y - search_area)
        x_max = min(prev_frame.shape[1] - 1, x + w + search_area)
        y_max = min(prev_frame.shape[0] - 1, y + h + search_area)
        search_region = next_frame[y_min:y_max, x_min:x_max]

        # Define the block in the previous frame
        block = prev_frame[y : y + h, x : x + w]

        # Calculate the block matching optical flow using template matching
        match_template_result = cv2.matchTemplate(
            search_region, block, cv2.TM_CCORR_NORMED
        )
        _, _, _, max_loc = cv2.minMaxLoc(match_template_result)

        # Calculate the displacement vector
        dx = max_loc[0] - (x_min + x)
        dy = max_loc[1] - (y_min + y)
        flow_vectors.append((dx, dy))

    return flow_vectors


def draw_flow(img, flow, bounding_boxes, step=16):
    vis = img.copy()
    for i, bbox in enumerate(bounding_boxes):
        x, y, w, h = bbox
        dx, dy = flow[i]
        x_mid = x + w // 2
        y_mid = y + h // 2
        cv2.arrowedLine(vis, (x_mid, y_mid), (x_mid + dx, y_mid + dy), (0, 255, 0), 1)
    return vis


# Path to the folder containing PNG images
folder_path = "/content/drive/MyDrive/2024_output"

# Define the block size and search area
block_size = 16
search_area = 16

# Bounding boxes extracted from the images earlier
bounding_boxes = [(211, 280, 2, 3), (211, 280, 2, 3)]

# Get a list of PNG file paths in the folder
png_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

# Sort the files to ensure proper order
png_files.sort()

# Read the first frame
prev_frame = cv2.imread(os.path.join(folder_path, png_files[0]))

for i in range(1, len(png_files)):
    # Read the next frame
    next_frame = cv2.imread(os.path.join(folder_path, png_files[i]))

    # Calculate optical flow using block matching algorithm with custom search regions
    flow_vectors = block_matching_optical_flow(
        prev_frame, next_frame, bounding_boxes, block_size, search_area
    )

    # Draw flow vectors for each bounding box
    output_image = draw_flow(next_frame, flow_vectors, bounding_boxes)

    # Display the result
    cv2.imshow(output_image)

    # Update the previous frame
    prev_frame = next_frame

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Release the window
cv2.destroyAllWindows()


class DetectedObject:
    def __init__(self, id, bbox, area, edges, corners, center, width, height):
        self.id = id  # Object ID
        self.bbox = bbox  # Bounding box (x, y, w, h)
        self.area = area  # Area of the object
        self.edges = edges  # Number of edges detected within the object
        self.corners = corners  # Number of corners detected within the object
        self.center = center  # Center coordinates of the object
        self.width = width  # Width of the object bounding box
        self.height = height  # Height of the object bounding box


# Initialize variables for object tracking
prev_objects = []  # List to store detected objects in the previous frame
first_frame = True  # Flag to indicate the first frame
obj_counter = 1  # Counter for assigning IDs to objects

# Iterate over FITS files and convert each to PNG
for fits_filename in os.listdir(fits_directory):
    if fits_filename.endswith(".fits"):
        # Full path to the FITS file
        full_path_fits = os.path.join(fits_directory, fits_filename)

        # Output PNG filename (assuming the same name with a different extension)
        output_image_filename = os.path.join(
            output_directory, os.path.splitext(fits_filename)[0] + "_preprocessed.png"
        )

        # Convert FITS to PNG with preprocessing
        convert_fits_to_image(full_path_fits, output_image_filename)

        # Read the PNG image
        image = cv2.imread(output_image_filename)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply the iterative thresholding algorithm to the image
        optimal_threshold = iterative_thresholding(img)

        # Threshold the image using the optimal threshold
        thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

        # Connected components labeling for thresholded image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresholded_img, connectivity=8
        )

        # Create DetectedObject instances for each detected object
        objects = []
        for i in range(1, num_labels):
            x, y, w, h = stats[i][:4]  # Bounding box coordinates
            area = stats[i][cv2.CC_STAT_AREA]  # Area
            component_mask = (labels == i).astype(np.uint8)
            edges = cv2.Canny(component_mask, 30, 100).sum() / 255  # Edge count
            corners = cv2.goodFeaturesToTrack(
                component_mask, maxCorners=100, qualityLevel=0.01, minDistance=0.1
            )
            # Number of corners
            num_corners = corners.shape[0] if corners is not None else 0
            center = (x + w // 2, y + h // 2)  # Center coordinates
            width, height = w, h  # Width and height
            objects.append(
                DetectedObject(
                    obj_counter,
                    (x, y, w, h),
                    area,
                    edges,
                    num_corners,
                    center,
                    width,
                    height,
                )
            )
            obj_counter += 1

        # Update previous objects for the next iteration
        prev_objects = objects[:]
