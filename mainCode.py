import astropy
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
from PIL import Image
# from google.colab.patches import cv2_imshow
import os
import numpy as np
from scipy.optimize import curve_fit
# from google.colab import drive
from Convert_debris import convert_fits_to_image
from threshold import iterative_thresholding, otsu_thresholding
from gaussian_curve import gaussian_curve
import pandas as pd


# Directory containing FITS files
fits_directory = 'C:\\Users\\ASUS\Desktop\\Space-Debris-Project\\dataset'

# Output directory for PNG images
output_directory = 'C:\\Users\\ASUS\\Desktop\\Space-Debris-Project\\dataset\\output_files'

# List of FITS filenames
fits_filenames = ['space5.fits', 'tria.fits', 'please4.fits', 'space8.fits',
                  'space6.fits', 'space3.fits']  # Add more filenames as needed

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
    output_image_filename = os.path.join(
        output_directory, os.path.splitext(fits_filename)[0] + '_preprocessed.png')

    # Convert FITS to PNG with preprocessing
    convert_fits_to_image(full_path_fits, output_image_filename)

    # Read the PNG image
    image = cv2.imread(output_image_filename)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Plot histogram for original image
    hist_original, bin_edges = np.histogram(
        img.flatten(), bins=256, range=[0, 256])

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
    print(
        f"The optimal threshold determined by the iterative algorithm: {optimal_threshold}")
    print(
        f"The optimal threshold determined by Otsu's method: {optimal_threshold_otsu}")

    # Display the original, thresholded (Iterative), and thresholded (Otsu) images for comparison
    cv2.imshow("The Images ", np.hstack(
        [img, thresholded_img, thresholded_img_otsu]))

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
    plt.plot(bin_edges[:-1], hist_original, label='Histogram', color='blue')
    plt.plot(bin_edges[:-1], gaussian_curve(bin_edges[:-1], *params),
             label='Gaussian Curve', linestyle='--', color='red')
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
    cv2.imshow("Colored Images using iterative threshold",
               colored_image_iterative)

    # Edge detection using the Canny edge detector
    edges = cv2.Canny(thresholded_img, 30, 100)

    # Save the processed images (Iterative method)
    cv2.imwrite(os.path.join(output_directory,
                f'processed_{fits_filename}_iterative.png'), colored_image_iterative)

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
    cv2.imshow("Colored Images using iterative threshold",
               colored_image_iterative)

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
            thresholded_img * component_mask, maxCorners=100, qualityLevel=0.01, minDistance=0.1)

        num_corners = corners.shape[0] if corners is not None else 0

        print(f"Component {label} (Iterative): Area = {area_iterative}, Center = ({center_x}, {center_y}), Edge count = {edge_count}, Number of Corners = {num_corners}")

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
    cv2.imshow("Images", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Display the Gaussian curves for all FITS files
plt.figure(figsize=(10, 6))
for i, fits_filename in enumerate(fits_filenames):
    plt.plot(bin_edges[:-1], gaussian_curve(bin_edges[:-1], all_amplitudes[i], all_means[i], all_stddevs[i]),
             label=f'{fits_filename} Gaussian Curve')

plt.legend()
plt.title('Gaussian Curves for FITS Files')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()


# Create a list to store DataFrames for each FITS file
dfs = []

# Iterate over FITS files and extract header information
for fits_filename in fits_filenames:
    # Full path to the FITS file
    full_path_fits = os.path.join(fits_directory, fits_filename)

    # Open the FITS file
    with fits.open(full_path_fits) as hdul:
        # Access the header of the primary HDU
        header = hdul[0].header

        # Extract relevant information
        try:
            # Extracting the date of observation
            dateobs = header['DATE-OBS']
        except KeyError:
            dateobs = "Not available"

        try:
            # Extracting the number of axis
            naxis = header['NAXIS']
        except KeyError:
            naxis = "Not available"

        try:
            # Extracting the number of axis
            wavelen = header['WAVELENG']
        except KeyError:
            wavelen = "Not available"

        try:
            # Extracting Pixel Scale from the header
            pixel_scale = header['BITPIX']
        except KeyError:
            pixel_scale = "Not available"

        try:
            # Extracting Calibration Data (filter) from the header
            filter_used = header['FILTER']
        except KeyError:
            filter_used = "Not available"

        try:
            # Extracting Calibration Data ( exposure time ) from the header
            exposure_time = header['EXPOSURE']
        except KeyError:
            exposure_time = "Not available"

        try:
            # Extracting Calibration Data (  ) from the header
            bzero = header['BZERO']
        except KeyError:
            bzero = "Not available"

        # Extract the image data (pixel values)
        image_data = hdul[0].data

        # Calculate the intensity
        intensity = (image_data - bzero) * exposure_time

        # Create a DataFrame for the current FITS file
        df = pd.DataFrame({
            'FITS File': [fits_filename],
            'Number of Axis': [naxis],
            'Wave Length': [wavelen],
            'Date of Observation': [dateobs],
            'Pixel Scale': [pixel_scale],
            'Exposure Time': [exposure_time],
            'Physical Zero Value': [bzero],
            'Filter Used': [filter_used],
            'Intensity': [intensity.sum()]
        })

        # Append the DataFrame to the list
        dfs.append(df)

# Concatenate the list of DataFrames into a single DataFrame
df_combined = pd.concat(dfs, ignore_index=True)

# Display the resulting DataFrame
print(df_combined)


# Function to track objects using optical flow
def track_objects(image_directory):
    # Get the list of image files in the directory
    image_files = sorted([f for f in os.listdir(
        image_directory) if f.endswith('.png')])

    # Read the first frame
    prev_frame = cv2.imread(os.path.join(image_directory, image_files[0]))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Create an empty list to store object tracks
    object_tracks = []

    for i in range(1, len(image_files)):
        # Read the current frame
        frame = cv2.imread(os.path.join(image_directory, image_files[i]))

        # Convert the current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use the Lucas-Kanade method for optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Example region of interest (you may need to customize this)
        object_roi = prev_gray[100:200, 200:300]

        # Calculate the optical flow for the object
        object_flow = flow[100:200, 200:300]

        # Update the position of the object based on optical flow
        object_new_position = (
            200 + np.mean(object_flow[:, :, 0]), 100 + np.mean(object_flow[:, :, 1]))

        # Draw a rectangle around the tracked object
        cv2.rectangle(frame, (int(object_new_position[0]), int(object_new_position[1])),
                      (int(object_new_position[0] + 100), int(object_new_position[1] + 100)), (0, 255, 0), 2)

        # Update the previous frame and gray image
        prev_gray = gray.copy()

        # Append the object position to the list
        object_tracks.append(object_new_position)

        # Display the frame
        cv2_imshow(frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Close the window
    cv2.destroyAllWindows()

    return object_tracks


# Example usage
image_directory = '/content/drive/MyDrive/2024_001_images'
tracks = track_objects(image_directory)

# Print the tracked object positions at each frame
for i, track in enumerate(tracks):
    print(f'Frame {i + 1}: Object position {track}')


# on 2024 images

# Function for Gaussian curve

def gaussian_curve(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))


# Directory containing PNG images
png_directory = '/content/drive/MyDrive/2024_001_images'

# Output directory for processed images
output_directory = '/content/drive/MyDrive/Colab-Debris'

# List to store all center coordinates
all_center_coordinates = []

# Iterate over PNG files in the directory
for png_filename in os.listdir(png_directory):
    # Skip non-PNG files
    if not png_filename.lower().endswith('.png'):
        continue

    # Full path to the PNG file
    full_path_png = os.path.join(png_directory, png_filename)

    # Read the PNG image
    image = cv2.imread(full_path_png)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Plot histogram for original image
    hist_original, bin_edges = np.histogram(
        img.flatten(), bins=256, range=[0, 256])

    # Apply the iterative thresholding algorithm to the image
    optimal_threshold = iterative_thresholding(img)

    # Threshold the image using the optimal threshold
    thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

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
    cv2.waitKey(0)

    # Edge detection using the Canny edge detector
    edges = cv2.Canny(thresholded_img, 30, 100)

    # Save the processed images (Iterative method)
    cv2.imwrite(os.path.join(output_directory,
                f'processed_{png_filename}_iterative.png'), colored_image_iterative)

    # Print the area of each component (Iterative method)
    center_coordinates_list = []
    for label in range(1, num_labels_iterative):
        area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
        center_x, center_y = centroids_iterative[label]
        component_mask = (labels_iterative == label).astype(np.uint8)
        edges_in_component = cv2.bitwise_and(edges, edges, mask=component_mask)
        edge_count = np.count_nonzero(edges_in_component)
        corners = cv2.goodFeaturesToTrack(
            thresholded_img * component_mask, maxCorners=100, qualityLevel=0.01, minDistance=0.1)
        num_corners = corners.shape[0] if corners is not None else 0

        print(f"Component {label} (Iterative): Area = {area_iterative}, Center = ({center_x}, {center_y}), Edge count = {edge_count}, Number of Corners = {num_corners}")

        # Append center coordinates to the list
        center_coordinates_list.append(
            [label, area_iterative, center_x, center_y, edge_count, num_corners])

    # Add center coordinates to the overall list
    all_center_coordinates.extend(center_coordinates_list)

# Save all center coordinates to a single CSV
csv_filename = os.path.join(output_directory, 'output2024coordinates_all.csv')
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Label', 'Area', 'Center_X',
                    'Center_Y', 'Edge_Count', 'Num_Corners'])
    # Write data
    writer.writerows(all_center_coordinates)

print(f'All center coordinates saved to {csv_filename}')

cv2.waitKey(0)
cv2.destroyAllWindows()


# Function to draw trajectories on separate images for each object
def draw_individual_trajectories(coordinates, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over each object's components and draw trajectories
    for obj_id, _, x, y, _, _ in coordinates:
        obj_coordinates = [(int(x), int(y))]

        # Find coordinates for the same object
        for _, _, next_x, next_y, _, _ in coordinates:
            if obj_id == _ and (next_x, next_y) not in obj_coordinates:
                obj_coordinates.append((int(next_x), int(next_y)))

        # Create a black image
        img = np.zeros((512, 512, 3), dtype=np.uint8)

        # Draw trajectories on the image
        color = tuple(np.random.randint(0, 255, 3).tolist())
        for i in range(len(obj_coordinates) - 1):
            cv2.line(img, obj_coordinates[i], obj_coordinates[i+1], color, 2)

        # Save the image with the trajectory
        output_filename = os.path.join(
            output_directory, f'object_{obj_id}_trajectory.png')
        cv2.imwrite(output_filename, img)


# Read the output coordinates CSV
# Update with the actual path
csv_filename = '/content/drive/MyDrive/Colab-Debris/output2024coordinates_all.csv'
with open(csv_filename, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    coordinates = [list(map(float, row)) for row in reader]

# Draw individual trajectories and save the images
output_trajectory_directory = '/content/drive/MyDrive/Colab-Debris/individual2024_trajectories'
draw_individual_trajectories(coordinates, output_trajectory_directory)

print(f'Individual trajectory images saved to {output_trajectory_directory}')
