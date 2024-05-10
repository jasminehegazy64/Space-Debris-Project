import astropy
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
from PIL import Image
# from google.colab.patches import cv2.imshow
import os
import numpy as np
from scipy.optimize import curve_fit
# from google.colab import drive
from Convert_debris import convert_fits_to_image
from threshold import iterative_thresholding, otsu_thresholding
from gaussian_curve import gaussian_curve
import pandas as pd


fits_directory = "C:\\Users\\ASUS\Desktop\\Space-Debris-Project\\dataset"
output_directory = (
    "C:\\Users\\ASUS\\Desktop\\Space-Debris-Project\\dataset\\output_files"
)

fits_filenames = [
    "space5.fits",
    "tria.fits",
    "please4.fits",
    "space8.fits",
    "space6.fits",
    "space3.fits",
]  

bin_edges = None

all_amplitudes = []
all_means = []
all_stddevs = []


for fits_filename in fits_filenames:
    full_path_fits = os.path.join(fits_directory, fits_filename)

    output_image_filename = os.path.join(
        output_directory, os.path.splitext(fits_filename)[0] + "_preprocessed.png"
    )

    convert_fits_to_image(full_path_fits, output_image_filename)

    image = cv2.imread(output_image_filename)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist_original, bin_edges = np.histogram(img.flatten(), bins=256, range=[0, 256])

    p0 = [1.0, np.mean(img), np.std(img)]
    params, _ = curve_fit(gaussian_curve, bin_edges[:-1], hist_original, p0=p0)

    all_amplitudes.append(params[0])
    all_means.append(params[1])
    all_stddevs.append(params[2])

    optimal_threshold = iterative_thresholding(img)
    thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255
    thresholded_img_otsu, optimal_threshold_otsu = otsu_thresholding(img)

    print(f"\nResults for Thresholding (FITS file: {fits_filename}):")
    print(
        f"The optimal threshold determined by the iterative algorithm: {optimal_threshold}"
    )
    print(
        f"The optimal threshold determined by Otsu's method: {optimal_threshold_otsu}"
    )

    cv2.imshow("The Images ", np.hstack([img, thresholded_img, thresholded_img_otsu]))

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(hist_original, color="blue")
    plt.title("Histogram for Original Image")
    plt.ylabel("Frequency")
    plt.subplot(2, 2, 2)
    hist_thresholded = cv2.calcHist([thresholded_img], [0], None, [256], [0, 256])
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
    plt.plot(
        bin_edges[:-1],
        gaussian_curve(bin_edges[:-1], *params),
        label="Gaussian Curve",
        linestyle="--",
        color="red",
    )
    plt.legend()
    plt.title(f"Histogram with Gaussian Curve for {fits_filename}")
    plt.ylabel("Frequency")
    plt.show()
    plt.tight_layout()
    plt.show()

    # Connected components labeling for thresholded image (Iterative method) becuase the iterative method gets better results
    num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = (
        cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)
    )

    colors_iterative = np.random.randint(
        0, 255, size=(num_labels_iterative, 3), dtype=np.uint8
    )

    colored_image_iterative = colors_iterative[labels_iterative]
    cv2.imshow("Colored Images using iterative threshold", colored_image_iterative)

    edges = cv2.Canny(thresholded_img, 30, 100)

    cv2.imwrite(
        os.path.join(output_directory, f"processed_{fits_filename}_iterative.png"),
        colored_image_iterative,
    )

   
    for label in range(1, num_labels_iterative):
        area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
        component_mask = (labels_iterative == label).astype(np.uint8)
        edges_in_component = cv2.bitwise_and(edges, edges, mask=component_mask)
        edge_count = np.count_nonzero(edges_in_component)

        # Apply Shi-Tomasi corner detection to the current component ROI
        corners = cv2.goodFeaturesToTrack(
            thresholded_img * component_mask,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=0.1,
        )

        num_corners = corners.shape[0] if corners is not None else 0

    num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = (
        cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)
    )

    colors_iterative = np.random.randint(
        0, 255, size=(num_labels_iterative, 3), dtype=np.uint8
    )

    colored_image_iterative = colors_iterative[labels_iterative]
    cv2.imshow("Colored Images using iterative threshold", colored_image_iterative)

    for label in range(1, num_labels_iterative):
        area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
        center_x, center_y = centroids_iterative[label]
        component_mask = (labels_iterative == label).astype(np.uint8)
        edges_in_component = cv2.bitwise_and(edges, edges, mask=component_mask)

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

    num_white_objects_iterative = (
        num_labels_iterative - 1
    )  # Subtract 1 for the background
    print(f"The number of white objects (Iterative) is: {num_white_objects_iterative}")

    num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = (
        cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)
    )
    object_id = 1

    # Iterate through each detected object
    for i in range(1, num_labels_iterative):
        x, y, w, h, area = stats_iterative[i]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
        object_id += 1

    cv2.imshow("Images", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

plt.figure(figsize=(10, 6))
for i, fits_filename in enumerate(fits_filenames):
    plt.plot(
        bin_edges[:-1],
        gaussian_curve(bin_edges[:-1], all_amplitudes[i], all_means[i], all_stddevs[i]),
        label=f"{fits_filename} Gaussian Curve",
    )

plt.legend()
plt.title("Gaussian Curves for FITS Files")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()


dfs = []

for fits_filename in fits_filenames:
    full_path_fits = os.path.join(fits_directory, fits_filename)

    with fits.open(full_path_fits) as hdul:
        header = hdul[0].header

        try:
            dateobs = header["DATE-OBS"]
        except KeyError:
            dateobs = "Not available"

        try:
            naxis = header["NAXIS"]
        except KeyError:
            naxis = "Not available"

        try:
            wavelen = header["WAVELENG"]
        except KeyError:
            wavelen = "Not available"

        try:
            pixel_scale = header["BITPIX"]
        except KeyError:
            pixel_scale = "Not available"

        try:
            filter_used = header["FILTER"]
        except KeyError:
            filter_used = "Not available"

        try:
            exposure_time = header["EXPOSURE"]
        except KeyError:
            exposure_time = "Not available"

        try:
            bzero = header["BZERO"]
        except KeyError:
            bzero = "Not available"

        image_data = hdul[0].data
        intensity = (image_data - bzero) * exposure_time

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

        dfs.append(df)

df_combined = pd.concat(dfs, ignore_index=True)
print(df_combined)


def track_objects(image_directory):
    # Get the list of image files in the directory
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith(".png")])
    prev_frame = cv2.imread(os.path.join(image_directory, image_files[0]))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    object_tracks = []

    for i in range(1, len(image_files)):
        frame = cv2.imread(os.path.join(image_directory, image_files[i]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Example region of interest
        object_roi = prev_gray[100:200, 200:300]
        object_flow = flow[100:200, 200:300]
        object_new_position = (
            200 + np.mean(object_flow[:, :, 0]),
            100 + np.mean(object_flow[:, :, 1]),
        )

        cv2.rectangle(
            frame,
            (int(object_new_position[0]), int(object_new_position[1])),
            (int(object_new_position[0] + 100), int(object_new_position[1] + 100)),
            (0, 255, 0),
            2,
        )

        prev_gray = gray.copy()
        object_tracks.append(object_new_position)

        cv2.imshow(frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return object_tracks


image_directory = "/content/drive/MyDrive/2024_001_images"
tracks = track_objects(image_directory)

for i, track in enumerate(tracks):
    print(f"Frame {i + 1}: Object position {track}")


def gaussian_curve(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c**2))


png_directory = "/content/drive/MyDrive/2024_001_images"
output_directory = "/content/drive/MyDrive/Colab-Debris"

all_center_coordinates = []

for png_filename in os.listdir(png_directory):
    if not png_filename.lower().endswith(".png"):
        continue

    full_path_png = os.path.join(png_directory, png_filename)
    image = cv2.imread(full_path_png)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist_original, bin_edges = np.histogram(img.flatten(), bins=256, range=[0, 256])
    optimal_threshold = iterative_thresholding(img)

    thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255
    num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = (
        cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)
    )

    colors_iterative = np.random.randint(
        0, 255, size=(num_labels_iterative, 3), dtype=np.uint8
    )

    colored_image_iterative = colors_iterative[labels_iterative]

    cv2.imshow(colored_image_iterative)
    cv2.waitKey(0)

    # Edge detection using the Canny edge detector
    edges = cv2.Canny(thresholded_img, 30, 100)
    cv2.imwrite(
        os.path.join(output_directory, f"processed_{png_filename}_iterative.png"),
        colored_image_iterative,
    )

    # Print the area of each component (Iterative method)
    center_coordinates_list = []
    for label in range(1, num_labels_iterative):
        area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
        center_x, center_y = centroids_iterative[label]
        component_mask = (labels_iterative == label).astype(np.uint8)
        edges_in_component = cv2.bitwise_and(edges, edges, mask=component_mask)
        edge_count = np.count_nonzero(edges_in_component)
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

        center_coordinates_list.append(
            [label, area_iterative, center_x, center_y, edge_count, num_corners]
        )

    all_center_coordinates.extend(center_coordinates_list)

# Save all center coordinates to a single CSV
csv_filename = os.path.join(output_directory, "output2024coordinates_all.csv")
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(
        ["Label", "Area", "Center_X", "Center_Y", "Edge_Count", "Num_Corners"]
    )
    # Write data
    writer.writerows(all_center_coordinates)

print(f"All center coordinates saved to {csv_filename}")

cv2.waitKey(0)
cv2.destroyAllWindows()


def draw_individual_trajectories(coordinates, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for obj_id, _, x, y, _, _ in coordinates:
        obj_coordinates = [(int(x), int(y))]

        for _, _, next_x, next_y, _, _ in coordinates:
            if obj_id == _ and (next_x, next_y) not in obj_coordinates:
                obj_coordinates.append((int(next_x), int(next_y)))

        img = np.zeros((512, 512, 3), dtype=np.uint8)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        for i in range(len(obj_coordinates) - 1):
            cv2.line(img, obj_coordinates[i], obj_coordinates[i + 1], color, 2)

        # Save the image with the trajectory
        output_filename = os.path.join(
            output_directory, f"object_{obj_id}_trajectory.png"
        )
        cv2.imwrite(output_filename, img)


csv_filename = "/content/drive/MyDrive/Colab-Debris/output2024coordinates_all.csv"
with open(csv_filename, mode="r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    coordinates = [list(map(float, row)) for row in reader]

# Draw individual trajectories and save the images
output_trajectory_directory = (
    "/content/drive/MyDrive/Colab-Debris/individual2024_trajectories"
)
draw_individual_trajectories(coordinates, output_trajectory_directory)

print(f"Individual trajectory images saved to {output_trajectory_directory}")
