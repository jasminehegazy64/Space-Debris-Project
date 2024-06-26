import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from skimage import feature
from scipy.stats import norm
from threshold import iterative_thresholding  # Assuming you have defined this function

# Define the moment of inertia functions
def moment_of_inertia(xWidth, yHeight, xCG, yCG):
    Ixx = sum((y - yCG)**2 for y in yHeight)
    Iyy = sum((x - xCG)**2 for x in xWidth)
    Ixy = sum((x - xCG)*(y - yCG) for x, y in zip(xWidth, yHeight))
    return Ixx, Iyy, Ixy

def main_inertia(Ixx, Iyy, Ixy):
    Imain1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4*(Ixy)**2))
    Imain2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy)**2 + 4*(Ixy)**2))
    final_inertia = Imain1 / Imain2
    return Imain1, Imain2, final_inertia

def process_image(image_path, csv_writer):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to create a binary image
    binary_image, optimal_threshold = iterative_thresholding(image)

    # Perform connected component analysis to label blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

    # Create a copy of the original image for visualization
    image_with_blobs = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Set thresholds for standard deviation and area
    std_threshold = 30  # Adjust this threshold value as needed
    area_threshold = 500  # Adjust this threshold value as needed

    # Iterate over each detected blob (skip the first label, which is the background)
    for label in range(1, num_labels):
        # Extract bounding box coordinates
        x, y, w, h = stats[label][:4]  # Extract bounding box coordinates
        center_x, center_y = centroids[label]

        # Extract blob region from original image
        blob_mask = (labels == label).astype(np.uint8) * 255
        blob_pixels = image[blob_mask == 255]

        # Measure normal distribution parameters (mean, std deviation)
        mean_intensity = np.mean(blob_pixels)
        std_deviation = np.std(blob_pixels)

        # Fit a normal distribution to the blob's pixel intensities
        params = norm.fit(blob_pixels)
        mean_fit, std_deviation_fit = params

        # Determine the area of the blob
        area = stats[label][cv2.CC_STAT_AREA]

        # Calculate LBP features
        roi = image[y:min(y+h, image.shape[0]), x:min(x+w, image.shape[1])]
        lbp_features = feature.local_binary_pattern(roi, P=8, R=1, method='uniform')
        lbp_mean = np.mean(lbp_features)
        lbp_std = np.std(lbp_features)

        # Ensure xWidth and yHeight are iterable (lists)
        xWidth = list(range(w))
        yHeight = list(range(h))

        # Calculate moment of inertia
        Ixx, Iyy, Ixy = moment_of_inertia(xWidth, yHeight, center_x, center_y)
        Imain1, Imain2, final_inertia = main_inertia(Ixx, Iyy, Ixy)

        # Write results to CSV file
        csv_writer.writerow([label, mean_intensity, std_deviation, area, lbp_mean, lbp_std, Imain1, Imain2, final_inertia])

        # Determine the color of the blob based on std deviation and area
        if std_deviation >= std_threshold and area >= area_threshold:
            color = (0, 0, 255)  # Red for blobs with high std deviation and large area
        else:
            color = (0, 255, 0)  # Green otherwise

        # Draw bounding box or centroid of the blob on image_with_blobs
        cv2.rectangle(image_with_blobs, (x, y), (x + w, y + h), color, 2)  # Draw rectangle

        # Display area as text near the blob
        label_text = f"{label}"
        cv2.putText(image_with_blobs, label_text, (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return image_with_blobs

# List of image paths to process
image_paths = [
    'E:\\finalGPbegad\\images\\NEOS_SCI_2024001000555.png',

    # Add more image paths as needed
]

# Open a CSV file for writing results
csv_filename = 'blob_analysis_results.csv'
with open(csv_filename, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Blob ID', 'Mean Intensity', 'Std Deviation', 'Area', 'LBP Mean', 'LBP Std', 'Imain1', 'Imain2', 'Final Inertia'])

    # Process each image
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        image_with_blobs = process_image(image_path, csv_writer)

        # Save and display the image with detected blobs
        image_with_blobs_filename = f'{image_path}_with_blobs.png'
        cv2.imwrite(image_with_blobs_filename, image_with_blobs)

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image_with_blobs, cv2.COLOR_BGR2RGB))
        plt.title('Image with Detected Blobs')
        plt.axis('off')
        plt.show()

print(f"Results saved to {csv_filename}")
