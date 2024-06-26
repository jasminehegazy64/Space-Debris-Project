import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from scipy.stats import norm
from threshold import iterative_thresholding

# Load your image (replace 'image.jpg' with your image path)
image_path = 'E:\\finalGPbegad\\images\\NEOS_SCI_2024001000555.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Perform preprocessing if necessary (e.g., smoothing, noise reduction)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply thresholding to create a binary image
binary_image, optimal_threshold = iterative_thresholding(image)

# Perform connected component analysis to label blobs
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

# Create a copy of the original image for visualization
image_with_blobs = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Set thresholds for standard deviation and area
std_threshold = 30  # Adjust this threshold value as needed
area_threshold = 5 # Adjust this threshold value as needed

# Iterate over each detected blob (skip the first label, which is the background)
for label in range(1, num_labels):
    # Extract blob region from original image
    blob_mask = (labels == label).astype(np.uint8) * 255
    blob_pixels = image[blob_mask == 255]
    
    # Measure normal distribution parameters (mean, std deviation)
    mean_intensity = np.mean(blob_pixels)
    std_deviation = np.std(blob_pixels)
    
    # Optionally, fit a normal distribution to the blob's pixel intensities
    params = norm.fit(blob_pixels)
    mean_fit, std_deviation_fit = params
    
    # Print the distribution parameters
    print(f"Blob {label}: Mean = {mean_intensity}, Std Deviation = {std_deviation}")
    
    # Determine the area of the blob
    area = stats[label][cv2.CC_STAT_AREA]
    
    # # Calculate LBP features
    # roi = image[y:y+h, x:x+w]
    # lbp_features = feature.local_binary_pattern(roi, P=8, R=1, method='uniform')
    # lbp_mean = np.mean(lbp_features)
    # lbp_std = np.std(lbp_features)
    
    # print(f"Blob {label}: LBP Mean = {lbp_mean}, LBP Std = {lbp_std}")
    
    # Determine the color of the blob based on std deviation and area
    if std_deviation >= std_threshold and area >= area_threshold:
        color = (0, 0, 255)  # Red for blobs with high std deviation and large area
        print(f"Blob {label} (Red): Area = {area} pixels")
    else:
        color = (0, 255, 0)  # Green otherwise
    
    # Draw bounding box or centroid of the blob on image_with_blobs
    x, y, w, h = stats[label][:4]  # Extract bounding box coordinates
    cv2.rectangle(image_with_blobs, (x, y), (x + w, y + h), color, 2)  # Draw rectangle
    
    # # Display area as text near the blob
    # label_text = f"Area={area}"
    # cv2.putText(image_with_blobs, label_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# Display the thresholded binary image
plt.figure(figsize=(8, 6))
plt.imshow(binary_image, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')
plt.show()

# Display the image with detected blobs
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image_with_blobs, cv2.COLOR_BGR2RGB))
plt.title('Image with Detected Blobs')
plt.axis('off')
plt.show()
