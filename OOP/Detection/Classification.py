import os
import cv2
import numpy as np
import csv
from skimage import feature
from conversion import convert_fits_to_image
from images_Preprocessing.iterative_Threshholding import iterative_thresholding

class DebrisAnalyzer:
    def __init__(self, threshed_directory, csv_file_path):
        
        self.threshed_directory = threshed_directory
        self.csv_file_path = csv_file_path

    def moment_of_inertia(self, xWidth, yHeight, xCG, yCG):
        Ixx = sum((y - yCG)**2 for y in yHeight)
        Iyy = sum((x - xCG)**2 for x in xWidth)
        Ixy = sum((x - xCG)*(y - yCG) for x, y in zip(xWidth, yHeight))
        return Ixx, Iyy, Ixy

    def main_inertia(self, Ixx, Iyy, Ixy, yHeight, xWidth):
        Imain1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4*(Ixy)**2))
        Imain2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy)**2 + 4*(Ixy)**2))
        epsilonn = 10
        final_inertia = Imain1 / Imain2
        if final_inertia > epsilonn:
            return 'Debris'
        else:
            return 'Celestial Object'

    def process_images(self):
        fitsfiles=os.listdir(self.threshed_directory)

        # Open the CSV file in write mode
        with open(self.csv_file_path, 'w', newline='') as csvfile:
            # Create a CSV writer
            csvwriter = csv.writer(csvfile)

            # Write the header row
            csvwriter.writerow(['Image', 'Object ID','Area','Edges','Center_x','Center_y','Width','Height','lbp_mean','lbp_std' ,'Prediction'])

            for fits_filename in fitsfiles:
                # Full path to the FITS file
                full_path_fits = os.path.join(self.threshed_directory, fits_filename)

                # Output PNG filename (assuming the same name with a different extension)
                output_image_filename = os.path.join(self.threshed_directory, os.path.splitext(fits_filename)[0] + '.png')
                #convert_fits_to_image(full_path_fits, output_image_filename)

                image = cv2.imread(output_image_filename)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # # Apply the iterative thresholding algorithm to the image
                # optimal_threshold = self.iterative_thresholding(img)

                # # Threshold the image using the optimal threshold
                # thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

                num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = cv2.connectedComponentsWithStats(
                    img, connectivity=8)

                # Reset object_id for each new image
                object_id = 1

                for label in range(1, num_labels_iterative):
                    area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
                    component_mask = (labels_iterative == label).astype(np.uint8)
                    center_x, center_y = centroids_iterative[label]
                    # Multiply the component mask with the edges to get edges within the component
                    edges = cv2.Canny(img, 30, 100)
                    edges_in_component = cv2.bitwise_and(edges, edges, mask=component_mask)

                    # Get the coordinates of the bounding box for the current object
                    x, y, w, h, area = stats_iterative[label]
                    # Count the number of edges in the component
                    edge_count = np.count_nonzero(edges_in_component)

                    # Extract the region of interest (ROI)
                    roi = img[y:min(y+h, img.shape[0]), x:min(x+w, img.shape[1])]

                    # Compute Local Binary Pattern (LBP) features
                    lbp_features = feature.local_binary_pattern(roi, P=8, R=1, method='uniform')
                    lbp_mean = np.mean(lbp_features)
                    lbp_std = np.std(lbp_features)

                    # Ensure xWidth and yHeight are iterable (lists)
                    xWidth = list(range(w))
                    yHeight = list(range(h))

                    # Print the coordinates of the bounding box
                    print(f"Object {object_id} in {fits_filename}:")

                    cv2.putText(image, str(object_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Increment Id
                    object_id += 1

                    Ixx, Iyy, Ixy = self.moment_of_inertia(xWidth, yHeight, center_x, center_y)
                    prediction = self.main_inertia(Ixx, Iyy, Ixy, yHeight, xWidth)

                    # Write the row to the CSV file
                    csvwriter.writerow([fits_filename, object_id - 1, area_iterative, edge_count, center_x, center_y, w, h, lbp_mean, lbp_std, prediction])
