import cv2 
import os 
import numpy as np 
def iterative_thresholding_folder(input_folder, output_folder):
    """
    Apply iterative thresholding to images in the input folder and save the thresholded images in the output folder.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply iterative thresholding
        thresholded_img, guessed_threshold = iterative_thresholding(img)

        # Write the thresholded image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, thresholded_img)

        print(f"Thresholded image saved: {output_path}")

def iterative_thresholding(img):
    """
    Apply iterative thresholding to the input image and return the thresholded image along with the guessed threshold.
    """
    # Initial guess for threshold
    threshold = 128

    # Loop until convergence
    while True:
        # Threshold the image using the current threshold
        _, thresholded_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        # Calculate the mean pixel values for foreground and background
        foreground_mean = np.mean(img[thresholded_img == 255])
        background_mean = np.mean(img[thresholded_img == 0])

        # Update the threshold using Otsu's method
        new_threshold = (foreground_mean + background_mean) / 2

        # Check for convergence
        if abs(threshold - new_threshold) < 0.5:
            break

        # Update the threshold
        threshold = new_threshold

    return thresholded_img, threshold

