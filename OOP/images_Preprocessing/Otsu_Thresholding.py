import cv2 
import os 
def otsu_thresholding_folder(input_folder, output_folder):
    """
    Apply Otsu's thresholding to images in the input folder and save the thresholded images in the output folder.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply Otsu's thresholding
        _, thresholded_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Write the thresholded image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, thresholded_img)

        print(f"Thresholded image saved: {output_path}")
