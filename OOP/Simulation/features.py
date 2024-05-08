# import numpy as np
# from astropy.io import fits
# import csv
# import os 
# import cv2 

# fits_file_path = r'C:\Users\USER\Desktop\TechnicalGP\2024-001\NEOS_SCI_2024001000312.fits'
# def extract_debris_data(fits_file_path):
#     try:
#         hdulist = fits.open(fits_file_path)
#         avg_vel = hdulist[0].header['AVG_VEL']
#         ra_vel = hdulist[0].header['RA_VEL']
#         dec_vel = hdulist[0].header['DEC_VEL']
#         rol_vel = hdulist[0].header['ROL_VEL']
#         hdulist.close()
#         return avg_vel, ra_vel, dec_vel, rol_vel
#     except Exception as e:
#         print("Error:", e)
#         return None

# def write_to_csv(csv_file_path, data):
#     try:
#         with open(csv_file_path, 'w', newline='') as csvfile:
#             fieldnames = ['AVG_VEL', 'RA_Vel', 'DEC_vel', 'Rol_vel']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerow(dict(zip(fieldnames, data)))
#         print("Data has been written to CSV file:", csv_file_path)
#     except Exception as e:
#         print("Error:", e)

# def main():
    
#     csv_file_path = r'C:\Users\USER\Desktop\TechnicalGP\sim_debris.xlsx'
#     debris_data = extract_debris_data(fits_file_path)
#     if debris_data:
#         print("Data extraction successful:")
#         print("AVG_VEL:", debris_data[0])
#         print("RA_Vel:", debris_data[1])
#         print("DEC_vel:", debris_data[2])
#         print("Rol_vel:", debris_data[3])
#         write_to_csv(csv_file_path, debris_data)
#     else:
#         print("Failed to extract data from FITS file.")

# if __name__ == "__main__":
#     main()


# def count_white_blobs_in_folder(folder_path):
#     # Get a list of all files in the folder
#     files = os.listdir(folder_path)

#     # Loop through each file in the folder
#     for file in files:
#         # Check if the file is an image (you can add more file extensions if needed)
#         if file.endswith((".jpg", ".jpeg", ".png")):
#             # Construct the full path to the image file
#             image_path = os.path.join(folder_path, file)

#             # Read the image in grayscale
#             image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#             # Apply threshold to binarize the image
#             _, binary_image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)

#             # Find contours in the binary image
#             contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             # Count the number of contours (white blobs)
#             num_blobs = len(contours)

#             # Print the number of blobs for the current image
#             print(f"Number of white blobs in {file}: {num_blobs}")

# # Example usage: Provide the path to the folder containing images
# folder_path = r"C:\Users\USER\Desktop\TechnicalGP\images_Preprocessing\iter_images"
# count_white_blobs_in_folder(folder_path)



#TRY TWO :
import numpy as np
from astropy.io import fits
import csv
import os 
import cv2 

def extract_debris_data(fits_file_path):
    try:
        hdulist = fits.open(fits_file_path)
        avg_vel = hdulist[0].header['AVG_VEL']
        ra_vel = hdulist[0].header['RA_VEL']
        dec_vel = hdulist[0].header['DEC_VEL']
        rol_vel = hdulist[0].header['ROL_VEL']
        hdulist.close()
        return avg_vel, ra_vel, dec_vel, rol_vel
    except Exception as e:
        print("Error:", e)
        return None

def write_to_csv(csv_file_path, data):
    try:
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['AVG_VEL', 'RA_Vel', 'DEC_vel', 'Rol_vel', 'Num_Blobs']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(data)
        print("Data has been written to CSV file:", csv_file_path)
    except Exception as e:
        print("Error:", e)

def count_white_blobs(image):
    # Apply threshold to binarize the image
    _, binary_image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of contours (white blobs)
    num_blobs = len(contours)
    
    return num_blobs

def process_image(image_path):
    try:
        # Read the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Count the number of white blobs
        num_blobs = count_white_blobs(image)

        return num_blobs
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def main():
    fits_file_path = r'C:\Users\USER\Desktop\TechnicalGP\2024-001\NEOS_SCI_2024001000312.fits'
    csv_file_path = r'C:\Users\USER\Desktop\TechnicalGP\sim_debris.xlsx'
    folder_path = r"C:\Users\USER\Desktop\TechnicalGP\images_Preprocessing\iter_images\NEOS_SCI_2024001000312.png"

    debris_data = extract_debris_data(fits_file_path)
    if debris_data:
        print("Data extraction successful:")
        print("AVG_VEL:", debris_data[0])
        print("RA_Vel:", debris_data[1])
        print("DEC_vel:", debris_data[2])
        print("Rol_vel:", debris_data[3])

        # Process images in the folder
        blob_data = []
        for file in os.listdir(folder_path):
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, file)
                num_blobs = process_image(image_path)
                if num_blobs is not None:
                    blob_data.append(num_blobs)

        # Add the number of blobs to the debris data
        debris_data_with_blobs = list(debris_data) + [sum(blob_data)]

        # Write debris data with blob count to CSV
        write_to_csv(csv_file_path, dict(zip(['AVG_VEL', 'RA_Vel', 'DEC_vel', 'Rol_vel', 'Num_Blobs'], debris_data_with_blobs)))
    else:
        print("Failed to extract data from FITS file.")

if __name__ == "__main__":
    main()
