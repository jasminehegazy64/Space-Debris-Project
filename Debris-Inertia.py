import pandas as pd
import numpy as np
import math
import os
import cv2
from astropy.io import fits
import matplotlib.pyplot as plt
from PIL import Image
from math import *
import csv


def convert_fits_to_image(fits_filename, output_image_filename):
    with fits.open(fits_filename) as hdul:
        data = hdul[0].data
        data = cv2.GaussianBlur(data, (5, 5), 0)

        laplacian = cv2.Laplacian(data, cv2.CV_64F)
        sharpened = data - 0.8 * laplacian

        plt.imshow(sharpened, cmap="gray")
        plt.axis("off")
        plt.savefig(output_image_filename, bbox_inches="tight", pad_inches=0)
        plt.close()


def iterative_thresholding(
    image, initial_threshold=128, max_iterations=50, tolerance=1e-3
):
    threshold = initial_threshold

    for iteration in range(max_iterations):
        foreground = image >= threshold
        background = image < threshold

        foreground_mean = np.mean(image[foreground])
        background_mean = np.mean(image[background])

        new_threshold = (foreground_mean + background_mean) / 2.0

        # Check for convergence
        if abs(new_threshold - threshold) < tolerance:
            break

        threshold = new_threshold
    return threshold


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


fits_directory = "/content/drive/MyDrive/Colab-Debris"
output_directory = "/content/drive/MyDrive/Colab-Debris/output_images"

csv_file_path = "/content/drive/MyDrive/Colab-Debris/output_images/InetriaOutPut.csv"
fits_filenames = [
    "space5.fits",
    "tria.fits",
    "please4.fits",
    "space8.fits",
    "space6.fits",
    "space3.fits",
]

with open(csv_file_path, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(["Image", "Object ID", "Prediction"])

    for fits_filename in fits_filenames:
        full_path_fits = os.path.join(fits_directory, fits_filename)

        output_image_filename = os.path.join(
            output_directory, os.path.splitext(fits_filename)[0] + "_preprocessed.png"
        )
        convert_fits_to_image(full_path_fits, output_image_filename)

        image = cv2.imread(output_image_filename)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        optimal_threshold = iterative_thresholding(img)

        thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

        num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = (
            cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)
        )

        object_id = 1

        for label in range(1, num_labels_iterative):
            center_x, center_y = centroids_iterative[label]

            x, y, w, h, area = stats_iterative[label]

            xWidth = list(range(w))
            yHeight = list(range(h))

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

            object_id += 1

            Ixx, Iyy, Ixy = momentOfInertia(xWidth, yHeight, center_x, center_y)
            finalint = mainInteria(Ixx, Iyy, Ixy, yHeight, xWidth)
            csvwriter.writerow(
                [
                    fits_filename,
                    object_id - 1,
                    "Debris" if finalint > 10 else "Celestial Object",
                ]
            )

fits_directory = "/content/drive/MyDrive/Colab-Debris/"

fits_filenames = [
    "space5.fits",
    "tria.fits",
    "please4.fits",
    "space8.fits",
    "space6.fits",
    "space3.fits",
]

fits_files_list = []
haralick_mean_list = []
haralick_std_list = []
lbp_hist_list = []

for fits_filename in fits_filenames:
    full_path_fits = os.path.join(fits_directory, fits_filename)

    with fits.open(full_path_fits) as hdul:
        data = hdul[0].data

    image = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    haralick_features = texture.haralick(image)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )

    fits_files_list.append(fits_filename)
    haralick_mean_list.append(np.mean(haralick_features, axis=0))
    haralick_std_list.append(np.std(haralick_features, axis=0))
    lbp_hist_list.append(lbp_hist)

    print(f"FITS File: {fits_filename}")
    print("Haralick Texture Features:")
    print("Mean:", np.mean(haralick_features, axis=0))
    print("Standard Deviation:", np.std(haralick_features, axis=0))
    print("Local Binary Pattern Histogram:", lbp_hist)
    print("-" * 50)


df_second = pd.DataFrame(
    {
        "FITS File": fits_files_list,
        "Haralick Mean": haralick_mean_list,
        "Haralick Std": haralick_std_list,
        "LBP Histogram": lbp_hist_list,
    }
)

csv_filename = "/content/drive/MyDrive/Colab-Debris/texture_features.csv"

df_combined = pd.merge(df_one, df_second, on="FITS File", how="outer")
df_combined.to_csv(csv_filename, index=False)
