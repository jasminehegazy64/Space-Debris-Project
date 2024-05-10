import numpy as np
import cv2


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

        if abs(new_threshold - threshold) < tolerance:
            break

        threshold = new_threshold
    return threshold


def otsu_thresholding(image):
    image = np.uint8(image)
    _, thresholded_img = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    optimal_threshold = _

    return thresholded_img, optimal_threshold
