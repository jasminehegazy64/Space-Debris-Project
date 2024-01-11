def iterative_thresholding(image, initial_threshold=128, max_iterations=50, tolerance=1e-3):
    threshold = initial_threshold

    for iteration in range(max_iterations):
        # Segment the image into foreground and background based on the threshold
        foreground = image >= threshold
        background = image < threshold

        # Compute the mean intensity of each group
        foreground_mean = np.mean(image[foreground])
        background_mean = np.mean(image[background])

        # Compute the new threshold as the average of the means
        new_threshold = (foreground_mean + background_mean) / 2.0

        # Check for convergence
        if abs(new_threshold - threshold) < tolerance:
            break

        threshold = new_threshold

    return threshold

def otsu_thresholding(image):
    # Ensure the image is of type np.uint8
    image = np.uint8(image)

    # Apply Otsu's thresholding
    # minimize intraclass variance
    _, thresholded_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Get the optimal threshold value determined by Otsu's method
    optimal_threshold = _

    return thresholded_img, optimal_threshold