import cv2 
import os 

### QUESTION: CAN THIS BE DONE BETTER IF THE FRAMES ARE VIDEOS? ###
def detect_objects(binary_image):
    # Find contours of objects in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize an empty list to store object information
    objects = []

    # Initialize object counter
    object_count = 0

    # Iterate through each contour
    for contour in contours:
        # Compute the area of the contour
        area = cv2.contourArea(contour)

        # Filter out small objects (noise)
        if area >=0:  # You can adjust this threshold based on your image characteristics
            # Compue the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Append object information to the list
            objects.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area})

            # Increment object counter
            object_count += 1

            # Draw number on detected object
            cv2.putText(binary_image, str(object_count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return objects, binary_image



