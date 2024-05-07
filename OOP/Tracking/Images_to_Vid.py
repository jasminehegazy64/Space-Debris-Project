import cv2 
import os 
import numpy as np 
def images_to_video(image_folder, output_video_path, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    if not images:
        print("No images found in the specified folder.")
        return

    images.sort()  # Ensure images are sorted in the correct order

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    if frame is None:
        print("Unable to read the first image.")
        return
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # Codec to use for video writing
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not video.isOpened():
        print("Error opening video writer.")
        return

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Unable to read image: {img_path}")
            continue
        video.write(img)

    video.release()
    cv2.destroyAllWindows()


