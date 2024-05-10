from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
import pandas as pd
import os
from PIL import Image
import csv
import numpy as np
import torch
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2


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


os.chdir(r"/content/drive/MyDrive/yolov5-master")

# Define custom class names
custom_classes = ["celestial object", "space debris"]


def detect_objects(image_path, model, classes, confidence_threshold=0.001):
    # Convert image to RGB color space
    img = Image.open(image_path).convert("RGB")
    img = img.resize((640, 640))  # Resize image to model input size (640x640)
    # Convert image to tensor and normalize
    img_tensor = torch.tensor(np.array(img) / 255.0).permute(2, 0, 1).float()
    results = model(img_tensor.unsqueeze(0))[0]  # Perform object detection
    return results


# Load pre-trained YOLOv5 large model
model = torch.hub.load("ultralytics/yolov5", "yolov5l", pretrained=True)

model.eval()
confidence_threshold = 0.001

fits_directory = "/content/drive/MyDrive/2024TOTAL"
output_directory = "/content/drive/MyDrive/2024_output"

# Iterate through each subdirectory in fits_directory
for root, dirs, files in os.walk(fits_directory):
    for fits_filename in files:
        if fits_filename.endswith(".fits"):
            full_path_fits = os.path.join(root, fits_filename)

            fits_folder_name = os.path.splitext(fits_filename)[0]
            fits_output_directory = os.path.join(output_directory, fits_folder_name)
            os.makedirs(fits_output_directory, exist_ok=True)

            output_image_filename = os.path.join(
                fits_output_directory, fits_folder_name + "_preprocessed.png"
            )

            convert_fits_to_image(full_path_fits, output_image_filename)

            results = detect_objects(output_image_filename, model, custom_classes)

            output_csv_path = os.path.join(
                output_directory, fits_folder_name + "_yolo_output.csv"
            )

            # Write results to CSV
            with open(output_csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for detection in results:  # Access detections using .xyxy attribute
                    class_idx = int(detection[5])
                    class_name = custom_classes[class_idx]
                    confidence = float(detection[4])
                    if confidence >= confidence_threshold:
                        coordinates = [int(coord) for coord in detection[:4]]
                        writer.writerow(
                            [
                                output_image_filename,
                                class_idx,
                                class_name,
                                confidence,
                                coordinates,
                            ]
                        )

print(
    "Object detection completed. Results saved in separate CSV files for each directory."
)


# YOLOv5 repository
os.chdir(r"/content/drive/MyDrive/yolov5-master")

# Define custom class names
custom_classes = ["celestial object", "space debris"]


def detect_objects(image_path, model, classes, confidence_threshold=0.001):
    # Convert image to RGB color space
    img = Image.open(image_path).convert("RGB")
    img = img.resize((640, 640))  # Resize image to model input size (640x640)
    # Convert image to tensor and normalize
    img_tensor = torch.tensor(np.array(img) / 255.0).permute(2, 0, 1).float()
    results = model(img_tensor.unsqueeze(0))[0]  # Perform object detection
    with open(output_csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(
                ["Image", "Object", "Description", "Confidence", "Coordinates"]
            )
        for detection in results:
            class_idx = int(detection[5])
            class_name = classes[class_idx]
            confidence = float(detection[4])
            if confidence >= confidence_threshold:
                coordinates = [int(coord) for coord in detection[:4]]
                writer.writerow(
                    [image_path, class_idx, class_name, confidence, coordinates]
                )


# Load pre-trained YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5x", pretrained=True)
model.eval()
image_dir = r"/content/drive/MyDrive/2022-001-fits/Threshed"
output_csv_path = r"/content/drive/MyDrive/2024_output/yolo_output.csv"

# Create CSV file and write headers
with open(output_csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image", "Object", "Description", "Confidence", "Coordinates"])

for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)
    detect_objects(image_path, model, custom_classes)

print("Object detection completed. Results saved in yolo_output.csv")

csv_filename = "/content/drive/MyDrive/2024_output/yolo_output.csv"
df = pd.read_csv(csv_filename)

print(df)

custom_classes = ["celestial object", "space debris"]


def detect_objects(image_path, model, classes, confidence_threshold=0.001):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img)
    model.eval()

    with torch.no_grad():
        predictions = model([img_tensor])

    # Write results to CSV
    with open(output_csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(
                ["Image", "Object", "Description", "Confidence", "Coordinates"]
            )
        for idx, detection in enumerate(predictions[0]["boxes"]):
            class_idx = int(predictions[0]["labels"][idx])
            class_name = classes[class_idx]
            confidence = float(predictions[0]["scores"][idx])
            if confidence >= confidence_threshold:
                coordinates = [int(coord) for coord in detection]
                writer.writerow(
                    [image_path, class_idx, class_name, confidence, coordinates]
                )


# Load pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)

image_dir = "/content/drive/MyDrive/2022-001-fits/Threshed"
output_csv_path = "/content/drive/MyDrive/2024_output/faster_rcnn_output.csv"

# Create CSV file and write headers
with open(output_csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image", "Object", "Description", "Confidence", "Coordinates"])

for filename in os.listdir(image_dir):
    if filename.startswith("processed_with") and filename.endswith(
        "fits_iterative.png"
    ):
        image_path = os.path.join(image_dir, filename)
        detect_objects(image_path, model, custom_classes)

print("Object detection completed. Results saved in faster_rcnn_output.csv")
