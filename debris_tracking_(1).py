# from google.colab import drive

drive.mount("/content/drive")

import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import cv2
from PIL import Image
from google.colab.patches import cv2_imshow
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
from skimage.color import label2rgb

detected_directory = "/content/drive/MyDrive/2024_001_trial/2024_001_detected"
vid_path = "/content/drive/MyDrive/2024_001_trial/2024_001_classification/ClearVid.MP4"
Opti_vid_path = (
    "/content/drive/MyDrive/2024_001_trial/2024_001_classification/opticalflow.MP4"
)

output_directory = "/content/drive/MyDrive/2024_001/2024_001_images"
binary_directory = "/content/drive/MyDrive/2024_001/2024_001_binary"

fitsfiles = os.listdir(output_directory)
fitsfiles


# Function to convert image to binary
def convert_to_binary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


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


def extract_properties(binary_image):
    num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = (
        cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    )
    area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
    component_mask = (labels_iterative == label).astype(np.uint8)
    center_x, center_y = centroids_iterative[label]
    x, y, w, h, area = stats_iterative[label]
    return x, y, w, h, area, center_x, center_y


for filename in os.listdir(output_directory):
    image_path = os.path.join(output_directory, filename)
    image = cv2.imread(image_path)
    binary_image = convert_to_binary(image)
    output_path = os.path.join(
        binary_directory, f"{os.path.splitext(filename)[0]}_binary.png"
    )
    cv2.imwrite(output_path, binary_image)
print("Conversion complete.")


csv_file_path = (
    "/content/drive/MyDrive/2024_001/Classification_Inertia/2024-001-outputs.xlsx"
)
with open(csv_file_path, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        [
            "Image",
            "Object ID",
            "x",
            "y",
            "w",
            "h",
            "area",
            "center_x",
            "center_y",
            "Classes",
        ]
    )

    # Loop over each file in the folder
    for filename in os.listdir(binary_directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(binary_directory, filename)
            binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            (
                num_labels_iterative,
                labels_iterative,
                stats_iterative,
                centroids_iterative,
            ) = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

            object_id = 1
            for label in range(1, num_labels_iterative):
                # Extract properties
                x, y, w, h, areas, center_x, center_y = extract_properties(binary_image)

                object_id += 1
                xWidth = list(range(w))
                yHeight = list(range(h))

                Ixx, Iyy, Ixy = momentOfInertia(xWidth, yHeight, center_x, center_y)
                finalint = mainInteria(Ixx, Iyy, Ixy, yHeight, xWidth)

                csvwriter.writerow(
                    [
                        filename,
                        object_id - 1,
                        x,
                        y,
                        w,
                        h,
                        areas,
                        center_x,
                        center_y,
                        "Debris" if finalint > 10 else "Celestial Object",
                    ]
                )


def segment(image, threshold=None):
    if threshold is None:
        threshold = threshold_otsu(image)
    bw = closing(image > threshold, square(3))
    cleared = clear_border(bw)

    labeled_image = label(cleared)
    regions = regionprops(labeled_image)

    return labeled_image, regions


def draw_bounding_boxes(image, regions):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(minc, minr, str(region.label), color="red", fontsize=12, weight="bold")

    ax.set_axis_off()
    plt.show()


for filename in os.listdir(binary_directory):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(binary_directory, filename)
        grey_scale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        labeled_image, regions = segment(grey_scale)
        draw_bounding_boxes(grey_scale, regions)
        output_path2 = os.path.join(
            detected_directory, f"{os.path.splitext(filename)[0]}_detected.png"
        )
        cv2.imwrite(output_path2, grey_scale)


def images_to_video(image_folder, output_video_path, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    if not images:
        print("No images found in the specified folder.")
        return
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    if frame is None:
        print("Unable to read the first image.")
        return
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
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


fps = 5  # Frames per second
images_to_video(binary_directory, vid_path, fps)


def calculate_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    color = (0, 255, 0)

    # Read the first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error reading video file.")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Initialize points for optical flow
    p0 = cv2.goodFeaturesToTrack(
        old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is None:
            print("Error: Optical flow points are None. Skipping frame.")
            continue

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
            frame = cv2.circle(frame, (a, b), 5, color, -1)

        img = cv2.add(frame, mask)
        cv2.imshow(img)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()


calculate_optical_flow(vid_path)


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


cap = cv2.VideoCapture(vid_path)

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    Opti_vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
)

while True:
    suc, img = cap.read()
    if not suc:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = time.time()
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray
    end = time.time()
    fps = 1 / (end - start)
    print(f"{fps:.2f} FPS")

    flow_img = draw_flow(gray, flow)
    hsv_img = draw_hsv(flow)

    out.write(hsv_img)

    cv2.imshow(flow_img)
    cv2.imshow(hsv_img)

    key = cv2.waitKey(5)
    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


def calculate_centroids(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    return centroids


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "output_video_with_centroids.mp4",
        fourcc,
        30.0,
        (int(cap.get(3)), int(cap.get(4))),
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        centroids = calculate_centroids(frame)

        for centroid in centroids:
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

        cv2.imshow(frame)
        out.write(frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


process_video(vid_path)


def plot_centroids_on_frames(video_path, centroids_df):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "opticalflow.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4)))
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_centroids = centroids_df[centroids_df["Object ID"] == frame_num]

        for index, row in frame_centroids.iterrows():
            centroid = (int(row["center_x"]), int(row["center_y"]))
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

        cv2.imshow(frame)
        out.write(frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


centroids_df = pd.read_csv(csv_file_path)
plot_centroids_on_frames(Opti_vid_path, centroids_df)

highlightes_path = "/content/drive/MyDrive/2024_001/2024_001_highlighted"


def plot_centroids_on_frames(video_path, centroids_df, output_folder):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_centroids = centroids_df[centroids_df["Object ID"] == frame_num]

        for index, row in frame_centroids.iterrows():
            centroid = (int(row["center_x"]), int(row["center_y"]))
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

        output_path = os.path.join(output_folder, f"frame_{frame_num}.jpg")
        cv2.imwrite(output_path, frame)

    cap.release()


centroids_df = pd.read_csv(csv_file_path)

plot_centroids_on_frames(vid_path, centroids_df, highlightes_path)

highlightes_path = "/content/drive/MyDrive/2024_001/2024_001_highlighted"
highlightvid_path = "/content/drive/MyDrive/2024_001/2024_001_highlighted/2024_001_highlightedVid/APGN4333 - Copy.MP4"
optihightlightvid_path = "/content/drive/MyDrive/2024_001/2024_001_highlighted/2024_001_highlightedVid/APGN4333.MP4"


def frames_to_video(input_folder, output_video_path, fps=5):
    frames = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])
    frame = cv2.imread(os.path.join(input_folder, frames[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_name in frames:
        frame_path = os.path.join(input_folder, frame_name)
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()


frames_to_video(highlightes_path, highlightvid_path)


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


cap = cv2.VideoCapture(highlightvid_path)

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    optihightlightvid_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height),
)

while True:
    suc, img = cap.read()
    if not suc:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = time.time()
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray
    end = time.time()
    fps = 1 / (end - start)
    print(f"{fps:.2f} FPS")

    flow_img = draw_flow(gray, flow)
    hsv_img = draw_hsv(flow)

    out.write(hsv_img)
    cv2.imshow(flow_img)
    cv2.imshow(hsv_img)
    key = cv2.waitKey(5)
    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
