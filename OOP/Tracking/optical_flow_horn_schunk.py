import cv2
import numpy as np

class HornSchunckOpticalFlow:
    def __init__(self, alpha=1.0, iterations=100, epsilon=0.001):
        self.alpha = alpha
        self.iterations = iterations
        self.epsilon = epsilon

    def calculate_optical_flow(self, prev_frame, next_frame):
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Initialize flow vectors
        flow_u = np.zeros_like(prev_gray, dtype=np.float32)
        flow_v = np.zeros_like(prev_gray, dtype=np.float32)

        # Compute derivatives of the frames
        Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)
        It = next_gray - prev_gray

        for _ in range(self.iterations):
            # Compute averages of flow vectors
            avg_u = cv2.boxFilter(flow_u, -1, (3, 3)) / 9.0
            avg_v = cv2.boxFilter(flow_v, -1, (3, 3)) / 9.0

            # Compute local averages of derivatives
            avg_Ix = cv2.boxFilter(Ix, -1, (3, 3)) / 9.0
            avg_Iy = cv2.boxFilter(Iy, -1, (3, 3)) / 9.0
            avg_It = cv2.boxFilter(It, -1, (3, 3)) / 9.0

            # Compute flow vectors incrementally
            numerator = avg_Ix * avg_u + avg_Iy * avg_v + avg_It
            denominator = self.alpha ** 2 + avg_Ix ** 2 + avg_Iy ** 2
            flow_u -= avg_u * (numerator / denominator)
            flow_v -= avg_v * (numerator / denominator)

            # Compute magnitude of flow vectors
            magnitude = np.sqrt(flow_u ** 2 + flow_v ** 2)

            # Terminate iterations if the change is small
            if np.mean(magnitude) < self.epsilon:
                break

        return flow_u, flow_v


# Example usage:
if __name__ == "__main__":
    input_video_path = r'C:\Users\USER\Desktop\TechnicalGP\Tracking\OG.MP4'
    output_video_path = r'C:\Users\USER\Desktop\TechnicalGP\Tracking\horn-shuckOUT.MP4'

    cap = cv2.VideoCapture(input_video_path)
    ret, prev_frame = cap.read()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    horn_schunck = HornSchunckOpticalFlow()

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        flow_u, flow_v = horn_schunck.calculate_optical_flow(prev_frame, next_frame)

        # Visualize the optical flow on the current frame
        flow_img = np.zeros_like(next_frame)
        flow_img[..., 0] = 255
        flow_img[..., 1] = 255
        flow_img[..., 2] = 255
        flow_img[..., 0] += flow_u.astype(np.uint8) * 15
        flow_img[..., 1] += flow_v.astype(np.uint8) * 15

        out.write(flow_img)  # Write the frame to the output video

        # Update the previous frame
        prev_frame = next_frame

    cap.release()
    out.release()
    cv2.destroyAllWindows()


