import cv2 
import numpy as np

class OpticalFlowAnalyzerLucasKanade:
    def __init__(self, vid_path, output_path):
        self.cap = cv2.VideoCapture(vid_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.frame_width, self.frame_height))

    def process_video(self):
        suc, prev = self.cap.read()
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        while True:
            suc, img = self.cap.read()
            if not suc:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # calculate optical flow using Lucas-Kanade method
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # draw the flow field onto the current frame
            flow_img = self.draw_flow(gray, flow)

            # write the frame to the output video
            self.out.write(flow_img)

            cv2.imshow("Dense Optical Flow", flow_img)

            key = cv2.waitKey(25)  # Increase the delay to 25 milliseconds
            if key == 27:  # press 'ESC' to exit
                break

            # Update the previous frame
            prevgray = gray.copy()

        # Release video capture and writer objects
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def draw_flow(self, img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T

        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

        return img_bgr


