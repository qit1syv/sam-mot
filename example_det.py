import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT


tracker = DeepOCSORT(
    model_weights=Path("osnet_x0_25_msmt17.pt"),  # which ReID model to use
    device="cuda:0",
    fp16=False,
)

import pyrealsense2 as rs

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()

# Configure streams
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)


while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    im = np.asanyarray(color_frame.get_data())

    # substitute by your object detector, output has to be N X (x, y, x, y, conf, cls)
    dets = np.array([[144, 212, 578, 480, 0.82, 0], [425, 281, 576, 472, 0.56, 65]])

    # Check if there are any detections
    if dets.size > 0:
        tracker.update(dets, im)  # --> M X (x, y, x, y, id, conf, cls, ind)
    # If no detections, make prediction ahead
    else:
        dets = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
        tracker.update(dets, im)  # --> M X (x, y, x, y, id, conf, cls, ind)
    tracker.plot_results(im, show_trajectories=True)

    # break on pressing q or space
    cv2.imshow("BoxMOT detection", im)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") or key == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
