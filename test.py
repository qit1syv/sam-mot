import cv2
import numpy as np
from pathlib import Path

from fastsam import FastSAM, FastSAMPrompt
from boxmot import DeepOCSORT
import os
import torch
import cv2


tracker = DeepOCSORT(
    model_weights=Path("osnet_x0_25_msmt17.pt"),  # which ReID model to use
    device="cuda:0",
    fp16=True,
)

rand_colors = torch.rand((25600, 3))


def fast_show_mask_gpu(annotation, ids):

    msak_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    # areas = torch.sum(annotation, dim=(1, 2))
    # sorted_indices = torch.argsort(areas, descending=False)
    # annotation = annotation[sorted_indices]
    GLASBEY = rand_colors[ids]

    index = (annotation != 0).to(torch.long).argmax(dim=0)

    # color = torch.rand((msak_sum, 1, 1, 3)).to(annotation.device)
    color = GLASBEY[:msak_sum].reshape(msak_sum, 1, 1, 3).to(annotation.device)

    transparency = torch.ones((msak_sum, 1, 1, 1)).to(annotation.device) * 0.5
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual

    show = torch.zeros((height, weight, 4)).to(annotation.device)
    h_indices, w_indices = torch.meshgrid(
        torch.arange(height), torch.arange(weight), indexing="ij"
    )
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))

    show[h_indices, w_indices, :] = mask_image[indices]
    show_cpu = show.cpu().numpy()
    return show_cpu


# Create a FastSAM model
model = FastSAM("FastSAM-s.pt")  # FastSAM-s.pt or FastSAM-x.pt

# With RTX3080M 115W
# FastSAM-s.pt -> 6.5ms
# FastSAM-x.pt -> 18.5ms

# Open the video file

import pyrealsense2 as rs

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()

# Configure streams
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

first = True
cv_window_str = "FastSAM. PRESS Q TO QUIT"
cv2.namedWindow(cv_window_str, cv2.WINDOW_NORMAL)
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    color_frame = np.asanyarray(color_frame.get_data())

    results = model(
        color_frame,
        device="cpu",
        retina_masks=True,
        imgsz=(640, 480),
        conf=0.4,
        iou=0.9,
    )
    # prompt_process = FastSAMPrompt(color_frame, results)
    # prompt_mask = prompt_process.text_prompt("game controller")

    dets = results[0].boxes.xyxy.cpu().numpy()
    dets = np.concatenate(
        (
            dets,
            results[0].boxes.conf.cpu().numpy().reshape(-1, 1),
            np.zeros((dets.shape[0], 1)),
        ),
        axis=1,
    )

    # substitute by your object detector, input to tracker has to be N X (x, y, x, y, conf, cls)
    # dets = np.array([[144, 212, 578, 480, 0.82, 0], [425, 281, 576, 472, 0.56, 65]])

    tracks = tracker.update(
        dets, color_frame
    )  # --> M x (x, y, x, y, id, conf, cls, ind)

    # xyxys = tracks[:, 0:4].astype('int') # float64 to int
    ids = tracks[:, 4].astype("int")  # float64 to int
    # confs = tracks[:, 5]
    # clss = tracks[:, 6].astype('int') # float64 to int
    inds = tracks[:, 7].astype("int")  # float64 to int
    if first:
        dets = dets[:2]
        first = False

    # print(inds)

    # in case you have segmentations or poses alongside with your detections you can use
    # the ind variable in order to identify which track is associated to each seg or pose by:
    # masks = masks[inds]
    # keypoints = keypoints[inds]
    # such that you then can: zip(tracks, masks) or zip(tracks, keypoints)

    # break on pressing q or space
    annotated_frame = fast_show_mask_gpu(results[0].masks.data[inds], ids)
    alpha = 0.5
    annotated_frame = (
        annotated_frame[:, :, :3] * alpha + color_frame * (1 - alpha) / 255
    )
    cv2.imshow(cv_window_str, annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # tracker.plot_results(color_frame, show_trajectories=True)

    # cv2.imshow("BoxMOT detection", color_frame)
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord(" ") or key == ord("q"):
    #     break

cv2.destroyAllWindows()
