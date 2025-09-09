import cv2
import numpy as np
from pathlib import Path

import kornia as K
import kornia.feature as KF
import time

import torch

import sys
# print(sys.executable)

from Util import *

def convert_image_kornia(img, resize_to=None):
    if resize_to:
        img = cv2.resize(img, resize_to)
    img = K.image_to_tensor(img, keepdim=False).float() / 255.0  # [1,1,H,W]
    return img

# p1 = target; p2 = render
def convert_to_database(points1_sorted, points2_sorted, target_frame, prior_position_buffer, prediction_width = 1920, cinf = None):
    running_idx = 0

    point_database = {}

    factor = target_frame.shape[1] / prediction_width

    for pt1, pt2 in zip(points1_sorted, points2_sorted):
        current_kp = pt2  * factor
        x = round(current_kp[0])
        y = round(current_kp[1])

        target_kp = pt1  * factor
        tx = round(target_kp[0])
        ty = round(target_kp[1])

        location = prior_position_buffer[y, x, :]
        location = np.array([location[0], location[1], location[2], 1.0])
        # print(f'{current_kp} -> {location}')
        name = f"Reference_{running_idx}"
        running_idx += 1

        point_database[name] = [
            np.array([tx, ty]),
            location
        ]

    return point_database

def RunLightGlueMatcher(target_frame, current_render, prior_position_buffer):
    target = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
    prior = cv2.cvtColor(current_render, cv2.COLOR_BGR2RGB)
    # DeDoDe needs size to be divisible by 14
    img1 = convert_image_kornia(target, resize_to=(966, 546))
    img2 = convert_image_kornia(prior, resize_to=(966, 546)) 

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1 = img1.to(device)
    img2 = img2.to(device)

    detector = KF.DeDoDe.from_pretrained(detector_weights="L-C4-v2", descriptor_weights="B-upright").to(device).eval()
    matcher = KF.LightGlue('dedodeb').to(device).eval()

    with torch.inference_mode():
        # print("running extraction 1")
        kp1, scores2, desc1 = detector(img1)
        # print("running extraction 2")
        kp2, scores2, desc2 = detector(img2)

    input_dict = {
        "image0": {
            "keypoints": kp1,
            "descriptors": desc1,
            "image": img1
        },
        "image1": {
            "keypoints": kp2,
            "descriptors": desc2,
            "image": img2
        }}

    # print(f"Expected dimension: {matcher.conf.input_dim}")
    with torch.inference_mode():
        matcher_out = matcher(input_dict)

    def get_matched_dense_points(matcher_out, kp1, kp2):
        matches = matcher_out["matches"][0].cpu().numpy()
        confidence = matcher_out["scores"][0].cpu().numpy()
        kp1_np = kp1[0].cpu().numpy()
        kp2_np = kp2[0].cpu().numpy()

        sorted_indices = np.argsort(-confidence)
        points1_sorted = matches[sorted_indices][:20]

        ret_match_1 = kp1_np[points1_sorted[:, 0]]
        ret_match_2 = kp2_np[points1_sorted[:, 1]]

        return ret_match_1, ret_match_2

    mkpts1, mkpts2 = get_matched_dense_points(matcher_out, kp1, kp2)
    # print(f"Found {len(mkpts1)} matches.")

    return convert_to_database(mkpts1, mkpts2, target_frame, prior_position_buffer, 966)