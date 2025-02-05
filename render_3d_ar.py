import os
import cv2
import numpy as np
from collections import deque

# -------------------------------
# 1. Disable OpenCL (to avoid errors)
# -------------------------------
cv2.ocl.setUseOpenCL(False)

# -------------------------------
# 2. Camera Calibration Parameters
# -------------------------------
# (Using your provided calibration values)
camera_matrix = np.array([[914.83565528, 0., 473.00332971],
                          [0., 916.97895899, 641.10946217],
                          [0., 0., 1.]], dtype=np.float32)
dist_coefs = np.array([2.28710423e-01, -7.17194756e-01,
                       2.12408818e-04, -2.47017341e-03,
                       3.13534277e-02], dtype=np.float32)

# -------------------------------
# 3. Reference Cover Image & Its Real-World Size
# -------------------------------
# Load the cover image (this image will be tracked in the video)
cover_img = cv2.imread("cover12.jpg")
if cover_img is None:
    raise ValueError("Error: Could not load reference cover image 'cover12.jpg'.")
(h_cover, w_cover, _) = cover_img.shape

# ASSUMED real‑world dimensions for the cover (in mm)
cover_width_mm = 200.0
cover_height_mm = 250.0

# Define the cover’s 3D object points (in mm) assuming the top‑left corner is (0,0,0)
cover_obj_points = np.array([
    [0, 0, 0],
    [cover_width_mm, 0, 0],
    [cover_width_mm, cover_height_mm, 0],
    [0, cover_height_mm, 0]
], dtype=np.float32)

# The corresponding cover image corners (in pixels)
cover_img_corners = np.array([
    [0, 0],
    [w_cover, 0],
    [w_cover, h_cover],
    [0, h_cover]
], dtype=np.float32)

# -------------------------------
# 4. Define the Virtual Cube (in cover coordinate system)
# -------------------------------
# For example, let the cube’s base be a 100x100 mm square starting at (50,50)
# and the cube extend 100 mm in height (z = -100).
cube_obj_points = np.array([
    [50, 50, 0],
    [150, 50, 0],
    [150, 150, 0],
    [50, 150, 0],
    [50, 50, -100],
    [150, 50, -100],
    [150, 150, -100],
    [50, 150, -100]
], dtype=np.float32)

# -------------------------------
# 5. Cube Drawing Function
# -------------------------------
def draw_cube(img, imgpts):
    """
    Draws a cube on the image:
      - The base is filled in green.
      - The vertical edges are drawn in blue.
      - The top face is outlined in red.
    """
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # Draw the base face filled in green.
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)
    # Draw the vertical edges in blue.
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
    # Draw the top face with a red outline.
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

# -------------------------------
# 6. Setup ORB Detector & FLANN Matcher for Cover Tracking
# -------------------------------
orb = cv2.ORB_create()
kp_ref, des_ref = orb.detectAndCompute(cover_img, None)

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# -------------------------------
# 7. Open the Video for Processing
# -------------------------------
# Updated video file name as specified.
video = cv2.VideoCapture("IMG_9144.MOV")
if not video.isOpened():
    raise ValueError("Error: Could not open video file 'IMG_9144.MOV'.")

# (Optional) Prepare a VideoWriter to save the output.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# For smoothing homographies over frames.
H_history = deque(maxlen=5)
last_valid_H = None
alpha = 0.5  # smoothing factor

# -------------------------------
# 8. Process Video Frames
# -------------------------------
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the current frame to grayscale.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors in the frame.
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
    if des_frame is None or len(des_frame) < 2:
        cv2.imshow("Overlay", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Find matches using FLANN with k=2.
    matches = flann.knnMatch(des_ref, des_frame, k=2)
    good_matches = []
    # Fix: Ensure each match has two elements before unpacking.
    for match in matches:
        if len(match) != 2:
            continue
        m, n = match
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    min_matches = 40  # Minimum number of good matches to proceed.
    if len(good_matches) > min_matches:
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography from cover to current frame.
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            # Optional: Smooth the homography.
            if last_valid_H is not None:
                H = alpha * last_valid_H + (1 - alpha) * H
            last_valid_H = H
        else:
            if last_valid_H is not None:
                H = last_valid_H
            else:
                cv2.imshow("Overlay", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

        # Warp the cover image onto the current frame.
        warped_cover = cv2.warpPerspective(cover_img, H, (frame.shape[1], frame.shape[0]))
        # Create a mask from the warped cover for blending.
        mask_cover = cv2.cvtColor(warped_cover, cv2.COLOR_BGR2GRAY)
        _, mask_cover = cv2.threshold(mask_cover, 1, 255, cv2.THRESH_BINARY)
        mask_cover = cv2.resize(mask_cover, (frame.shape[1], frame.shape[0]))
        mask_cover = mask_cover.astype(np.uint8)
        frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_cover))
        overlay_frame = cv2.add(frame_bg, warped_cover)

        # Transform the cover image corners to the current frame.
        cover_img_corners_reshaped = cover_img_corners.reshape(-1, 1, 2)
        video_cover_corners = cv2.perspectiveTransform(cover_img_corners_reshaped, H)
        video_cover_corners = video_cover_corners.reshape(-1, 2)

        # Estimate the cover’s pose (rvec, tvec) using solvePnP.
        ret_pnp, rvec, tvec = cv2.solvePnP(cover_obj_points, video_cover_corners,
                                            camera_matrix, dist_coefs)
        if ret_pnp:
            # Project the cube’s 3D points onto the current frame.
            imgpts, _ = cv2.projectPoints(cube_obj_points, rvec, tvec,
                                          camera_matrix, dist_coefs)
            overlay_frame = draw_cube(overlay_frame, imgpts)

        # Initialize VideoWriter if not already done.
        if out is None:
            out = cv2.VideoWriter('warped_cube_output.avi', fourcc, 20.0,
                                  (frame.shape[1], frame.shape[0]))
        out.write(overlay_frame)
        cv2.imshow("Overlay", overlay_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.imshow("Overlay", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# -------------------------------
# 9. Cleanup
# -------------------------------
video.release()
if out:
    out.release()
cv2.destroyAllWindows()
