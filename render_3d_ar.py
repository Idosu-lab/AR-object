import os
import cv2
import numpy as np
from collections import deque

# -------------------------------
# 1. Disable OpenCL (to avoid errors)
# -------------------------------
cv2.ocl.setUseOpenCL(False)

# -------------------------------
# 2. Updated Camera Calibration Parameters
# -------------------------------
# RMS Error: 9.521700391689423
# Camera Matrix (K):
#  [[5.85493857e+03, 0.00000000e+00, 8.43369971e+02],
#   [0.00000000e+00, 4.13083151e+03, 6.01898161e+02],
#   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
# Distortion Coefficients:
#  [5.38606908e+00, -7.73244839e+01, 7.51650400e-01, 1.17381409e-01, 2.67996027e+02]

camera_matrix = np.array([[5.85493857e+03, 0.0, 8.43369971e+02],
                          [0.0, 4.13083151e+03, 6.01898161e+02],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
dist_coefs = np.array([5.38606908e+00, -7.73244839e+01,
                       7.51650400e-01, 1.17381409e-01,
                       2.67996027e+02], dtype=np.float32)

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
# and let the cube extend 100 mm in height (i.e. the top face is at z = -100).
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
# You may experiment with ORB parameters (e.g. increasing the number of features) if needed.
orb = cv2.ORB_create(nfeatures=1500)
kp_ref, des_ref = orb.detectAndCompute(cover_img, None)

# FLANN parameters (using LSH index for binary descriptors)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# -------------------------------
# 7. Open the Video for Processing
# -------------------------------
video = cv2.VideoCapture("IMG_9144.MOV")
if not video.isOpened():
    raise ValueError("Error: Could not open video file 'IMG_9144.MOV'.")

# (Optional) Prepare a VideoWriter to save the output.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# Variables for smoothing the homography and processing every other frame.
H_history = deque(maxlen=5)
last_valid_H = None
alpha = 0.5  # Smoothing factor for the homography
frame_count = 0

# -------------------------------
# 8. Process Video Frames with Temporal Filtering
# -------------------------------
while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1

    # For performance, update homography only on every second frame.
    update_H = (frame_count % 2 == 0)

    # If we are not updating, but we have a valid last homography, use it.
    if not update_H and last_valid_H is not None:
        H = last_valid_H
    else:
        # Convert the current frame to grayscale.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors in the frame.
        kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
        if des_frame is None or len(des_frame) < 2:
            # If not enough descriptors, show the frame and continue.
            cv2.imshow("Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Find matches using FLANN (with k=2) and use ratio test.
        matches = flann.knnMatch(des_ref, des_frame, k=2)
        good_matches = []
        for match in matches:
            if len(match) != 2:
                continue
            m, n = match
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Increase the threshold to 50 good matches for stability.
        min_matches = 50
        if len(good_matches) > min_matches:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography from cover to current frame.
            # Increase RANSAC reprojection threshold to 8.0.
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)
            if H is not None:
                # Smooth the homography using exponential smoothing.
                if last_valid_H is not None:
                    H = alpha * last_valid_H + (1 - alpha) * H
                last_valid_H = H
            else:
                # If no homography is found, try to use the last valid one.
                if last_valid_H is not None:
                    H = last_valid_H
                else:
                    cv2.imshow("Overlay", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
        else:
            # Not enough good matches: reuse last valid H if available.
            if last_valid_H is not None:
                H = last_valid_H
            else:
                cv2.imshow("Overlay", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

    # Use the (updated or reused) homography H to warp the cover image.
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

# -------------------------------
# 9. Cleanup
# -------------------------------
video.release()
if out:
    out.release()
cv2.destroyAllWindows()

# -------------------------------
# 10. Ensure the output video is saved in the same folder as this script
# -------------------------------
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_filename = 'cube_on_meditation.avi'
destination_path = os.path.join(script_dir, output_filename)

# Get the path where the VideoWriter saved the file (usually the current working directory)
current_output_path = os.path.join(os.getcwd(), output_filename)

# If the current working directory is not the script directory, move the file
if os.path.abspath(current_output_path) != os.path.abspath(destination_path):
    import shutil
    shutil.move(current_output_path, destination_path)

print("Output video has been saved to:", destination_path)
