import cv2
import numpy as np
from collections import deque

# ======= imports

# ======= constants
ref_img = cv2.imread("cover12.jpg")  # The original reference image
if ref_img is None:
    raise ValueError("Error: Could not load reference image.")

orb = cv2.ORB_create()
kp_ref, des_ref = orb.detectAndCompute(ref_img, None)

video = cv2.VideoCapture("IMG_9144.MOV")  # The recorded video for tracking
if not video.isOpened():
    raise ValueError("Error: Could not open video.")

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# Homography storage
H_history = deque(maxlen=5)
last_valid_H = None
alpha = 0.5  # Reduced smoothing factor for better responsiveness

# === template image keypoint and descriptors

# ===== video input, output and metadata

# ========== run on all frames
while True:
    ret, frame = video.read()
    if not ret:
        break

    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    if des_frame is None or len(des_frame) < 2:
        continue  # Skip frame if not enough descriptors are found

    matches = flann.knnMatch(des_ref, des_frame, k=2)
    good_matches = []
    for match in matches:
        if len(match) == 2:  # Ensure we have two matches before unpacking
            m, n = match
            if m.distance < 0.7 * n.distance:  # Stricter match filtering
                good_matches.append(m)

    min_matches = 40  # Keep dynamic frame processing
    if len(good_matches) > min_matches:
        # ======== find homography
        # also in SIFT notebook
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # Increased RANSAC threshold

        # Ensure H is valid before using it
        if H is not None:
            if last_valid_H is not None:
                H = alpha * last_valid_H + (1 - alpha) * H  # Blended smoothing
            last_valid_H = H  # Store the smoothed homography
        else:
            if last_valid_H is not None:
                H = last_valid_H  # Reuse the last valid homography
            else:
                continue  # If no valid homography, skip this frame

        h, w, _ = ref_img.shape
        # ++++++++ do warping of another image on template image
        # we saw this in SIFT notebook
        warped = cv2.warpPerspective(ref_img, H, (frame.shape[1], frame.shape[0]))

        # Ensure mask size and type are correct
        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Resize to match frame size
        mask = mask.astype(np.uint8)  # Convert to uint8 (CV_8U)

        # Blend using masked region
        frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        warped_fg = cv2.bitwise_and(warped, warped, mask=mask)
        overlay = cv2.add(frame_bg, warped_fg)

        if out is None:
            out = cv2.VideoWriter('warped_output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        out.write(overlay)

        # =========== plot and save frame
        cv2.imshow("Warped Image", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ======== end all
video.release()
if out:
    out.release()
cv2.destroyAllWindows()
cv2.imwrite("keypoints.jpg", cv2.drawKeypoints(ref_img, kp_ref, None, color=(0, 255, 0), flags=0))
