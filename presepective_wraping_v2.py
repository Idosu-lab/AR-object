import cv2
import numpy as np

# ======== Configuration ==========
REF_IMAGE_PATH = "cover1.jpg"
VIDEO_PATH = "vid1.MOV"
OUTPUT_VIDEO_PATH = "stable_output.avi"
SMOOTHING_FACTOR = 0.3  # Between 0 (no smoothing) and 1 (full previous frame)
MIN_MATCHES = 60  # Increase for more stability, decrease for more responsiveness
MATCH_RATIO = 0.7  # Lower = stricter matches (0.6-0.75 recommended)

# ======== Initialize Systems ==========
# Load reference image
ref_img = cv2.imread(REF_IMAGE_PATH)
if ref_img is None:
    raise ValueError(f"Could not load reference image at {REF_IMAGE_PATH}")

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=2000)
kp_ref, des_ref = orb.detectAndCompute(ref_img, None)

# Initialize FLANN matcher
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Initialize video capture
video = cv2.VideoCapture(VIDEO_PATH)
if not video.isOpened():
    raise ValueError(f"Could not open video at {VIDEO_PATH}")

# Get video properties for output
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# ======== Tracking State ==========
H_prev = None  # For homography smoothing

# ======== Main Processing Loop ==========
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Preprocess frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    # Skip frame if not enough features
    if des_frame is None or len(des_frame) < 10:
        continue

    # Feature matching
    matches = flann.knnMatch(des_ref, des_frame, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < MATCH_RATIO * n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCHES:
        # Convert keypoints to arrays
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

        if H is not None:
            # Apply temporal smoothing
            if H_prev is not None:
                H = SMOOTHING_FACTOR * H_prev + (1 - SMOOTHING_FACTOR) * H
            H_prev = H

            # Warp reference image to frame perspective
            warped = cv2.warpPerspective(ref_img, H, (frame_width, frame_height))

            # Create blended overlay
            overlay = cv2.addWeighted(frame, 0.5, warped, 0.5, 0)

            # Write and display
            out.write(overlay)
            cv2.imshow("Stabilized AR Overlay", overlay)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======== Cleanup ==========
video.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete. Output saved to", OUTPUT_VIDEO_PATH)