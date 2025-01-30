import cv2
import numpy as np

# ======= imports

# ======= constants
ref_img = cv2.imread("cover1.jpg")  # The original reference image
if ref_img is None:
    raise ValueError("Error: Could not load reference image.")

orb = cv2.ORB_create()
kp_ref, des_ref = orb.detectAndCompute(ref_img, None)

video = cv2.VideoCapture("vid1.MOV")  # The recorded video for tracking
if not video.isOpened():
    raise ValueError("Error: Could not open video.")

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

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
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) > 10:
        # ======== find homography
        # also in SIFT notebook
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w, _ = ref_img.shape

        # ++++++++ do warping of another image on template image
        # we saw this in SIFT notebook
        warped = cv2.warpPerspective(ref_img, H, (frame.shape[1], frame.shape[0]))
        overlay = cv2.addWeighted(frame, 0.5, warped, 0.5, 0)

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
