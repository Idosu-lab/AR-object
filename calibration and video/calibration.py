import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Disable OpenCL to prevent OpenCV errors
cv2.ocl.setUseOpenCL(False)

# Chessboard dimensions (7x9 squares means 6x8 inner corners)
chessboard_size = (6, 8)
square_size = 25  # Square size in mm

# Prepare object points (3D points in real-world space)
obj_points = []  # 3D points
img_points = []  # 2D points

# List of calibration images
image_files = [
    "IMG_9093 Large.jpeg", "IMG_9095 Large.jpeg", "IMG_9097 Large.jpeg", "IMG_9098 Large.jpeg",
    "IMG_9099 Large.jpeg", "IMG_9100 Large.jpeg", "IMG_9101 Large.jpeg", "IMG_9102 Large.jpeg",
    "IMG_9103 Large.jpeg", "IMG_9104 Large.jpeg"
]

print("Found Images:", image_files)
last_gray = None  # Store last valid grayscale image

for fname in image_files:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Unable to read {fname}")
        continue

    print(f"Processing: {fname}, Shape: {img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve brightness using histogram equalization
    gray = cv2.equalizeHist(gray)

    # Detect chessboard using standard method first
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if not ret:
        # If normal detection fails, try the more robust method
        ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size, None)

    if ret:
        print(f"✅ Chessboard detected in {fname} with size {chessboard_size}")
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        obj_points.append(objp)
        img_points.append(corners)
        last_gray = gray

        # Refine corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points[-1] = corners2
    else:
        print(f"❌ Chessboard NOT detected in {fname}. Try adjusting brightness or rotating the image.")

cv2.destroyAllWindows()

# Ensure we have a valid grayscale image for calibration
if len(obj_points) >= 8:  # Ensure we have at least 8 successful detections
    h, w = last_gray.shape[:2]  # Get image width & height
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    # Save calibration results
    np.save("camera_intrinsics.npy", camera_matrix)
    np.save("distortion_coeffs.npy", dist_coefs)

    # Print results
    print("\nRMS Error:", rms)
    print("Camera Matrix (K):\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coefs.ravel())

    # Apply undistortion
    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            continue

        undistorted = cv2.undistort(img, camera_matrix, dist_coefs, None)
        output_file = f"undistorted_{fname}"
        cv2.imwrite(output_file, undistorted)
        print(f"Saved undistorted image: {output_file}")
else:
    print(
        "⚠️ Error: Not enough valid chessboard images found. Try adjusting brightness, contrast, or adding more images.")
