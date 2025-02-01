import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Chessboard dimensions (verify correct size)
chessboard_size = (8, 6)  # Adjusted to match 9x7 squares (8x6 inner corners)
square_size = 25  # Size of each square in mm (adjust accordingly)

# Prepare object points (3D points in real-world space)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# List of good calibration images
image_files = [
    "IMG_9093 Large.jpeg", "IMG_9095 Large.jpeg", "IMG_9097 Large.jpeg", "IMG_9098 Large.jpeg",
    "IMG_9099 Large.jpeg", "IMG_9100 Large.jpeg", "IMG_9101 Large.jpeg", "IMG_9102 Large.jpeg",
    "IMG_9103 Large.jpeg", "IMG_9104 Large.jpeg"
]

# Print found images to debug
print("Found Images:", image_files)

last_gray = None  # Variable to store the last valid grayscale image

for fname in image_files:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Unable to read {fname}")
        continue

    print(f"Processing: {fname}, Shape: {img.shape}")  # Debug: Check image dimensions
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to improve contrast
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Use the more robust chessboard detection method
    ret, corners = cv2.findChessboardCornersSB(enhanced, chessboard_size, None)

    if ret:
        print(f"✅ Chessboard detected in {fname}")
        objpoints.append(objp)
        imgpoints.append(corners)
        last_gray = gray  # Store the last valid grayscale image

        # Refine corner detection
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints[-1] = corners2

cv2.destroyAllWindows()

# Ensure we have a valid grayscale image for calibration
if last_gray is not None:
    # Camera calibration
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, last_gray.shape[::-1], None, None)

    # Save calibration results
    np.save("camera_intrinsics.npy", K)
    np.save("distortion_coeffs.npy", dist_coeffs)

    # Print results
    print("Camera Matrix (K):\n", K)
    print("Distortion Coefficients:\n", dist_coeffs)
else:
    print("⚠️ Error: No valid chessboard images found. Try adjusting chessboard size or improving contrast.")
