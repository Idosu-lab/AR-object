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

# List of calibration images excluding "Large" versions
image_files = [
    "IMG_9041.jpeg", "IMG_9042.jpeg", "IMG_9043.jpeg", "IMG_9044.jpeg", "IMG_9045.jpeg", "IMG_9046.jpeg",
    "IMG_9047.jpeg", "IMG_9048.jpeg", "IMG_9049.jpeg", "IMG_9050.jpeg", "IMG_9051.jpeg", "IMG_9052.jpeg",
    "IMG_9053.jpeg", "IMG_9054.jpeg", "IMG_9055.jpeg", "IMG_9056.jpeg", "IMG_9057.jpeg", "IMG_9058.jpeg",
    "IMG_9059.jpeg", "IMG_9060.jpeg", "IMG_9061.jpeg"
]

# Print found images to debug
print("Found Images:", image_files)

last_gray = None  # Variable to store the last valid grayscale image

for fname in image_files[:5]:  # Test with first 5 images first
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Unable to read {fname}")
        continue

    print(f"Processing: {fname}, Shape: {img.shape}")  # Debug: Check image dimensions
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to improve contrast
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Display grayscale and thresholded images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title(f"Grayscale: {fname}")
    axes[1].imshow(enhanced, cmap='gray')
    axes[1].set_title(f"Thresholded: {fname}")
    plt.show()

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

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Corners: {fname}")
        plt.show()
    else:
        print(f"❌ Chessboard NOT detected in {fname}")

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
