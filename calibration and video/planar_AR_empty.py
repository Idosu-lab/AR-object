import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Disable OpenCL to prevent OpenCV errors
cv2.ocl.setUseOpenCL(False)

# Camera calibration parameters from calibration step
camera_matrix = np.array([[914.83565528, 0., 473.00332971],
                          [0., 916.97895899, 641.10946217],
                          [0., 0., 1.]])
dist_coefs = np.array([2.28710423e-01, -7.17194756e-01, 2.12408818e-04, -2.47017341e-03, 3.13534277e-02])

# Chessboard dimensions (7x9 squares means 6x8 inner corners)
chessboard_size = (6, 8)
square_size = 25  # Square size in mm

# Prepare object points (3D points in real-world space for entire chessboard)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Define the 3D cube points relative to chessboard size
objectPoints = square_size * np.array([
    [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],  # Bottom square
    [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]  # Top square
], dtype=np.float32)


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # Draw pillars in blue
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)

    # Draw top layer in red
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


# Single image filename for debugging
img_name = "IMG_9093 Large.jpeg"

# Process the single image
imgBGR = cv2.imread(img_name)
if imgBGR is None:
    print(f"Error: Could not read image {img_name}")
else:
    print(f"Loaded image {img_name}, shape: {imgBGR.shape}")
    gray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)

    # Detect chessboard
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if not ret:
        ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size, None)

    if ret:
        print(f"✅ Chessboard detected in {img_name}")

        # Refine corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Ensure we have enough points for solvePnP
        if len(corners2) >= 4:
            # Solve PnP using the full chessboard 3D object points
            success, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coefs,
                                                 flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                imgpts, _ = cv2.projectPoints(objectPoints, rvecs, tvecs, camera_matrix, dist_coefs)
                imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
                drawn_image = draw(imgRGB, imgpts)

                plt.figure(figsize=(6, 6))
                plt.imshow(drawn_image)
                plt.title("Final Image with Cube Projection")
                plt.axis("off")
                plt.show()
            else:
                print(f"Error: solvePnP failed for {img_name}")
        else:
            print(
                f"❌ Not enough points detected for solvePnP ({len(corners2)} points found). Try adjusting brightness or angle.")
    else:
        print(f"❌ Chessboard NOT detected in {img_name}")

print("Processing complete. Check the displayed image for results.")
