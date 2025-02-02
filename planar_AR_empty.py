import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Disable OpenCL to prevent OpenCV errors
cv2.ocl.setUseOpenCL(False)

# Camera calibration parameters
camera_matrix = np.array([[914.83565528, 0., 473.00332971],
                          [0., 916.97895899, 641.10946217],
                          [0., 0., 1.]])
dist_coefs = np.array([2.28710423e-01, -7.17194756e-01, 2.12408818e-04, -2.47017341e-03, 3.13534277e-02])

# Chessboard properties
chessboard_size = (6, 8)
square_size = 25  # Square size in mm

# Prepare object points for the chessboard
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Adjusted cube position - enlarge it on the board
objectPoints = square_size * np.array([
    [1, 1, 0], [1, 4, 0], [4, 4, 0], [4, 1, 0],  # Bottom square (enlarged)
    [1, 1, -3], [1, 4, -3], [4, 4, -3], [4, 1, -3]  # Top square (enlarged)
], dtype=np.float32)


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw base in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # Draw vertical edges in blue
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)

    # Draw top face in red
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


# Process specific images in the same directory
image_directory = os.path.dirname(os.path.abspath(__file__))  # Get script directory
output_directory = os.path.join(image_directory, "output")

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

image_files = [
    "IMG_9091 Large.jpeg", "IMG_9092 Large.jpeg", "IMG_9093 Large.jpeg", "IMG_9094 Large.jpeg",
    "IMG_9095 Large.jpeg", "IMG_9096 Large.jpeg", "IMG_9097 Large.jpeg", "IMG_9098 Large.jpeg",
    "IMG_9099 Large.jpeg", "IMG_9100 Large.jpeg", "IMG_9101 Large.jpeg", "IMG_9102 Large.jpeg"
]

for img_name in image_files:
    img_path = os.path.join(image_directory, img_name)
    if not os.path.exists(img_path):
        print(f"Error: Image {img_name} not found.")
        continue

    imgBGR = cv2.imread(img_path)
    if imgBGR is None:
        print(f"Error: Could not read image {img_name}")
        continue

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

        if len(corners2) >= 4:
            # Solve PnP
            success, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coefs,
                                                 flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                imgpts, _ = cv2.projectPoints(objectPoints, rvecs, tvecs, camera_matrix, dist_coefs)
                imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
                drawn_image = draw(imgRGB, imgpts)

                # Save the output image
                output_path = os.path.join(output_directory, f"output_{img_name}")
                cv2.imwrite(output_path, cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR))

                plt.figure(figsize=(6, 6))
                plt.imshow(drawn_image)
                plt.title(f"Cube Projection on {img_name}")
                plt.axis("off")
                plt.show()
            else:
                print(f"Error: solvePnP failed for {img_name}")
        else:
            print(f"❌ Not enough points detected for solvePnP in {img_name} ({len(corners2)} found).")
    else:
        print(f"❌ Chessboard NOT detected in {img_name}")

print("Processing complete. Check the output images in the directory.")
