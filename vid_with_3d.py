import cv2
import numpy as np
import open3d as o3d
import copy

# ---------------------------
# 1. Load Calibration, Video, and Template Data
# ---------------------------
# Load camera calibration data
camera_matrix = np.load("camera_intrinsics.npy")
dist_coeffs = np.load("distortion_coeffs.npy")

# Open the video
VIDEO_PATH = "./Charlie/Fifa17-vid3.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError("Could not open video file.")

# Load the template image (the planar object you wish to augment)
TEMPLATE_PATH = "./Charlie/Fifa17-cov1.jpg"
template = cv2.imread(TEMPLATE_PATH)
if template is None:
    raise ValueError("Template image not found.")
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
h_temp, w_temp = template_gray.shape

# ---------------------------
# 2. Initialize Feature Detection and Matching
# ---------------------------
# Create ORB detector and compute features on the template
orb = cv2.ORB_create()
kp_template, desc_template = orb.detectAndCompute(template_gray, None)

# Setup FLANN matcher (using the LSH algorithm for ORB)
index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# ---------------------------
# 3. Load and Prepare the 3D Model
# ---------------------------
mesh = o3d.io.read_triangle_mesh("Pikachu_tri.obj")
if mesh.is_empty():
    raise ValueError("Failed to load the 3D model.")
mesh.compute_vertex_normals()
mesh.scale(10.0, center=mesh.get_center())  # Adjust scale if needed
# Paint the mesh a uniform yellow color (RGB: [1,1,0])
mesh.paint_uniform_color([1.0, 1.0, 0])

# ---------------------------
# 4. Setup Video Writer
# ---------------------------
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_AR.mp4", fourcc, fps, (frame_width, frame_height))

# ---------------------------
# 5. Define the 3D Reference Object Points
# ---------------------------
# These 3D points should correspond to the corners of your template.
# Here we assume the template is a 100x100 square.
object_points = np.array([
    [0, 0, 0],
    [100, 0, 0],
    [100, 100, 0],
    [0, 100, 0]
], dtype=np.float32)

# ---------------------------
# 6. Main Processing Loop with Pose Persistence
# ---------------------------
last_extrinsic_o3d = None  # holds the last valid transformation
lost_frames = 0           # counts consecutive frames with no valid pose
max_lost_frames = 30      # maximum frames to hold the last pose

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ORB features in the current frame.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)
    
    valid_pose = False
    extrinsic_o3d = None

    if desc_frame is not None and len(kp_frame) >= 10:
        # Match template descriptors to the frame descriptors using FLANN (k=2)
        matches = flann.knnMatch(desc_template, desc_frame, k=2)
        good_matches = []
        pts_template = []
        pts_frame = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                pts_template.append(kp_template[m.queryIdx].pt)
                pts_frame.append(kp_frame[m.trainIdx].pt)
        pts_template = np.float32(pts_template)
        pts_frame = np.float32(pts_frame)
        
        if len(good_matches) > 10:
            # Compute the homography from the template to the current frame.
            H, mask = cv2.findHomography(pts_template, pts_frame, cv2.RANSAC, 5.0)
            if H is not None:
                # Define the four corners of the template image.
                template_corners = np.float32([[0, 0],
                                               [w_temp, 0],
                                               [w_temp, h_temp],
                                               [0, h_temp]]).reshape(-1, 1, 2)
                # Transform the template corners into the current frame.
                frame_corners = cv2.perspectiveTransform(template_corners, H)
                image_points = frame_corners.reshape(-1, 1, 2)

                # (Optional) Draw the detected corners on the frame.
                for corner in frame_corners:
                    pt = (int(corner[0][0]), int(corner[0][1]))
                    cv2.circle(frame, pt, 5, (0, 255, 0), -1)

                # Estimate the pose using solvePnP.
                success, rvecs, tvecs = cv2.solvePnP(object_points, image_points,
                                                      camera_matrix, dist_coeffs)
                if success:
                    print("rvecs:", rvecs.flatten(), "tvecs:", tvecs.flatten())
                    R, _ = cv2.Rodrigues(rvecs)
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = R
                    extrinsic[:3, 3] = tvecs.flatten()
                    # Convert from OpenCV coordinate system to Open3D coordinate system.
                    cv_to_o3d = np.array([
                        [1,  0,  0, 0],
                        [0, -1,  0, 0],
                        [0,  0, -1, 0],
                        [0,  0,  0, 1]
                    ])
                    extrinsic_o3d = cv_to_o3d @ extrinsic
                    valid_pose = True
                    lost_frames = 0
                    last_extrinsic_o3d = extrinsic_o3d

    if not valid_pose:
        lost_frames += 1
        if lost_frames < max_lost_frames and last_extrinsic_o3d is not None:
            extrinsic_o3d = last_extrinsic_o3d
        else:
            extrinsic_o3d = None

    # Render the model if we have a valid (or persistent) transformation.
    if extrinsic_o3d is not None:
        mesh_transformed = copy.deepcopy(mesh)
        mesh_transformed.transform(extrinsic_o3d)

        # ---------------------------
        # Offscreen Rendering with Open3D
        # ---------------------------
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=frame_width, height=frame_height)
        vis.add_geometry(mesh_transformed)
        vis.get_render_option().background_color = np.array([0, 0, 0])
        
        # Update the virtual camera.
        ctr = vis.get_view_control()
        pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            frame_width, frame_height,
            camera_matrix[0, 0], camera_matrix[1, 1],
            camera_matrix[0, 2], camera_matrix[1, 2]
        )
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        cam_params.intrinsic = pinhole_intrinsic
        cam_params.extrinsic = np.linalg.inv(extrinsic_o3d)
        try:
            ctr.convert_from_pinhole_camera_parameters(cam_params)
        except RuntimeError as e:
            print("Warning during camera parameter update:", e)
        
        for _ in range(5):
            vis.poll_events()
            vis.update_renderer()
        
        img_o3d = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        
        rendered_img = (255 * np.asarray(img_o3d)).astype(np.uint8)
        rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
        rendered_img = cv2.resize(rendered_img, (frame_width, frame_height))
        
        overlay = cv2.addWeighted(frame, 0.7, rendered_img, 0.3, 0)
        out.write(overlay)
        cv2.imshow("AR Overlay", overlay)
    else:
        out.write(frame)
        cv2.imshow("AR Overlay", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------
# 7. Cleanup
# ---------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved as output_AR.mp4")
