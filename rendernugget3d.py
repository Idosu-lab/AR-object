import os
import cv2
import numpy as np
from collections import deque

# -------------------------------
# 1. Disable OpenCL (to avoid errors)
# -------------------------------
cv2.ocl.setUseOpenCL(False)

# -------------------------------
# 2. Camera Calibration Parameters
# -------------------------------
camera_matrix = np.array([[914.83565528, 0., 473.00332971],
                          [0., 916.97895899, 641.10946217],
                          [0., 0., 1.]], dtype=np.float32)
dist_coefs = np.array([2.28710423e-01, -7.17194756e-01,
                       2.12408818e-04, -2.47017341e-03,
                       3.13534277e-02], dtype=np.float32)

# -------------------------------
# 3. Reference Cover Image & Its Real-World Size
# -------------------------------
cover_img = cv2.imread("cover12.jpg")
if cover_img is None:
    raise ValueError("Error: Could not load reference cover image 'cover12.jpg'.")
(h_cover, w_cover, _) = cover_img.shape

# Assumed real‑world dimensions for the cover (in mm)
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
# 4. Load the OBJ Model and Its Texture
# -------------------------------
# A simple OBJ loader that extracts vertices, texture coordinates, and faces.
class OBJ:
    def __init__(self, filename, swapyz=False):
        self.vertices = []  # list of [x, y, z]
        self.normals = []  # list of [nx, ny, nz] (not used here)
        self.texcoords = []  # list of [u, v] (assumed normalized in [0,1])
        self.faces = []  # list of tuples: (vertex_indices, normal_indices, texcoord_indices, material)

        material = None
        with open(filename, "r") as f:
            for line in f:
                if line.startswith('#'):
                    continue
                values = line.split()
                if not values:
                    continue
                if values[0] == 'v':
                    v = list(map(float, values[1:4]))
                    if swapyz:
                        v = [v[0], v[2], v[1]]
                    self.vertices.append(v)
                elif values[0] == 'vn':
                    vn = list(map(float, values[1:4]))
                    if swapyz:
                        vn = [vn[0], vn[2], vn[1]]
                    self.normals.append(vn)
                elif values[0] == 'vt':
                    vt = list(map(float, values[1:3]))
                    self.texcoords.append(vt)
                elif values[0] in ('usemtl', 'usemat'):
                    material = values[1]
                elif values[0] == 'f':
                    face_v = []
                    face_vt = []
                    face_vn = []
                    for v in values[1:]:
                        w = v.split('/')
                        vertex_index = int(w[0]) - 1
                        face_v.append(vertex_index)
                        if len(w) >= 2 and w[1]:
                            tex_index = int(w[1]) - 1
                            face_vt.append(tex_index)
                        else:
                            face_vt.append(-1)
                        if len(w) >= 3 and w[2]:
                            norm_index = int(w[2]) - 1
                            face_vn.append(norm_index)
                        else:
                            face_vn.append(-1)
                    self.faces.append((face_v, face_vn, face_vt, material))


# Load the model and its texture.
obj_filename = "whole chicken nugget.obj"
model = OBJ(obj_filename, swapyz=True)
texture_img = cv2.imread("whole chicken nugget_1.jpg")
if texture_img is None:
    raise ValueError("Error: Could not load texture image 'whole chicken nugget_1.jpg'.")


# -------------------------------
# 5. Define a Model-to-Cover Transformation
# -------------------------------
def compute_model_transform(vertices, desired_base_width=100, desired_offset=(50, 50)):
    pts = np.array(vertices)
    min_coords = pts.min(axis=0)
    max_coords = pts.max(axis=0)
    size = max_coords - min_coords
    scale = desired_base_width / size[0]
    offset = np.array([desired_offset[0], desired_offset[1], 0], dtype=np.float32)
    T = np.array([
        [scale, 0, 0, offset[0] - scale * min_coords[0]],
        [0, scale, 0, offset[1] - scale * min_coords[1]],
        [0, 0, scale, -scale * min_coords[2]],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    return T


model_transform = compute_model_transform(model.vertices, desired_base_width=100, desired_offset=(50, 50))


def transform_vertices(vertices, T):
    pts = np.array(vertices)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    pts_trans = (T @ pts_h.T).T
    return pts_trans[:, :3]


# -------------------------------
# 6. Utility: Render a Textured Triangle
# -------------------------------
def draw_textured_triangle(frame, tri_dst, tri_src, texture):
    # Compute bounding rectangle for destination triangle.
    x, y, w, h = cv2.boundingRect(np.int32(tri_dst))
    if w == 0 or h == 0:
        return
    tri_dst_rect = tri_dst - np.array([[x, y]], dtype=np.float32)
    M = cv2.getAffineTransform(np.float32(tri_src), tri_dst_rect)
    warped_patch = cv2.warpAffine(texture, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(tri_dst_rect), 255)
    roi = frame[y:y + h, x:x + w]
    warped_patch_masked = cv2.bitwise_and(warped_patch, warped_patch, mask=mask)
    roi_masked = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    frame[y:y + h, x:x + w] = cv2.add(roi_masked, warped_patch_masked)


# -------------------------------
# 7. Render the OBJ Model with Texture
# -------------------------------
def render_obj(frame, model, rvec, tvec, camera_matrix, dist_coefs, model_transform, texture_img):
    # Transform the model's vertices to cover-space.
    transformed_vertices = transform_vertices(model.vertices, model_transform)
    projected, _ = cv2.projectPoints(transformed_vertices, rvec, tvec, camera_matrix, dist_coefs)
    projected = projected.reshape(-1, 2)

    # Compute camera-space coordinates for depth sorting.
    R, _ = cv2.Rodrigues(rvec)
    cam_pts = (R @ transformed_vertices.T + tvec).T

    face_depths = []
    for face in model.faces:
        face_idx, _, _, _ = face
        zs = [cam_pts[i][2] for i in face_idx]
        avg_z = np.mean(zs)
        face_depths.append(avg_z)

    # Sort faces by average depth (farthest first)
    sorted_faces = [face for _, face in sorted(zip(face_depths, model.faces),
                                               key=lambda pair: pair[0],
                                               reverse=True)]

    tex_h, tex_w = texture_img.shape[:2]

    # Render each face.
    for face in sorted_faces:
        face_idx, _, tex_idx, material = face
        if any(ti < 0 for ti in tex_idx):
            continue
        pts_img = [projected[i] for i in face_idx]
        pts_tex = []
        for ti in tex_idx:
            u, v = model.texcoords[ti]
            pts_tex.append([u * tex_w, (1 - v) * tex_h])
        pts_img = np.array(pts_img, dtype=np.float32)
        pts_tex = np.array(pts_tex, dtype=np.float32)
        if len(pts_img) < 3:
            continue
        # Triangulate using fan triangulation.
        for i in range(1, len(pts_img) - 1):
            tri_img = np.array([pts_img[0], pts_img[i], pts_img[i + 1]], dtype=np.float32)
            tri_tex = np.array([pts_tex[0], pts_tex[i], pts_tex[i + 1]], dtype=np.float32)
            draw_textured_triangle(frame, tri_img, tri_tex, texture_img)
    return frame


# -------------------------------
# 8. Setup ORB Detector & FLANN Matcher for Cover Tracking
# -------------------------------
# (You can adjust nfeatures to trade off speed vs. robustness.)
orb = cv2.ORB_create(nfeatures=1000)
kp_ref, des_ref = orb.detectAndCompute(cover_img, None)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# -------------------------------
# 9. Open the Video for Processing
# -------------------------------
video = cv2.VideoCapture("IMG_9144.MOV")
if not video.isOpened():
    raise ValueError("Error: Could not open video file 'IMG_9144.MOV'.")

# Let’s try writing output at 30 fps for smoother playback.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# Variables for pose smoothing.
last_rvec = None
last_tvec = None
alpha_pose = 0.7  # (0 = no smoothing, 1 = full smoothing)

# Downscale factor for feature matching (smaller is faster, but too small may lose detail)
scale_factor = 0.3

# We'll process every frame for the first 'initial_frame_threshold' frames
# to “bootstrap” tracking, then switch to a processing interval.
initial_frame_threshold = 30
processing_interval = 2  # heavy processing every 2 frames after initialization

frame_count = 0
last_valid_H = None

# -------------------------------
# 10. Process Video Frames
# -------------------------------
while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1

    # Decide on update frequency and matching threshold based on frame number.
    if frame_count < initial_frame_threshold:
        current_min_matches = 20  # lower threshold during initialization
        current_interval = 1  # process every frame
    else:
        current_min_matches = 50
        current_interval = processing_interval

    # Only update heavy computations on frames according to the interval.
    if frame_count % current_interval == 0:
        # Downscale frame for faster feature matching.
        frame_small = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        kp_frame_small, des_frame_small = orb.detectAndCompute(gray_small, None)
        if des_frame_small is None or len(des_frame_small) < 2:
            cv2.imshow("Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        matches = flann.knnMatch(des_ref, des_frame_small, k=2)
        good_matches = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        if len(good_matches) > current_min_matches:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame_small[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = dst_pts / scale_factor  # Scale back to original size.
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)
            if H is None:
                cv2.imshow("Overlay", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
        else:
            if last_valid_H is not None:
                H = last_valid_H
            else:
                cv2.imshow("Overlay", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
        last_valid_H = H

        # Warp the cover image using the computed homography.
        warped_cover = cv2.warpPerspective(cover_img, H, (frame.shape[1], frame.shape[0]))
        mask_cover = cv2.cvtColor(warped_cover, cv2.COLOR_BGR2GRAY)
        _, mask_cover = cv2.threshold(mask_cover, 1, 255, cv2.THRESH_BINARY)
        mask_cover = cv2.resize(mask_cover, (frame.shape[1], frame.shape[0]))
        mask_cover = mask_cover.astype(np.uint8)
        frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_cover))
        overlay_frame = cv2.add(frame_bg, warped_cover)

        # Compute the cover image corners and estimate pose.
        cover_img_corners_reshaped = cover_img_corners.reshape(-1, 1, 2)
        video_cover_corners = cv2.perspectiveTransform(cover_img_corners_reshaped, H)
        video_cover_corners = video_cover_corners.reshape(-1, 2)
        ret_pnp, rvec, tvec = cv2.solvePnP(cover_obj_points, video_cover_corners,
                                           camera_matrix, dist_coefs)
        if ret_pnp:
            # Smooth the pose parameters if we have previous values.
            if last_rvec is not None and last_tvec is not None:
                rvec = alpha_pose * last_rvec + (1 - alpha_pose) * rvec
                tvec = alpha_pose * last_tvec + (1 - alpha_pose) * tvec
            last_rvec, last_tvec = rvec, tvec
        else:
            if last_rvec is not None and last_tvec is not None:
                rvec = last_rvec
                tvec = last_tvec
    else:
        # For frames when heavy processing is skipped, reuse last computed values.
        if last_valid_H is None or last_rvec is None or last_tvec is None:
            cv2.imshow("Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        H = last_valid_H
        rvec = last_rvec
        tvec = last_tvec
        warped_cover = cv2.warpPerspective(cover_img, H, (frame.shape[1], frame.shape[0]))
        mask_cover = cv2.cvtColor(warped_cover, cv2.COLOR_BGR2GRAY)
        _, mask_cover = cv2.threshold(mask_cover, 1, 255, cv2.THRESH_BINARY)
        mask_cover = cv2.resize(mask_cover, (frame.shape[1], frame.shape[0]))
        mask_cover = mask_cover.astype(np.uint8)
        frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_cover))
        overlay_frame = cv2.add(frame_bg, warped_cover)

    # Render the OBJ model using the current (or last computed) pose.
    if rvec is not None and tvec is not None:
        overlay_frame = render_obj(overlay_frame, model, rvec, tvec,
                                   camera_matrix, dist_coefs, model_transform, texture_img)

    # Initialize VideoWriter if not already done (set to 30 fps for smoother playback).
    if out is None:
        out = cv2.VideoWriter('warped_nugget_output.avi', fourcc, 30.0,
                              (frame.shape[1], frame.shape[0]))
    out.write(overlay_frame)
    cv2.imshow("Overlay", overlay_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
if out:
    out.release()
cv2.destroyAllWindows()
