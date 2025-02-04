import os
# Force Open3D to use the OpenGL backend (instead of Filament/EGL)
os.environ["OPEN3D_RENDERING_BACKEND"] = "OpenGL"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import argparse
import copy

# -------------------------------
# Mesh Loading & Preparation
# -------------------------------
def load_mesh(mesh_path="Pikachu_tri.obj", scale_factor=0.0005, extra_translation=0.0):
    """
    Loads the mesh from an OBJ file, computes vertex normals, scales it,
    translates it so that its base is at z=0, applies an additional translation,
    and rotates it so that it faces the correct direction.
    
    Parameters:
      mesh_path: path to the OBJ file.
      scale_factor: scaling factor for the model.
      extra_translation: additional translation along the z-axis.
    
    Returns:
      The prepared Open3D TriangleMesh.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertices():
        print(f"Error: Could not load the mesh from '{mesh_path}'.")
        exit(1)
    mesh.compute_vertex_normals()
    
    # Print the original bounding box for debugging.
    print("Original bounding box:")
    print("Min bound:", mesh.get_min_bound())
    print("Max bound:", mesh.get_max_bound())
    
    # Scale the mesh.
    mesh.scale(scale_factor, center=mesh.get_center())
    print("After scaling bounding box:")
    print("Min bound:", mesh.get_min_bound())
    print("Max bound:", mesh.get_max_bound())
    
    # Translate so that the base is at z = 0.
    min_bound = mesh.get_min_bound()
    mesh.translate([0, 0, -min_bound[2]])
    
    # Apply additional translation if needed.
    if extra_translation != 0.0:
        mesh.translate([0, 0, extra_translation])
        print(f"Applied extra translation along z: {extra_translation}")
    
    # Rotate the mesh 180° about the Y-axis.
    R_model = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
    mesh.rotate(R_model, center=mesh.get_center())

    # Assign a yellow color to all vertices
    yellow = np.array([1.0, 0.9, 0.0])  # RGB values for yellow
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(yellow, (len(mesh.vertices), 1)))

    # Print the final bounding box for debugging.
    print("Final bounding box after all transformations:")
    print("Min bound:", mesh.get_min_bound())
    print("Max bound:", mesh.get_max_bound())
    
    return mesh


# -------------------------------
# Interactive Mesh Viewer
# -------------------------------
def show_mesh_interactively(mesh):
    """
    Displays the mesh in an interactive window.
    """
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="3D Object Viewer - Pikachu",
        width=800,
        height=600,
        left=50,
        top=50,
        mesh_show_back_face=True
    )

# -------------------------------
# Rendering Function with Fallback
# -------------------------------
def render_mesh(mesh, intrinsic, extrinsic, width, height, lookat=None, bg_img=None):
    """
    Attempts to render the mesh using Open3D's OffscreenRenderer.
    If that fails, falls back to an interactive Visualizer with a magenta background.
    If bg_img is provided, uses chroma-keying to composite the rendered model over bg_img.
    """
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(int(width), int(height))
        renderer.scene.set_background([0, 0, 0, 0])
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        renderer.scene.add_geometry("model", mesh, material)
        renderer.setup_camera(intrinsic, extrinsic)
        rendered_o3d = renderer.render_to_image()
        renderer.scene.remove_geometry("model")
        return np.asarray(rendered_o3d)
    except RuntimeError as e:
        print("OffscreenRenderer creation failed:")
        print(e)
        print("Falling back to interactive Visualizer rendering with chroma-key...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=int(width), height=int(height))
        ro = vis.get_render_option()
        ro.background_color = np.array([1.0, 0.0, 1.0])  # Magenta.
        vis.add_geometry(mesh)
        ctr = vis.get_view_control()
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        cam_pos = -R.T @ t
        if lookat is None:
            lookat = np.array([0, 0, 0])
        front = lookat - cam_pos
        front_norm = np.linalg.norm(front)
        if front_norm != 0:
            front = front / front_norm
        else:
            front = np.array([0, 0, 1])
        up = R.T @ np.array([0, 1, 0])
        try:
            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic = intrinsic
            param.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters(param)
        except Exception as ex:
            ctr.set_lookat(lookat)
            ctr.set_front(front)
            ctr.set_up(up)
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        rendered_img = (255 * np.asarray(img)).astype(np.uint8)
        if bg_img is not None and (rendered_img.shape[0] != bg_img.shape[0] or rendered_img.shape[1] != bg_img.shape[1]):
            rendered_img = cv2.resize(rendered_img, (bg_img.shape[1], bg_img.shape[0]))
        if bg_img is not None:
            magenta = np.array([255, 0, 255], dtype=np.uint8)
            diff = np.abs(rendered_img.astype(np.int16) - magenta.astype(np.int16))
            mask = np.all(diff < 30, axis=2)
            alpha = np.where(mask[..., None], 0, 1).astype(np.float32)
            composite = alpha * rendered_img.astype(np.float32) + (1 - alpha) * bg_img.astype(np.float32)
            composite = np.clip(composite, 0, 255).astype(np.uint8)
            return composite
        else:
            return rendered_img

# -------------------------------
# AR Processing Function
# -------------------------------
def process_images(mesh):
    """
    For each image, detects the chessboard, computes the camera pose,
    renders the mesh from that pose, and composites the rendered image
    over the original chessboard image.
    """
    camera_matrix = np.array([[914.83565528, 0., 473.00332971],
                              [0., 916.97895899, 641.10946217],
                              [0., 0., 1.]])
    dist_coefs = np.array([0.228710423, -0.717194756, 0.000212408818,
                           -0.00247017341, 0.0313534277])
    
    chessboard_size = (6, 8)  # (rows, columns) of inner corners.
    square_size = 25  # in mm.
    
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    
    # Compute the chessboard center using (rows-1) and (cols-1)
    chessboard_center = np.array([ (chessboard_size[0]-1) * square_size / 2,
                                   (chessboard_size[1]-1) * square_size / 2,
                                   0 ])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [
        "IMG_9091 Large.jpeg", "IMG_9092 Large.jpeg", "IMG_9093 Large.jpeg",
        "IMG_9094 Large.jpeg", "IMG_9095 Large.jpeg", "IMG_9096 Large.jpeg",
        "IMG_9097 Large.jpeg", "IMG_9098 Large.jpeg", "IMG_9099 Large.jpeg",
        "IMG_9100 Large.jpeg", "IMG_9101 Large.jpeg", "IMG_9102 Large.jpeg"
    ]
    
    for img_name in image_files:
        img_path = os.path.join(script_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image {img_name} not found.")
            continue
        
        imgBGR = cv2.imread(img_path)
        if imgBGR is None:
            print(f"Could not read image {img_name}.")
            continue
        
        gray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if not ret:
            ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size, None)
        if not ret:
            print(f"Chessboard not detected in {img_name}.")
            continue
        else:
            print(f"Chessboard detected in {img_name}.")
        
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        
        if len(corners2) < 4:
            print(f"Not enough points for solvePnP in {img_name} (found {len(corners2)}).")
            continue
        
        success, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coefs,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            print(f"solvePnP failed for {img_name}.")
            continue
        
        R, _ = cv2.Rodrigues(rvecs)
        T_cv = np.eye(4)
        T_cv[:3, :3] = R
        T_cv[:3, 3] = tvecs.flatten()
        
        R_flip = np.array([[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])
        extrinsic_o3d = np.eye(4)
        extrinsic_o3d[:3, :3] = R_flip @ T_cv[:3, :3]
        extrinsic_o3d[:3, 3] = R_flip @ T_cv[:3, 3]
        
        height_img, width_img, _ = imgBGR.shape
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width_img, height_img, fx, fy, cx, cy)
        
        bg_img = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        
        rendered = render_mesh(copy.deepcopy(mesh), intrinsic_o3d, extrinsic_o3d,
                               width_img, height_img, lookat=chessboard_center, bg_img=bg_img)
        
        if rendered.ndim == 3 and rendered.shape[2] == 4:
            img_rgb = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB).astype(np.float32)
            alpha = rendered[:, :, 3:4] / 255.0
            rendered_rgb = rendered[:, :, :3].astype(np.float32)
            composite = (1 - alpha) * img_rgb + alpha * rendered_rgb
            composite = np.clip(composite, 0, 255).astype(np.uint8)
        else:
            composite = rendered
        
        output_path = os.path.join(output_dir, f"output_{img_name}")
        cv2.imwrite(output_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        plt.figure(figsize=(6, 6))
        plt.imshow(composite)
        plt.title(f"Pikachu Projection on {img_name}")
        plt.axis("off")
        plt.show()
    
    print("Processing complete. Check the output images in the 'output' directory.")

# -------------------------------
# Main Function & Argument Parsing
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AR Object Placement with Pikachu Mesh and Chessboard Pose Estimation"
    )
    parser.add_argument(
        "--show-mesh", action="store_true",
        help="Show the Pikachu mesh in an interactive window and exit."
    )
    parser.add_argument(
        "--scale", type=float, default=20.0,
        help="Scale factor for the 3D model (e.g., try 0.0001 for a smaller model)."
    )
    parser.add_argument(
        "--translate", type=float, default=0.0,
        help="Additional translation along the z-axis after scaling (e.g., 10.0 to move it further away)."
    )
    args = parser.parse_args()
    
    print("Using scale factor:", args.scale)
    print("Using extra translation along z:", args.translate)
    mesh = load_mesh("Pikachu_tri.obj", scale_factor=args.scale, extra_translation=args.translate)
    
    if args.show_mesh:
        show_mesh_interactively(mesh)
    else:
        process_images(mesh)

if __name__ == "__main__":
    main()
