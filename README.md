# AR-object
# 🧠 Augmented Reality Object Projection

This project demonstrates real-time planar **Augmented Reality (AR)** by overlaying 2D and 3D virtual objects onto live video footage using computer vision techniques.

---

## 📸 Overview

This AR system uses feature matching and homography to detect planar surfaces and project either images or 3D models onto them. It's built with Python and OpenCV and is designed to run offline on any webcam feed or pre-recorded video.

### Features

- 📐 **Perspective Warping** – Detects reference images in real-time and warps content to fit the surface.
- 🖼️ **2D Object Rendering** – Overlays flat images (e.g., cartoons, textures) onto surfaces in the scene.
- 🧱 **3D Object Rendering** – Projects .OBJ models (like Pikachu!) realistically using pose estimation.
- 🎥 **Video Integration** – Supports both live webcam input and prerecorded videos.
- 🧪 **Modular Structure** – Separated scripts for testing warping, rendering, and object projection independently.

---


📌 Technologies Used

Python
OpenCV
NumPy
OpenGL / 3D rendering tools (for OBJ parsing)
👤 Author

Ido.s
📍 Computer Vision & AR Enthusiast
💼 2nd Year CS Student

Feel free to reach out or explore other projects on my GitHub!

