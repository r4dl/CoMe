import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import rendering

USE_NORMALS = True

def srgb_to_linear(img):
    img = img / 255.0
    return np.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4
    )

def load_and_show_ply2(filepath):
    mesh = o3d.io.read_triangle_mesh(filepath)
    mesh.compute_vertex_normals()
    
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])

def is_triangle_mesh(filepath):
    if filepath.endswith(".obj"):
        return True

    # Check the contents of the PLY file for the 'element face' line
    try:
        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                if 'element face' in line:
                    return True
    except Exception as e:
        print(f"Error reading file: {e}")
    return False

# if we want to render videos down the line, this might be nice
# https://www.open3d.org/docs/0.12.0/tutorial/visualization/customized_visualization.html#change-field-of-view
def build_camera_parameters(selected_camera, window_width, window_height):
    R = np.asarray(selected_camera.get("rotation"))
    t = np.asarray(selected_camera.get("position")).reshape(3)
    fx, fy = selected_camera.get("fx"), selected_camera.get("fy")
    cx, cy = selected_camera.get("width") / 2, selected_camera.get("height") / 2

    # Scale intrinsics based on height to keep vertical FOV consistent,
    # and center horizontally if the aspect ratio changes.
    base_width = selected_camera.get("width")
    base_height = selected_camera.get("height")
    scale = window_height / base_height
    fx = fx * scale
    fy = fy * scale
    cx = cx * scale
    cy = cy * scale
    scaled_width = base_width * scale
    cx += (window_width - scaled_width) * 0.5

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        window_width,
        window_height,
        fx,
        fy,
        cx,
        cy,
    )

    extrinsic = np.eye(4, dtype=float)
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = -R.T @ t

    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intrinsic
    params.extrinsic = extrinsic
    return params


def resolve_screenshot_path(screenshot_path, image_name):
    if screenshot_path is None:
        return None
    target_dir = Path(screenshot_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{image_name}.png"


def load_and_show_ply(
    filepath,
    *,
    flip_triangles=False,
    color_normals=True,
    window_width=1600,
    window_height=1200,
    screenshot_path=None,
    image_name=None,
):
    IS_MESH = is_triangle_mesh(filepath)
    if IS_MESH:
        mesh = o3d.io.read_triangle_mesh(filepath)
        if flip_triangles:
            triangles = np.asarray(mesh.triangles)
            # flip triangles
            triangles = triangles[:, [0, 2, 1]]
            mesh = o3d.geometry.TriangleMesh(vertices=mesh.vertices, triangles=o3d.utility.Vector3iVector(triangles))

        mesh.compute_vertex_normals()
        print("read triangle")
    else:
        mesh = o3d.io.read_point_cloud(filepath)
        print("read point cloud")
    # Visualize the geometry (whether it's a point cloud or triangle mesh)
    # o3d.visualization.draw_geometries([mesh])

    if IS_MESH and color_normals and mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals)
        colors = (normals + 1) / 2  # Normalize to [0, 1]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    CAM_FILE_FOUND = False
    try:
        # TODO: load the camera from the cameras.json file
        cameras_json = None
        cameras_path = Path(filepath).parents[2] / "cameras.json"
        if not cameras_path.exists():
            cameras_path = Path(filepath).parents[1] / "cameras.json"
        with open(cameras_path, "r") as f:
            cameras_json = json.load(f)
        CAM_FILE_FOUND = True
        
        if image_name:
            image_names = image_name if isinstance(image_name, list) else [image_name]
            selected_cameras = [
                cameras_json[next(i for i, l in enumerate(cameras_json) if l["img_name"] == name)]
                for name in image_names
            ]
        else:
            image_names = []
            selected_cameras = [cameras_json[0]]

        
    except Exception as e:
        print(f"Error reading camera file: {e}")
        
    if screenshot_path:
        # Offscreen rendering to ensure exact output size.
        render = rendering.OffscreenRenderer(window_width, window_height)
        render.scene.set_background([0.05, 0.08, 0.2, 1.0])

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        if not IS_MESH:
            material.point_size = 2.0

        render.scene.add_geometry("geometry", mesh, material)
        render.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0, 0, 0))
        render.scene.scene.enable_indirect_light(False)
        render.scene.scene.enable_sun_light(False)
        render.scene.scene.set_indirect_light_intensity(0.)
        render.scene.set_background([0.5, 0.5, 0.5, 1.0])  # mid-gray

        for name, camera in zip(image_names or ["screenshot"], selected_cameras):
            params = build_camera_parameters(camera, window_width, window_height)
            render.setup_camera(params.intrinsic, params.extrinsic)
            image = render.render_to_image()
            image_np = np.asarray(image)
            if image_np.shape[-1] == 4:
                rgb_linear = srgb_to_linear(image_np[:, :, :3])
                alpha = image_np[:, :, 3:4] / 255.0
                linear_np = np.concatenate([rgb_linear, alpha], axis=2)
            else:
                linear_np = srgb_to_linear(image_np)
            linear_8bit = np.clip(linear_np * 255.0, 0, 255).astype(np.uint8)
            image = o3d.geometry.Image(linear_8bit)
            target_path = resolve_screenshot_path(screenshot_path, name)
            o3d.io.write_image(str(target_path), image)
    else:
        # Interactive visualizer path.
        vis = o3d.visualization.Visualizer()
        filename = os.path.basename(filepath)
        vis.create_window(window_name=filename, width=window_width, height=window_height)
        vis.add_geometry(mesh)
        ctr = vis.get_view_control()
        ctr.set_constant_z_near(0.001)
        ctr.set_constant_z_far(1000)

        if CAM_FILE_FOUND:
            params = build_camera_parameters(selected_cameras[0], window_width, window_height)
            ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

        renderoption = vis.get_render_option()
        renderoption.light_on = False
        renderoption.background_color = np.asarray([0.05, 0.08, 0.2])

        vis.run()
        vis.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser(description="Quick viewer for meshes or point clouds.")
    parser.add_argument("path", help="Path to the mesh/point cloud file to visualize.")
    parser.add_argument(
        "--flip-triangles",
        action="store_true",
        help="Flip triangle winding before rendering (useful if normals look inverted).",
    )
    parser.add_argument(
        "--no-color-normals",
        action="store_true",
        help="Disable coloring vertices by their normals.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1600,
        help="Viewer window width in pixels (default: 1600).",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=1200,
        help="Viewer window height in pixels (default: 1200).",
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="If set, save screenshots into this directory.",
    )
    parser.add_argument(
        "--image_name",
        type=str,
        nargs="+",
        default=None,
        help="Image name(s) from cameras.json to render.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("press h for controls (;")

    load_and_show_ply(
        args.path,
        flip_triangles=args.flip_triangles,
        color_normals=not args.no_color_normals,
        window_width=args.window_width,
        window_height=args.window_height,
        screenshot_path=args.screenshot,
        image_name=args.image_name,
    )