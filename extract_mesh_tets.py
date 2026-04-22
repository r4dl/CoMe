import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render, integrate, ExtendedSettings, GlobalSortOrder
from arguments import BoundingSetting, MeshingSettings
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import SplattingSettings, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra
import open3d as o3d

RED = '\033[91m'
RESET = '\033[0m'
GREEN = '\033[92m'

DEPTH_CHANNEL = 6
EXPORT_STEPS = [7]

# This file is based on code from GOF (https://github.com/autonomousvision/gaussian-opacity-fields)
# https://github.com/autonomousvision/gaussian-opacity-fields/blob/5245b20e5d11acd6d1ff5af4b890dc2bedd99693/extract_mesh.py#L17

@torch.no_grad()
def evaluate_alpha(points, views, gaussians, pipeline, background, kernel_size, splat_args: ExtendedSettings, exact_alpha_eval = False):
    alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    
    if splat_args.meshing_settings.return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")
        # these settings are required to obtain faithful colors
        splat_args.meshing_settings.alpha_early_stop = False
        splat_args.sort_settings.sort_order = GlobalSortOrder.PTD_MAX
        exact_alpha_eval = True
    
    with torch.no_grad():       
        for _, view in enumerate(tqdm(views, desc="Meshing progress")):
            ret = integrate(points, alpha, view, gaussians, pipeline, background, kernel_size=kernel_size, splat_args=splat_args)

            if splat_args.meshing_settings.return_color:
                color_integrated = ret["color_integrated"]
                final_color = torch.where((alpha < final_alpha).reshape(-1, 1), color_integrated, final_color)
                
            if exact_alpha_eval:
                final_alpha = torch.min(final_alpha, alpha)
                alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda") * ~(alpha == 0)
            
        # if we have exact_eval, the result is in final_alpha, else it is in alpha
        if exact_alpha_eval:
            alpha = 1 - final_alpha
        else:
            alpha = 1 - alpha

    if splat_args.meshing_settings.return_color:
        return alpha, final_color
    return alpha, None

def save_opacity_field(points, field, filename):
    colors = torch.stack([field, torch.zeros_like(field), 1 - field], dim=1)

    # Convert tensors to numpy
    points_np = points.cpu().numpy()
    colors_np = (colors * 255).byte().cpu().numpy()

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np / 255.0)  # Normalize colors to [0,1]

    # Save as .ply
    o3d.io.write_point_cloud(filename, pcd)

@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, iteration, views, gaussians : GaussianModel, pipeline, background, kernel_size, meshing_settings : MeshingSettings, splat_args : ExtendedSettings):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))

    makedirs(render_path, exist_ok=True)
    
    # set mesh settings
    # if we sort by min z contribution, we can do early stopping in any case
    splat_args.sort_settings.sort_order = GlobalSortOrder.MIN_Z_BOUNDING
    
    # generate tetra points here
    points, points_scale = gaussians.get_tetra_points(views, meshing_settings)
    if meshing_settings.load_cells and os.path.exists(os.path.join(render_path, "cells.pt")):
        print("load existing cells")
        cells = torch.load(os.path.join(render_path, "cells.pt"), weights_only=True)
    else:
        # create cell and save cells
        print("create cells and save")
        cells = cpp.triangulate(points)
        # we should filter the cell if it is larger than the gaussians
        # torch.save(cells, os.path.join(render_path, "cells.pt"))
    
    # (1st Iteration): we don't need the exact depth anymore, lets accelerate
    splat_args.meshing_settings.alpha_early_stop = True
    
    alpha, _ = evaluate_alpha(points, views, gaussians, pipeline, background, kernel_size, splat_args=splat_args, exact_alpha_eval=True)

    vertices = points.cuda()[None]
    tets = cells.cuda().long()

    print(vertices.shape, tets.shape, alpha.shape)
    def alpha_to_sdf(alpha):    
        sdf = alpha - 0.5
        sdf = sdf[None]
        return sdf
    
    sdf = alpha_to_sdf(alpha)
    
    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
    torch.cuda.empty_cache()
    
    end_points, end_sdf = verts_list[0]
    
    faces=faces_list[0].cpu().numpy()
    del faces_list
        
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    mid_points = (left_points + right_points) / 2
    
    n_binary_steps = 8
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        
        splat_args.meshing_settings.return_color = False
        if step in EXPORT_STEPS and meshing_settings.texture_mesh:
            splat_args.meshing_settings.return_color = True
            
        alpha, color = evaluate_alpha(mid_points, views, gaussians, pipeline, background, kernel_size, splat_args=splat_args)
        mid_sdf = alpha_to_sdf(alpha).squeeze().unsqueeze(-1)
        
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
    
        mid_points = (left_points + right_points) / 2
        
        if step not in EXPORT_STEPS:
            continue
    
        if meshing_settings.texture_mesh:
            vertex_colors=(color.cpu().numpy() * 255).astype(np.uint8)
        else:
            vertex_colors=None
        mesh = trimesh.Trimesh(vertices=mid_points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)
        
        mesh.export(os.path.join(render_path, f"{meshing_settings.mesh_name}_{step}.ply"))
        print(f"Exported Mesh at step {step} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces to {os.path.join(render_path, f'{meshing_settings.mesh_name}_{step}.ply')}")
    

def extract_mesh(dataset : ModelParams, iteration : int, pipeline : PipelineParams, meshing_settings : MeshingSettings, splat_args : ExtendedSettings):
    dataset.init_type = "sfm"
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = 0.003
        
        # hotfix for SBs
        pipeline.convert_SBs_python = gaussians.use_SBs
        
        cams = scene.getTrainCameras()
        gaussians.compute_3D_filter(cameras=cams.copy())
        marching_tetrahedra_with_binary_search(dataset.model_path, "test", iteration, cams, gaussians, pipeline, background, kernel_size, meshing_settings, splat_args)

if __name__ == "__main__":
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--texture_mesh", action="store_true")
    parser.add_argument("--near", default=0.02, type=float)
    parser.add_argument("--far", default=1e6, type=float)
    parser.add_argument("--bounding_mode", type=lambda sortmode: BoundingSetting[sortmode], choices=list(BoundingSetting), default=BoundingSetting.STP)
    parser.add_argument("--mesh_name", type=str, default="mesh_faster_binary_search")
    parser.add_argument("--disable_near_far_culling", action="store_true", default=False)
    parser.add_argument("--opacity_cutoff_tetra", default=0.0039, type=float)
    parser.add_argument("--load_cells", action="store_true", default=False)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    print(args)
    
    mesh_settings = MeshingSettings(args.near, args.far, args.texture_mesh, args.bounding_mode, args.mesh_name,
                                    near_far_culling=not args.disable_near_far_culling, 
                                    opacity_cutoff_tetra=args.opacity_cutoff_tetra, 
                                    load_cells=args.load_cells)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    splat_args = ss.get_settings(args)
    
    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args), mesh_settings, splat_args=splat_args)