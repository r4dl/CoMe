#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import json
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from arguments import ModelParams, PipelineParams, SplattingSettings
from diff_gaussian_rasterization import ExtendedSettings
from gaussian_renderer import GaussianModel
from scene.cameras import CustomCam
from utils.vis_utils import gui_visualize

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, splat_args: ExtendedSettings):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, splat_args=splat_args)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering[0:3], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, image_name : list[str], skip_train : bool, skip_test : bool, splat_args: ExtendedSettings):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=args.skip_test, skip_train=False)
        
        cams = scene.getTrainCameras()
        gaussians.compute_3D_filter(cams.copy())

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if image_name is not None:
            
            cameras_path = os.path.join(dataset.model_path, "cameras.json")
            with open(cameras_path, "r") as f:
                cameras_json = json.load(f)
            image_names = image_name if isinstance(image_name, list) else [image_name]
            selected_cameras = [
                cameras_json[next(i for i, l in enumerate(cameras_json) if l["img_name"] == name)]
                for name in image_names
            ]
            
            render_path = os.path.join(dataset.model_path, "eval/render/")
            render_conf_path = os.path.join(dataset.model_path, "eval/confidence/")
            makedirs(render_path, exist_ok=True)
            makedirs(render_conf_path, exist_ok=True)
            
            # taking this as a reference
            ref_camera = cams[0]
            for c in selected_cameras:
                extr = torch.eye(4, device="cuda")
                extr[:3, :3] = torch.tensor(c["rotation"])
                extr[:3, 3] = torch.tensor(c["position"])
                c_cam = CustomCam(ref_camera.image_width, ref_camera.image_height, ref_camera.FoVy, ref_camera.FoVx, extr)
                rendering = render(c_cam, gaussians, pipeline, background, splat_args=splat_args)["render"]
                torchvision.utils.save_image(rendering[0:3], os.path.join(render_path, c["img_name"] + ".png"))

                conf_rendering = rendering[10:11]
                conf_w_colormap = gui_visualize(None, None, None, None, None, None,  conf_rendering, None, None, {"render_confidence":True})
                torchvision.utils.save_image(conf_w_colormap, os.path.join(render_conf_path, c["img_name"] + ".png"))
        else:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, splat_args)

            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, splat_args)
            
        # write number of gaussians too
        num_gaussians = scene.gaussians.get_xyz.shape[0]
        with open(os.path.join(dataset.model_path, "point_cloud", f'iteration_{scene.loaded_iter}', 'num_gaussians.json'), 'w') as fp:
            json.dump(obj={
                "num_gaussians": num_gaussians,
            }, fp=fp, indent=2)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--image_name", type=str, default=None, nargs="+", help="Image name(s) from cameras.json to render.")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    splat_args = ss.get_settings(args)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), getattr(args, "image_name", None), args.skip_train, args.skip_test, splat_args)