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
import os
from importlib import import_module
import torch
import json
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, SplattingSettings, OptimizationParams, SplattingSettings, MeshingParams
from utils.depth_utils import depths_to_points, depth_to_normal, central_diff
from utils.vis_utils import gui_visualize, export_image
from scene.gaussian_model import build_scaling_rotation
from fused_ssim import fused_ssim
from diff_gaussian_rasterization import ExtendedSettings, DebugVisualization, DebugVisualizationType
import numpy as np
from scene.appearance_network import AppearanceEmbedding, VastGaussianAppearanceEmbedding, SSIMDecoupledAppearanceEmbedding
from functools import partial
import copy
from scene.densifier import AbsGradDensifier, MCMCDensifier, MSv2AbsGradDensifier
import warnings

RED = '\033[31m'
RESET = '\033[0m'

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# TODO: can we precompute this? should be easy enough (to store as well)
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if lr_init == 0:
            return 0
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return (delay_rate * log_lerp)

    return helper

def training(dataset, opt, pipe : PipelineParams, mesh : MeshingParams, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, splat_args: ExtendedSettings):
    import time
    start_event = time.time()
    
    first_iter = 0
    # TODO: reintroduce tensorboard to log how many Gaussians we densify, etc.
    tb_writer = prepare_output_and_logger(dataset, splat_args, opt, pipe, mesh)
    gaussians = GaussianModel(dataset.sh_degree, use_SBs=pipe.convert_SBs_python)
    scene = Scene(dataset, gaussians, MCMC_init=mesh.cap_max != -1)
    trainCameras = scene.getTrainCameras().copy() 
    
    if mesh.use_vastgaussian_appearance:
        appearance_embedding = VastGaussianAppearanceEmbedding(num_views=len(trainCameras), lambda_ssim=opt.lambda_dssim)
    elif mesh.use_ssimdecoupled_appearance:
        appearance_embedding = SSIMDecoupledAppearanceEmbedding(num_views=len(trainCameras), lambda_ssim=opt.lambda_dssim)
    else:
        warnings.warn("Unknown appearance embedding, using default (No Appearance Embedding)")
        appearance_embedding = AppearanceEmbedding(num_views=len(trainCameras), lambda_ssim=opt.lambda_dssim)
    gaussians.training_setup(opt, mesh, appearance_embedding)
    if checkpoint:
        (model_params, first_iter, (_appearance_embedding, _appearance_net)) = torch.load(checkpoint)
        appearance_embedding.restore(_appearance_embedding, _appearance_net)
        gaussians.restore(model_params, opt, mesh, appearance_embedding)
        

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # TODO: same strategy as for the appearance embedding
    if mesh.use_msv2_simplification:
        densifier = MSv2AbsGradDensifier(gaussians, opt, mesh, dataset, pipe)
    elif mesh.cap_max == -1:
        densifier = AbsGradDensifier(gaussians, opt, mesh, dataset, pipe)
    else:
        densifier = MCMCDensifier(gaussians, opt, mesh, dataset, pipe)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    for idx, camera in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
        camera.idx = idx
    # because I did this error once in the past
    del camera, idx
        
    # at first, we don't need the opacity
    splat_args.render_opacity = False
    
    gaussians.compute_3D_filter(cameras=trainCameras, CUDA=not pipe.compute_filter3D_python)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, message = network_gui.receive()
                if custom_cam != None:
                    with torch.no_grad():
                        debugVis = DebugVisualization(**message["debug_data"])
                        net_image = render(custom_cam, gaussians, pipe, background, message["scaling_modifier"], splat_args=splat_args, debugVis=debugVis)["render"]

                    if debugVis.type == 0 or debugVis.type == DebugVisualizationType.CONFIDENCE:
                        image = gui_visualize(
                            render_cam=custom_cam,
                            alpha=net_image[7:8],
                            distortion=net_image[8:9],
                            depth=net_image[6:7],
                            normal=net_image[3:6],
                            confidence=net_image[10:11],
                            render=net_image[:3],
                            color_variance=net_image[11:12],
                            normal_variance=net_image[12:13],
                            other_args=message
                        )
                        if message["render_appearance_embedding"]:
                            image = net_image[:3]
                            image = appearance_embedding.appearance_mapping(image, message["camera_idx"])
                    else:
                        image = net_image[:3]

                    image = torch.clamp(image, 0., 1.)
                    net_image_bytes = memoryview((image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                net_image_bytes = memoryview((image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, json.dumps({"method_dir": dataset.model_path}))
                if bool(message["train"]) and ((iteration < int(opt.iterations)) or not bool(message["keep_alive"])):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        if iteration > mesh.distortion_from_iter and mesh.lambda_opacity_field > 0.0:
            splat_args.render_opacity = True

        gt_image = viewpoint_cam.original_image.cuda()
        # not sure we need detach here
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, splat_args=splat_args, gt_color=gt_image.detach())
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        image = rendering[:3, :, :]
        
        # custom variance losses
        variance = rendering[11, :, :]
        normal_variance = rendering[12, :, :]
      
        confidence_pp_rgb_loss_mean = None
        confidence_scaled_rgb_loss_mean = None
        confidence_log_term_mean = None
        confidence_neg_alpha_log_term_mean = None

        # TODO: don't mean the SSIM
        if mesh.color_confidence and iteration >= mesh.color_confidence_from_iter:  

            # Use a higher minimum to avoid numerical instability in log and gradient computation
            # log(1e-3) ≈ -6.9, which gives gradient of ~200 instead of ~200,000 at minimum
            # This prevents gradient explosion while maintaining the regularization effect
            confidence = torch.clamp(rendering[10, :, :], min=1e-3, max=5.0).unsqueeze(0)

            pp_rgb_loss = appearance_embedding(image, gt_image, viewpoint_cam.idx)           
            alpha = mesh.color_confidence_max

            # Numerically stable: higher minimum clamp prevents extreme gradients
            # Gradient w.r.t. confidence: rgb_loss_og - alpha/confidence
            # At confidence=1e-3: alpha/confidence = 0.2/1e-3 = 200 (reasonable)
            # At confidence=1e-6: alpha/confidence = 0.2/1e-6 = 200,000 (problematic)
            rgb_loss = pp_rgb_loss * confidence - alpha * torch.log(confidence)

            # Confidence-specific terms for TensorBoard diagnostics.
            confidence_pp_rgb_loss_mean = pp_rgb_loss.mean()
            confidence_scaled_rgb_loss_mean = (pp_rgb_loss * confidence).mean()
            confidence_log_term_mean = torch.log(confidence).mean()
            confidence_neg_alpha_log_term_mean = (-alpha * torch.log(confidence)).mean()
            # TODO: confidence into a CUDA kernel for speed
        else:
            rgb_loss = appearance_embedding(image, gt_image, viewpoint_cam.idx)

        # depth distortion regularization
        distortion_map = rendering[8, :, :]
        distortion_loss = distortion_map.mean()
        
        # depth normal consistency
        depth = rendering[6, :, :]
        if depth.isnan().sum() > 0:
            print("DEPTH IS NAN!!!!!")
            depth[depth.isnan()] = 0.0
        depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
        depth_normal = depth_normal.permute(2, 0, 1)

        render_normal = rendering[3:6, :, :]
        render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
        
        # c2w = (viewpoint_cam.world_view_transform.T).inverse()
        # if we only need the rotation, why bother with the inverse
        c2w = (viewpoint_cam.world_view_transform)
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
        
        nabla_I = central_diff(viewpoint_cam.original_image.permute(1,2,0)).cuda()
        
        normal_error = (1 - (render_normal_world * depth_normal).sum(dim=0))
        depth_normal_loss = normal_error.mean()
        
        lambda_distortion = mesh.lambda_distortion if iteration >= mesh.distortion_from_iter else 0.0
        lambda_depth_normal = mesh.lambda_depth_normal if iteration >= mesh.depth_normal_from_iter else 0.0
            
        # Normal regularization (smoothness)
        normal_loss = central_diff(render_normal_world.permute(1,2,0)) * torch.exp(-nabla_I)
        normal_loss = normal_loss.mean()
        lambda_normal = mesh.lambda_smoothness if iteration >= mesh.depth_normal_from_iter else 0.0

        lambda_opacity_field = mesh.lambda_opacity_field if iteration >= mesh.distortion_from_iter else 0.0
        opacity = rendering[7]
        opa_loss = (opacity - 0.5)**2

        #Ll1opacity_smoothness = central_diff(rendering[7][..., None]) * torch.exp(-nabla_I)
        opa_loss = opa_loss.mean()
        
        lambda_extent = mesh.lambda_extent if iteration >= mesh.distortion_from_iter else 0.0
        extent_loss = rendering[9]
        extent_loss = extent_loss.mean()
        
        rgb_loss_mean = rgb_loss.mean()
        
        lambda_variance = mesh.lambda_variance if iteration >= mesh.variance_from_iter else 0.0
        lambda_normal_variance = mesh.lambda_normal_variance if iteration >= mesh.normal_variance_from_iter else 0.0
        variance_loss = variance.mean()
        normal_variance_loss = normal_variance.mean()
        
        # Final loss
        loss =  rgb_loss_mean + \
                depth_normal_loss    * lambda_depth_normal + \
                distortion_loss      * lambda_distortion +  \
                normal_loss          * lambda_normal + \
                opa_loss             * lambda_opacity_field + \
                extent_loss          * lambda_extent + \
                variance_loss        * lambda_variance + \
                normal_variance_loss * lambda_normal_variance

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Size": f"{len(gaussians._xyz)}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration % 10 == 0 or iteration == opt.iterations:
                training_report(
                    tb_writer=tb_writer,
                    iteration=iteration,
                    rgb_loss=rgb_loss_mean,
                    total_loss=loss,
                    elapsed_ms=iter_start.elapsed_time(iter_end),
                    depth_normal_loss=depth_normal_loss,
                    lambda_depth_normal=lambda_depth_normal,
                    distortion_loss=distortion_loss,
                    lambda_distortion=lambda_distortion,
                    normal_loss=normal_loss,
                    lambda_normal=lambda_normal,
                    opacity_loss=opa_loss,
                    lambda_opacity_field=lambda_opacity_field,
                    extent_loss=extent_loss,
                    lambda_extent=lambda_extent,
                    variance_loss=variance_loss,
                    lambda_variance=lambda_variance,
                    confidence_pp_rgb_loss_mean=confidence_pp_rgb_loss_mean,
                    confidence_scaled_rgb_loss_mean=confidence_scaled_rgb_loss_mean,
                    confidence_log_term_mean=confidence_log_term_mean,
                    confidence_neg_alpha_log_term_mean=confidence_neg_alpha_log_term_mean,
                )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, appearance_embedding.capture())

            # Densification (AbsGrad or MCMC)
            temp_splat_args = copy.deepcopy(splat_args)
            temp_splat_args.consider_max_weight = True
            render_simp = partial(render, pipe=pipe, bg_color=background, splat_args=temp_splat_args)
            densifier.densify(
                iteration=iteration,
                visibility_filter=visibility_filter,
                radii=radii,
                viewspace_point_tensor=viewspace_point_tensor,
                cameras_extent=scene.cameras_extent,
                trainCameras=trainCameras,
                render_simp=render_simp
            )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                densifier.postfix(xyz_lr=xyz_lr)
            
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration, appearance_embedding.capture()), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    end_event = time.time() 
    
    print(f'Training in {end_event - start_event :.4f} seconds!')

def prepare_output_and_logger(args, settings: ExtendedSettings, opt, pipe, mesh):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
        
    # write config file
    with open(os.path.join(args.model_path, "config.json"), 'w') as config_json:
        json.dump(settings.to_dict(), config_json)

    # write output config files for opt, pipe, mesh
    with open(os.path.join(args.model_path, "mesh_args"), 'w') as f:
        f.write(str(Namespace(**vars(mesh))))
    with open(os.path.join(args.model_path, "rem_args"), 'w') as f:
        f.write(str(Namespace(**{**vars(opt), **vars(pipe)})))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_log_dir = os.path.join(args.model_path, "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_log_dir)

        scene_name = os.path.basename(os.path.normpath(args.model_path))
        tb_writer.add_text("run/scene_name", scene_name, 0)
        tb_writer.add_text("run/model_path", args.model_path, 0)
        tb_writer.add_text(
            "run/config",
            (
                f"iterations={opt.iterations}\n"
                f"lambda_dssim={opt.lambda_dssim}\n"
                f"lambda_distortion={mesh.lambda_distortion}\n"
                f"lambda_depth_normal={mesh.lambda_depth_normal}\n"
                f"lambda_smoothness={mesh.lambda_smoothness}\n"
                f"lambda_opacity_field={mesh.lambda_opacity_field}\n"
                f"lambda_extent={mesh.lambda_extent}\n"
                f"lambda_variance={mesh.lambda_variance}"
            ),
            0,
        )
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer,
    iteration,
    rgb_loss,
    total_loss,
    elapsed_ms,
    depth_normal_loss,
    lambda_depth_normal,
    distortion_loss,
    lambda_distortion,
    normal_loss,
    lambda_normal,
    opacity_loss,
    lambda_opacity_field,
    extent_loss,
    lambda_extent,
    lambda_variance,
    variance_loss,
    confidence_pp_rgb_loss_mean=None,
    confidence_scaled_rgb_loss_mean=None,
    confidence_log_term_mean=None,
    confidence_neg_alpha_log_term_mean=None,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss/rgb_loss", rgb_loss.item(), iteration)
        tb_writer.add_scalar("train_loss/total_loss", total_loss.item(), iteration)
        tb_writer.add_scalar("timing/iter_time_ms", elapsed_ms, iteration)

        tb_writer.add_scalar("regularization/depth_normal", depth_normal_loss.item(), iteration)
        tb_writer.add_scalar("regularization/distortion", distortion_loss.item(), iteration)
        tb_writer.add_scalar("regularization/normal_smoothness", normal_loss.item(), iteration)
        tb_writer.add_scalar("regularization/opacity_field", opacity_loss.item(), iteration)
        tb_writer.add_scalar("regularization/extent", extent_loss.item(), iteration)
        tb_writer.add_scalar("regularization/variance", variance_loss.item(), iteration)
        
        if lambda_depth_normal > 0.0:
            tb_writer.add_scalar(
                "weighted_regularization/depth_normal",
                (depth_normal_loss * lambda_depth_normal).item(),
                iteration,
            )
        if lambda_distortion > 0.0:
            tb_writer.add_scalar(
                "weighted_regularization/distortion",
                (distortion_loss * lambda_distortion).item(),
                iteration,
            )
        if lambda_normal > 0.0:
            tb_writer.add_scalar(
                "weighted_regularization/normal_smoothness",
                (normal_loss * lambda_normal).item(),
                iteration,
            )
        if lambda_opacity_field > 0.0:
            tb_writer.add_scalar(
                "weighted_regularization/opacity_field",
                (opacity_loss * lambda_opacity_field).item(),
                iteration,
            )
        if lambda_extent > 0.0:
            tb_writer.add_scalar(
                "weighted_regularization/extent",
                (extent_loss * lambda_extent).item(),
                iteration,
            )
        if lambda_variance > 0.0:
            tb_writer.add_scalar(
                "weighted_regularization/variance",
                (variance_loss * lambda_variance).item(),
                iteration,
            )

        if confidence_pp_rgb_loss_mean is not None:
            tb_writer.add_scalar(
                "confidence_terms/pp_rgb_loss_mean",
                confidence_pp_rgb_loss_mean.item(),
                iteration,
            )
        if confidence_scaled_rgb_loss_mean is not None:
            tb_writer.add_scalar(
                "confidence_terms/pp_rgb_loss_scaled_mean",
                confidence_scaled_rgb_loss_mean.item(),
                iteration,
            )
        if confidence_log_term_mean is not None:
            tb_writer.add_scalar(
                "confidence_terms/log_confidence_mean",
                confidence_log_term_mean.item(),
                iteration,
            )
        if confidence_neg_alpha_log_term_mean is not None:
            tb_writer.add_scalar(
                "confidence_terms/neg_alpha_log_confidence_mean",
                confidence_neg_alpha_log_term_mean.item(),
                iteration,
            )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    mp = MeshingParams(parser)
    ss = SplattingSettings(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    splat_args = ss.get_settings(args)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), mp.extract(args), 
             args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, splat_args)

    # All done
    print("\nTraining complete.")
