import copy
import os
import traceback
from typing import List
from argparse import Namespace
import imageio
import numpy as np
import torch
import torch.nn
from tqdm import tqdm
from pathlib import Path

from gaussian_renderer import render_simple
from scene import GaussianModel
from scene.cameras import CustomCam
from renderer.base_renderer import Renderer
from splatviz_utils.dict_utils import EasyDict
from utils.vis_utils import gui_visualize
from diff_gaussian_rasterization import DebugVisualizationType

class GaussianRenderer(Renderer):
    def __init__(self, num_parallel_scenes=16):
        super().__init__()
        self.num_parallel_scenes = num_parallel_scenes
        self.gaussian_models: List[GaussianModel | None] = [None] * num_parallel_scenes
        self._current_ply_file_paths: List[str | None] = [None] * num_parallel_scenes
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
        self._last_num_scenes = 0

    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        eval_text,
        resolution,
        ply_file_paths,
        cam_params,
        current_ply_names,
        background_color,
        video_cams=[],
        scaling_modifier=1,
        img_normalize=False,
        use_splitscreen=False,
        highlight_border=False,
        save_ply_path=None,
        slider={},
        splat_args=None,
        debug_data=None,
        **other_args,
    ):
        cam_params = cam_params.to("cuda")
        slider = EasyDict(slider)
        if len(ply_file_paths) == 0:
            res.error = "Select a .ply file"
            return

        # Remove old scenes
        if len(ply_file_paths) < self._last_num_scenes:
            for i in range(ply_file_paths, self.num_parallel_scenes):
                self.gaussian_models[i] = None
            self._last_num_scenes = len(ply_file_paths)

        images = []
        for scene_index, ply_file_path in enumerate(ply_file_paths):
            # Load
            if ply_file_path != self._current_ply_file_paths[scene_index]:
                self.gaussian_models[scene_index] = self._load_model(ply_file_path)
                self._current_ply_file_paths[scene_index] = ply_file_path

            # Edit
            gs: GaussianModel = copy.deepcopy(self.gaussian_models[scene_index])
            try:
                exec(self.sanitize_command(edit_text))
            except Exception as e:
                error = traceback.format_exc()
                error += str(e)
                res.error = error

            # Render video
            if len(video_cams) > 0:
                self.render_video("./_videos", video_cams, gs)

            # Render current view
            fov_rad = fov / 360 * 2 * np.pi
            render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
            gs.active_sh_degree = other_args.get('sh_degree')
            render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"), splat_args=splat_args, scaling_modifier=scaling_modifier, debug_data=debug_data)

            if other_args.get('render_appearance_embedding'):
                render["render"] = gs.appearance_embedding.appearance_mapping(render["render"], other_args.get('camera_idx'))

            if not other_args.get("render_confidence", False):
                other_args["render_confidence"] = debug_data.type == DebugVisualizationType.CONFIDENCE if debug_data is not None else False

            image = gui_visualize(
                render_cam=render_cam,
                alpha=render["alpha"],
                distortion=render["distortion"],
                depth=render["depth"],
                normal=render["normal"],
                render=render["render"],
                confidence=render["confidence"],
                color_variance=render["color_variance"],
                normal_variance=render["normal_variance"],
                other_args=other_args
            )
            images.append(image)            

            # Save ply
            if save_ply_path is not None:
                self.save_ply(gs, save_ply_path)

        self._return_image(
            images,
            res,
            normalize=img_normalize,
            use_splitscreen=use_splitscreen,
            highlight_border=highlight_border,
        )

        res.mean_xyz = torch.mean(gs.get_xyz, dim=0)
        res.std_xyz = torch.std(gs.get_xyz)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

    def _load_model(self, ply_file_path):
        if ply_file_path.endswith(".ply"):
            model = GaussianModel(sh_degree=3)
            model.load_ply(ply_file_path)
            
            # try to load the appearance embedding
            try:
                appearance_embedding_path = ply_file_path.replace("point_cloud.ply", "appearance_embedding.pth")
                appearance_embedding_params = torch.load(appearance_embedding_path, weights_only=True)
                from scene.appearance_network import AppearanceEmbedding
                app_embed = AppearanceEmbedding.load_from_capture(appearance_embedding_params)
                model.appearance_embedding = app_embed.cuda()
            except Exception as e:
                print(f"Error loading appearance embedding: {e}")
                
            # try to load the GT images (not needed, just store the path of the images)
            # scene_info_path = Path(ply_file_path).parent.parent.parent / "cfg_args"
            # if scene_info_path.exists():
            #     try:
            #         with open(scene_info_path, "r") as fp:
            #             cfgfile_string = fp.read()
            #         # cfg_args is stored as stringified argparse.Namespace
            #         model.cfg_args = eval(cfgfile_string, {"Namespace": Namespace})
            #         model.images_path = Path(model.cfg_args.source_path) / model.cfg_args.images
            #     except Exception as e:
            #         print(f"Error loading cfg_args: {e}")
                    
            
        else:
            raise NotImplementedError("Only .ply or .yml files are supported.")
        return model

    def render_video(self, save_path, video_cams, gaussian):
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/rotate_{len(os.listdir(save_path))}.mp4"
        video = imageio.get_writer(filename, mode="I", fps=30, codec="libx264", bitrate="16M", quality=10)
        for render_cam in tqdm(video_cams):
            img = render_simple(viewpoint_camera=render_cam, pc=gaussian, bg_color=self.bg_color)["render"]
            img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            video.append_data(img)
        video.close()
        print(f"Video saved in {filename}.")

    @staticmethod
    def save_ply(gaussian, save_ply_path):
        os.makedirs(save_ply_path, exist_ok=True)
        save_path = os.path.join(save_ply_path, f"model_{len(os.listdir(save_ply_path))}.ply")
        print("Model saved in", save_path)
        gaussian.save_ply(save_path)
