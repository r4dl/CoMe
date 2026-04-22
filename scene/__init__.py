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
from pathlib import Path
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
import warnings
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], skip_train=False, skip_test=False, MCMC_init=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, init_type=args.init_type)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "dslr")):
            scene_info = sceneLoadTypeCallbacks["ScanNET"](args.source_path, args.eval, init_type=args.init_type)
        else:
            assert False, "Could not recognize scene type!"

        # no need to store the initial point cloud
        # if not self.loaded_iter:
        #     with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
        #         dest_file.write(src_file.read())

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            
        if not self.loaded_iter:
            json_cams = []
            id_ = 0
            for _, cam in enumerate(scene_info.train_cameras):
                json_cams.append(camera_to_JSON(id_, cam, is_train_camera=True))
                id_ += 1
            for offset, cam in enumerate(scene_info.test_cameras):
                json_cams.append(camera_to_JSON(id_ + offset, cam, is_train_camera=False))
                
            # just to be extra safe this time
            cameras_path = Path(self.model_path) / "cameras.json"
            if cameras_path.exists():
                warnings.warn(f"Cameras file already exists at {cameras_path}")
            else:
                with open(cameras_path, 'w') as file:
                    json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            if not skip_train:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            if not skip_test:
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, MCMC_init=MCMC_init)

    def save(self, iteration, appearance_embedding):
        import torch
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        try:
            with open(os.path.join(point_cloud_path, "appearance_embedding.pth"), 'wb') as file:
                torch.save(appearance_embedding, file)
        except Exception as e:
            print(f"Error saving appearance embedding: {e}")

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]