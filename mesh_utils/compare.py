# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

# this script requires Open3D python binding
# please follow the intructions in setup.py before running this script.
import numpy as np
import open3d as o3d
import os
import argparse
# import torch

from config import scenes_tau_dict
from registration import (
    trajectory_alignment,
    registration_vol_ds,
    registration_unif,
    read_trajectory,
)
# from help_func import auto_orient_and_center_poses
from trajectory_io import CameraPose
from evaluation import EvaluateHisto2
from util import make_dir
from plot import plot_graph


def run_evaluation(dataset_dir, traj_path, ply_path1, ply_path2, out_dir, view_crop):
    print("before:" + dataset_dir)
    scene = os.path.basename(dataset_dir)
    print("after" + dataset_dir)
    print("Scene: " + scene)
    if scene not in scenes_tau_dict:
        print(dataset_dir, scene)
        raise Exception("invalid dataset-dir, not in scenes_tau_dict")

    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    dTau = scenes_tau_dict[scene]
    # put the crop-file, the GT file, the COLMAP SfM log file and
    # the alignment of the according scene in a folder of
    # the same scene name in the dataset_dir
    colmap_ref_logfile = os.path.join(dataset_dir, scene + "_COLMAP_SfM.log")

    # this is for groundtruth pointcloud, we can use it
    alignment = os.path.join(dataset_dir, scene + "_trans.txt")
    gt_filen = os.path.join(dataset_dir, scene + ".ply")
    # this crop file is also w.r.t the groundtruth pointcloud, we can use it. 
    # Otherwise we have to crop the estimated pointcloud by ourself
    cropfile = os.path.join(dataset_dir, scene + ".json")
    # this is not so necessary
    map_file = os.path.join(dataset_dir, scene + "_mapping_reference.txt")
    if not os.path.isfile(map_file):
        map_file = None
    map_file = None
    print("out: " + out_dir)
    make_dir(out_dir)

    print("gt_filen: " +gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_filen)
    
    def align(ply_path):
        # Load reconstruction and according GT
        pcd = o3d.io.read_point_cloud(ply_path)
        # add center points
        import trimesh
        mesh = trimesh.load_mesh(ply_path)
        # add center points
        sampled_vertices = mesh.vertices[mesh.faces].mean(axis=1)
        # add 4 points based on the face vertices
        # face_vertices = mesh.vertices[mesh.faces]# .mean(axis=1)
        # weights = np.array([[3, 3, 3],
        #                     [4, 4, 1],
        #                     [4, 1, 4],
        #                     [1, 4, 4]],dtype=np.float32) / 9.0
        # sampled_vertices = np.sum(face_vertices.reshape(-1, 1, 3, 3) * weights.reshape(1, 4, 3, 1), axis=2).reshape(-1, 3)
        
        vertices = np.concatenate([mesh.vertices, sampled_vertices], axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        ### end add center points


        gt_trans = np.loadtxt(alignment)
        print("traj_path: " + traj_path)
        traj_to_register = []
        if traj_path.endswith('.npy'):
            ld = np.load(traj_path)
            for i in range(len(ld)):
                traj_to_register.append(CameraPose(meta=None, mat=ld[i]))
        elif traj_path.endswith('.json'): # instant-npg or sdfstudio format
            import json
            with open(traj_path, encoding='UTF-8') as f:
                meta = json.load(f)
            poses_dict = {}
            for i, frame in enumerate(meta['frames']):
                filepath = frame['file_path']
                new_i = int(filepath[13:18]) - 1
                poses_dict[new_i] = np.array(frame['transform_matrix'])
            poses = []
            for i in range(len(poses_dict)):
                poses.append(poses_dict[i])
            poses = torch.from_numpy(np.array(poses).astype(np.float32))
            poses, _ = auto_orient_and_center_poses(poses, method='up', center_poses=True)
            scale_factor = 1.0 / float(torch.max(torch.abs(poses[:, :3, 3])))
            poses[:, :3, 3] *= scale_factor
            poses = poses.numpy()
            for i in range(len(poses)):
                traj_to_register.append(CameraPose(meta=None, mat=poses[i]))

        else:
            traj_to_register = read_trajectory(traj_path)
        print("colmap_ref_logfile: " + colmap_ref_logfile)
        gt_traj_col = read_trajectory(colmap_ref_logfile)

        trajectory_transform = trajectory_alignment(map_file, traj_to_register,
                                                    gt_traj_col, gt_trans, scene)

        print("aligned Trajectories")
        # big pointclouds will be downlsampled to this number to speed up alignment
        dist_threshold = dTau
        # Refine alignment by using the actual GT and MVS pointclouds
        vol = o3d.visualization.read_selection_polygon_volume(cropfile)
        print("read_selection_polygon_volume")
        # Registration refinment in 3 iterations
        r2 = registration_vol_ds(pcd, gt_pcd, trajectory_transform, vol, dTau,
                                dTau * 80, 20)
        r3 = registration_vol_ds(pcd, gt_pcd, r2.transformation, vol, dTau / 2.0,
                                dTau * 20, 20)
        r = registration_unif(pcd, gt_pcd, r3.transformation, vol, 2 * dTau, 20)
        trajectory_transform = r.transformation
        
        return pcd, trajectory_transform, vol
    
    pcd1, trans1, vol = align(ply_path1)
    pcd2, trans2, vol = align(ply_path2)
        
    print("registration done")
    # Histogramms and P/R/F1
    plot_stretch = 5
    
    EvaluateHisto2(
        pcd1, pcd2,
        gt_pcd,
        trans1, trans2, # r.transformation,
        vol,
        dTau / 2.0,
        dTau,
        out_dir,
        scene,
        view_crop
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="path to a dataset/scene directory containing X.json, X.ply, ...",
    )
    parser.add_argument(
        "--traj-path",
        type=str,
        required=False,
        default="",
        help=
        "path to trajectory file. See `convert_to_logfile.py` to create this file. Default: \{dataset_dir\}/\{scene\}_traj_path.log",
    )
    parser.add_argument(
        "--ply-path1",
        type=str,
        required=True,
        help="path to reconstruction ply file",
    )
    parser.add_argument(
        "--ply-path2",
        type=str,
        required=True,
        help="path to reconstruction ply file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=
        "output directory, default: an evaluation directory is created in the directory of the ply file",
    )
    parser.add_argument(
        "--view-crop",
        type=int,
        default=0,
        help="whether view the crop pointcloud after aligned",
    )
    args = parser.parse_args()
    
    if args.traj_path == "":
        args.traj_path = args.dataset_dir + f'/{os.path.basename(args.dataset_dir)}_traj_path.log'
        print(args.traj_path)

    args.view_crop = False #  (args.view_crop > 0)
    if args.out_dir.strip() == "":
        args.out_dir = os.path.join(os.path.dirname(args.ply_path1),
                                    "evaluation")

    run_evaluation(
        dataset_dir=args.dataset_dir,
        traj_path=args.traj_path,
        ply_path1=args.ply_path1,
        ply_path2=args.ply_path2,
        out_dir=args.out_dir,
        view_crop=args.view_crop
    )
