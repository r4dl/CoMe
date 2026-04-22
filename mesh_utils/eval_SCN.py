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
# please follow the instructions in setup.py before running this script.
import numpy as np
import open3d as o3d
import os
import argparse

from registration import (
    registration_vol_ds,
    registration_unif,
)
from evaluation import EvaluateHisto
from util import make_dir
from plot import plot_graph
import trimesh

def run_evaluation(dataset_dir, ply_path, out_dir, view_crop):
    print("before:" + dataset_dir)
    scene = os.path.basename(dataset_dir)
    print("after" + dataset_dir)
    print("Scene: " + scene)

    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    dTau = 0.05 # TODO
    gt_filen = os.path.join(dataset_dir, "scans", "mesh_aligned_0.05.ply")
    print("out: " + out_dir)
    make_dir(out_dir)

    # Load reconstruction and according GT
    print(ply_path)
    # add center points
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

    # transform pcd to gt_pcd space
    transform = np.identity(4)
    transform[:3, :3] = o3d.geometry.TriangleMesh.create_coordinate_frame().get_rotation_matrix_from_xyz(
        (0, np.pi, np.pi / 2))
    pcd.transform(transform)
    
    print("gt_filen: " +gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_filen)

    # big pointclouds will be downlsampled to this number to speed up alignment
    dist_threshold = dTau
    # Refine alignment by using the actual GT and MVS pointclouds
    vol = gt_pcd.get_axis_aligned_bounding_box()

    # Registration refinment in 3 iterations
    r2 = registration_vol_ds(pcd, gt_pcd, np.identity(4), vol, dTau,
                             dTau * 80, 20)
    r3 = registration_vol_ds(pcd, gt_pcd, r2.transformation, vol, dTau / 2.0,
                             dTau * 20, 20)
    r = registration_unif(pcd, gt_pcd, r3.transformation, vol, 2 * dTau, 20)
    trajectory_transform = r.transformation
    print("registration done")
    # Histogramms and P/R/F1
    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = EvaluateHisto(
        pcd,
        gt_pcd,
        trajectory_transform, # r.transformation,
        vol,
        dTau / 2.0,
        dTau,
        out_dir,
        plot_stretch,
        scene,
        view_crop
    )
    eva = [precision, recall, fscore]
    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")

    # What if we had a json to actually see whats up?
    import json
    results = {
        'dist tau': dTau,
        'precision': eva[0],
        'recall': eva[1],
        'fscore': eva[2],
    }
    
    with open(f'{out_dir}/results.json', 'w') as fp:
        json.dump(results, fp, indent=2)

    # Plotting
    plot_graph(
        scene,
        fscore,
        dist_threshold,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
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
        "--ply-path",
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

    args.view_crop = False #  (args.view_crop > 0)
    if args.out_dir.strip() == "":
        args.out_dir = os.path.join(os.path.dirname(args.ply_path),
                                    "evaluation")

    run_evaluation(
        dataset_dir=args.dataset_dir,
        ply_path=args.ply_path,
        out_dir=args.out_dir,
        view_crop=args.view_crop
    )
