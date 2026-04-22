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

import json
import copy
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def read_alignment_transformation(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return np.asarray(data["transformation"]).reshape((4, 4)).transpose()


def write_color_distances(path, pcd, distances, max_distance):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # cmap = plt.get_cmap("afmhot")
    cmap = plt.get_cmap("hot_r")
    distances = np.array(distances)
    colors = cmap(np.minimum(distances, max_distance) / max_distance)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)

def write_color_distances_2(path, pcd, distances_1, distances_2, max_distance):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # cmap = plt.get_cmap("afmhot")
    cmap = plt.get_cmap("bwr")
    d1 = np.minimum(max_distance, np.array(distances_1))
    d2 = np.minimum(max_distance, np.array(distances_2))
    dist = (d1 - d2) / 2
    dist += max_distance / 2
    colors = cmap(dist / max_distance)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)

def EvaluateHisto(
    source,
    target,
    trans,
    crop_volume,
    voxel_size,
    threshold,
    filename_mvs,
    plot_stretch,
    scene_name,
    view_crop,
    verbose=True,
):
    print("[EvaluateHisto]")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    s = copy.deepcopy(source)
    s.transform(trans)
    if crop_volume is not None:
        if isinstance(crop_volume, o3d.geometry.AxisAlignedBoundingBox):
            s = s.crop(crop_volume)
        else:
            s = crop_volume.crop_point_cloud(s)
        if view_crop:
            o3d.visualization.draw_geometries([s, ])
    else:
        print("No bounding box provided to crop estimated point cloud, leaving it as the loaded version!!")
    s = s.voxel_down_sample(voxel_size)
    print(filename_mvs + "/" + scene_name + ".precision.ply")

    t = copy.deepcopy(target)
    if crop_volume is not None:
        if isinstance(crop_volume, o3d.geometry.AxisAlignedBoundingBox):
            t = t.crop(crop_volume)
        else:
            t = crop_volume.crop_point_cloud(t)
    else:
        print("No bounding box provided to crop groundtruth point cloud, leaving it as the loaded version!!")

    t = t.voxel_down_sample(voxel_size)
    print("[compute_point_cloud_to_point_cloud_distance]")
    distance1 = s.compute_point_cloud_distance(t)
    print("[compute_point_cloud_to_point_cloud_distance]")
    distance2 = t.compute_point_cloud_distance(s)

    # write the distances to bin files
    # np.array(distance1).astype("float64").tofile(
    #     filename_mvs + "/" + scene_name + ".precision.bin"
    # )
    # np.array(distance2).astype("float64").tofile(
    #     filename_mvs + "/" + scene_name + ".recall.bin"
    # )

    # Colorize the poincloud files prith the precision and recall values

    source_n_fn = filename_mvs + "/" + scene_name + ".precision.ply"
    target_n_fn = filename_mvs + "/" + scene_name + ".recall.ply"

    # Align outputs to the mesh/source frame for visualization.
    s_vis = copy.deepcopy(s)
    t_vis = copy.deepcopy(t)
    if trans is not None:
        vis_trans = np.linalg.inv(trans)
        s_vis.transform(vis_trans)
        t_vis.transform(vis_trans)

    #print("[ViewDistances] Add color coding to visualize error")
    #eval_str_viewDT = (
    #    OPEN3D_EXPERIMENTAL_BIN_PATH
    #    + "ViewDistances "
    #    + source_n_fn
    #    + " --max_distance "
    #    + str(threshold * 3)
    #    + " --write_color_back --without_gui"
    #)
    #os.system(eval_str_viewDT)
    # write_color_distances(source_n_fn, s_vis, distance1, 3 * threshold)

    #print("[ViewDistances] Add color coding to visualize error")
    #eval_str_viewDT = (
    #    OPEN3D_EXPERIMENTAL_BIN_PATH
    #    + "ViewDistances "
    #    + target_n_fn
    #    + " --max_distance "
    #    + str(threshold * 3)
    #    + " --write_color_back --without_gui"
    #)
    #os.system(eval_str_viewDT)
    # write_color_distances(target_n_fn, t_vis, distance2, 3 * threshold)

    # get histogram and f-score
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = get_f1_score_histo2(threshold, filename_mvs, plot_stretch, distance1,
                            distance2)
    np.savetxt(filename_mvs + "/" + scene_name + ".recall.txt", cum_target)
    np.savetxt(filename_mvs + "/" + scene_name + ".precision.txt", cum_source)
    np.savetxt(
        filename_mvs + "/" + scene_name + ".prf_tau_plotstr.txt",
        np.array([precision, recall, fscore, threshold, plot_stretch]),
    )

    return [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ]
    
def EvaluateHisto2(
    source1,
    source2,
    target,
    trans1,
    trans2,
    crop_volume,
    voxel_size,
    threshold,
    filename_mvs,
    scene_name,
    view_crop,
    verbose=True,
):
    print("[EvaluateHisto]")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    s = copy.deepcopy(source1)
    s.transform(trans1)
    if crop_volume is not None:
        s = crop_volume.crop_point_cloud(s)
        if view_crop:
            o3d.visualization.draw_geometries([s, ])
    else:
        print("No bounding box provided to crop estimated point cloud, leaving it as the loaded version!!")
    s = s.voxel_down_sample(voxel_size)
    print(filename_mvs + "/" + scene_name + ".precision.ply")

    t = copy.deepcopy(target)
    if crop_volume is not None:
        t = crop_volume.crop_point_cloud(t)
    else:
        print("No bounding box provided to crop groundtruth point cloud, leaving it as the loaded version!!")

    t = t.voxel_down_sample(voxel_size)
    # print("[compute_point_cloud_to_point_cloud_distance]")
    # distance11 = s.compute_point_cloud_distance(t)
    print("[compute_point_cloud_to_point_cloud_distance]")
    distance21 = t.compute_point_cloud_distance(s)

    s = copy.deepcopy(source2)
    s.transform(trans2)
    if crop_volume is not None:
        s = crop_volume.crop_point_cloud(s)
        if view_crop:
            o3d.visualization.draw_geometries([s, ])
    else:
        print("No bounding box provided to crop estimated point cloud, leaving it as the loaded version!!")
    s = s.voxel_down_sample(voxel_size)
    print(filename_mvs + "/" + scene_name + ".precision.ply")
    
    # print("[compute_point_cloud_to_point_cloud_distance]")
    # distance12 = s.compute_point_cloud_distance(t)
    print("[compute_point_cloud_to_point_cloud_distance]")
    distance22 = t.compute_point_cloud_distance(s)

    # write the distances to bin files
    # np.array(distance1).astype("float64").tofile(
    #     filename_mvs + "/" + scene_name + ".precision.bin"
    # )
    # np.array(distance2).astype("float64").tofile(
    #     filename_mvs + "/" + scene_name + ".recall.bin"
    # )

    # Colorize the poincloud files prith the precision and recall values
    o3d.io.write_point_cloud(
        filename_mvs + "/" + scene_name + ".precision.ply", s
    )
    o3d.io.write_point_cloud(filename_mvs + "/" + scene_name + ".recall.ply", t)

    source_n_fn = filename_mvs + "/" + scene_name + ".precision_compare.ply"
    target_n_fn = filename_mvs + "/" + scene_name + ".recall_compare.ply"

    print("[ViewDistances] Add color coding to visualize error")
    #eval_str_viewDT = (
    #    OPEN3D_EXPERIMENTAL_BIN_PATH
    #    + "ViewDistances "
    #    + source_n_fn
    #    + " --max_distance "
    #    + str(threshold * 3)
    #    + " --write_color_back --without_gui"
    #)
    #os.system(eval_str_viewDT)
    write_color_distances_2(target_n_fn, t, distance21, distance22, 3 * threshold)


def get_f1_score_histo2(threshold,
                        filename_mvs,
                        plot_stretch,
                        distance1,
                        distance2,
                        verbose=True):
    print("[get_f1_score_histo2]")
    dist_threshold = threshold
    if len(distance1) and len(distance2):

        recall = float(sum(d < threshold for d in distance2)) / float(
            len(distance2))
        precision = float(sum(d < threshold for d in distance1)) / float(
            len(distance1))
        fscore = 2 * recall * precision / (recall + precision)
        num = len(distance1)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_source = np.histogram(distance1, bins)
        cum_source = np.cumsum(hist).astype(float) / num

        num = len(distance2)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_target = np.histogram(distance2, bins)
        cum_target = np.cumsum(hist).astype(float) / num

    else:
        precision = 0
        recall = 0
        fscore = 0
        edges_source = np.array([0])
        cum_source = np.array([0])
        edges_target = np.array([0])
        cum_target = np.array([0])

    return [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ]
