# adapted from https://github.com/jzhangbs/DTUeval-python
import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import glob
from skimage.morphology import binary_dilation, disk
import argparse

import trimesh
from pathlib import Path

import render_utils as rend_util

def cull_scan(scan, mesh_path, result_mesh_file, instance_dir):
    
    image_dir = '{0}/images'.format(instance_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    n_images = len(image_paths)
    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())
    
    # load mask
    mask_dir = '{0}/mask'.format(instance_dir)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p)
        masks.append(mask)

    # hard-coded image shape
    W, H = 1600, 1200

    # load mesh
    mesh = trimesh.load(mesh_path)
    
    # load transformation matrix

    vertices = mesh.vertices

    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()

    sampled_masks = []
    for i in range(n_images):
        pose = pose_all[i]
        w2c = torch.inverse(pose).cuda()
        intrinsic = intrinsics_all[i].cuda()

        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()
            
            # dialate mask similar to unisurf
            maski = masks[i][:, :, 0].astype(np.float32) / 256.
            maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()
            
            # # if scan == '83': 
            # import matplotlib.pyplot as plt
            # plt.imshow(maski.cpu().numpy()[0,0])
            # points = (cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)).permute(1,0)[valid==1].cpu().numpy()
            # scatters = points[np.random.permutation(len(points))[:10000]]
            # plt.scatter(scatters[:,0], scatters[:,1], color='r')
            # plt.savefig(f'test{i}')
            # plt.clf()
            # plt.close()

            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]
            # print(f'culling {i}')
            sampled_mask = sampled_mask + (1. - valid)
            sampled_masks.append(sampled_mask)

    sampled_masks = torch.stack(sampled_masks, -1)
    # filter
    
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)

    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    # triangle_clusters = np.asarray(triangle_clusters)
    # cluster_n_triangles = np.asarray(cluster_n_triangles)
    # cluster_area = np.asarray(cluster_area)
    # largest_cluster_idx = cluster_n_triangles.argmax()
    # triangles_to_remove = (triangle_clusters != largest_cluster_idx)
    
    # transform vertices to world 
    scale_mat = scale_mats[0]
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    mesh.export(result_mesh_file)
    del mesh

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mesh', type=str, help='path to the mesh to be evaluated')
    parser.add_argument('--scan', type=int, default=None)
    parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument('--instance_dir', type=str,  help='path to the training data')
    parser.add_argument('--dataset_dir', type=str, default='.')
    parser.add_argument('--vis_out_dir', type=str, default='.')
    parser.add_argument('--downsample_density', type=float, default=0.2)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)
    parser.add_argument('--visualize_threshold', type=float, default=10)
    args = parser.parse_args()

    Offical_DTU_Dataset = args.dataset_dir
    out_dir = args.vis_out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    import re

    match = re.search(r'scan(\d{1,3})', args.instance_dir)
    scan_str = None

    if match:
        scan_str = match.group(1)
        print(scan_str)

    # prefer explicit --scan argument, else fall back to regex result
    if args.scan is not None:
        scan_id = args.scan
        if scan_str is not None and scan_id != int(scan_str):
            print(f"Warning: --scan ({scan_id}) overrides path-derived scan ({scan_str})")
    elif scan_str is not None:
        scan_id = int(scan_str)
    else:
        raise ValueError("Could not determine scan id; provide --scan or use an instance_dir containing 'scanXX'.")

    args.scan = scan_id
    ply_file = args.input_mesh
    
    result_mesh_file = os.path.join(out_dir, "culled_mesh.ply")
    cull_scan(scan_id, ply_file, result_mesh_file, args.instance_dir)
    
    print(args.input_mesh.split('/')[1])


    thresh = args.downsample_density
    if args.mode == 'mesh':
        pbar = tqdm(total=9)
        pbar.set_description('read data mesh')
        data_mesh = o3d.io.read_triangle_mesh(result_mesh_file)

        vertices = np.asarray(data_mesh.vertices)
        triangles = np.asarray(data_mesh.triangles)
        tri_vert = vertices[triangles]

        pbar.update(1)
        pbar.set_description('sample pcd from mesh')
        v1 = tri_vert[:,1] - tri_vert[:,0]
        v2 = tri_vert[:,2] - tri_vert[:,0]
        l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
        non_zero_area = (area2 > 0)[:,0]
        l1, l2, area2, v1, v2, tri_vert = [
            arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
        ]
        thr = thresh * np.sqrt(l1 * l2 / area2)
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)

        with mp.Pool() as mp_pool:
            new_pts = mp_pool.map(sample_single_tri, ((n1[i,0], n2[i,0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1,0]) for i in range(len(n1))), chunksize=1024)

        new_pts = np.concatenate(new_pts, axis=0)
        data_pcd = np.concatenate([vertices, new_pts], axis=0)
    
    elif args.mode == 'pcd':
        pbar = tqdm(total=8)
        pbar.set_description('read data pcd')
        data_pcd_o3d = o3d.io.read_point_cloud(result_mesh_file)
        data_pcd = np.asarray(data_pcd_o3d.points)

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description('masking data pcd')
    obs_mask_file = loadmat(f'{args.dataset_dir}/ObsMask/ObsMask{args.scan}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    patch = args.patch_size
    inbound = ((data_down >= BB[:1]-patch) & (data_down < BB[1:]+patch*2)).sum(axis=-1) ==3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) ==3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:,0], data_grid_in[:,1], data_grid_in[:,2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    pbar.set_description('read STL pcd')
    stl_pcd = o3d.io.read_point_cloud(f'{args.dataset_dir}/Points/stl/stl{args.scan:03}_total.ply')
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    max_dist = args.max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')
    ground_plane = loadmat(f'{args.dataset_dir}/ObsMask/Plane{args.scan}.mat')['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
    above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description('visualize error')
    vis_dist = args.visualize_threshold
    R = np.array([[1,0,0]], dtype=np.float64)
    G = np.array([[0,1,0]], dtype=np.float64)
    B = np.array([[0,0,1]], dtype=np.float64)
    W = np.array([[1,1,1]], dtype=np.float64)
    data_color = np.tile(B, (data_down.shape[0], 1))
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
    data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
    write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_d2s.ply', data_down, data_color)
    stl_color = np.tile(B, (stl.shape[0], 1))
    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
    stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
    write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_s2d.ply', stl, stl_color)

    pbar.update(1)
    pbar.set_description('done')
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2
    print(mean_d2s, mean_s2d, over_all)
    
    import json
    with open(f'{args.vis_out_dir}/results.json', 'w') as fp:
        json.dump({
            'mean_d2s': mean_d2s,
            'mean_s2d': mean_s2d,
            'overall': over_all,
        }, fp, indent=True)