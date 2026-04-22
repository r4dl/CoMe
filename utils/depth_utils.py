# copy from 2DGS
import math
import torch
import numpy as np

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

threshold = 2
def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.cross(dx, dy, dim=-1)
    
    #w = torch.clamp_max(1.0 / (dx*dx + dy*dy), 2.0).detach()
    #boundary_mask = torch.abs(dx[0,:]) > threshold | torch.abs(dy[0,:]) > threshold
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points

threshold = 2
def central_diff(image):
    """
        image
    """
    output = torch.zeros_like(image)[:,:,0]
    dx = torch.cat([image[2:, 1:-1] - image[:-2, 1:-1]], dim=0)
    dy = torch.cat([image[1:-1, 2:] - image[1:-1, :-2]], dim=1)
    
    #w = torch.clamp_max(1.0 / (dx*dx + dy*dy), 2.0).detach()
    #boundary_mask = torch.abs(dx[0,:]) > threshold | torch.abs(dy[0,:]) > threshold
    output[1:-1, 1:-1] = torch.norm(dx, dim=-1) + torch.norm(dy, dim=-1)
    return output

