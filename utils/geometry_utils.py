import math
import torch
from typing import List
from utils.general_utils import build_rotation
from scene.cameras import Camera
from tqdm import tqdm


def transform_points_world_to_view(
    points:torch.Tensor,
    cameras:List[Camera],
    use_p3d_convention:bool=False,
):
    """Transform points from world space to view space.

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    world_view_transforms = torch.stack([camera.world_view_transform for camera in cameras], dim=0)  # (n_cameras, 4, 4)
    
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    view_points = (points_h @ world_view_transforms)[..., :3]  # (n_cameras, N, 3)
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        view_points = factors * view_points  # (n_cameras, N, 3)
    return view_points


def transform_points_view_to_world(
    points:torch.Tensor,
    cameras:List[Camera],
    use_p3d_convention:bool=False,
):
    """Transform points from view space to world space.

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    view_world_transforms = torch.stack([camera.world_view_transform.inverse() for camera in cameras], dim=0)  # (n_cameras, 4, 4)
    
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        points = factors * points  # (n_cameras, N, 3)
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    world_points = (points_h @ view_world_transforms)[..., :3]  # (n_cameras, N, 3)
    return world_points

        
def transform_points_to_pixel_space(
        points:torch.Tensor,
        cameras:List[Camera],
        points_are_already_in_view_space:bool=False,
        use_p3d_convention:bool=False,
        znear:float=1e-6,
        keep_float:bool=False,
    ):
        """Transform points from world space (3 coordinates) to pixel space (2 coordinates).

        Args:
            points (torch.Tensor): Should have shape (n_cameras, N, 3).
            cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
            points_are_already_in_view_space (bool, optional): Defaults to False.
            use_p3d_convention (bool, optional): Defaults to False.
            znear (float, optional): Defaults to 1e-6.

        Returns:
            torch.Tensor: Has shape (n_cameras, N, 2). 
                In pixel space, (0, 0) is the center of the left-top pixel,
                and (W-1, H-1) is the center of the right-bottom pixel.
        """
        if points_are_already_in_view_space:
            full_proj_transforms = torch.stack([camera.projection_matrix for camera in cameras])  # (n_depth, 4, 4)
            if use_p3d_convention:
                points = torch.tensor([[[-1, -1, 1]]], device=points.device) * points
        else:
            full_proj_transforms = torch.stack([camera.full_proj_transform for camera in cameras])  # (n_cameras, 4, 4)
        
        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
        proj_points = points_h @ full_proj_transforms  # (n_cameras, N, 4)
        proj_points = proj_points[..., :2] / proj_points[..., 3:4].clamp_min(znear)  # (n_cameras, N, 2)
        # proj_points is currently in a normalized space where 
        # (-1, -1) is the left-top corner of the left-top pixel,
        # and (1, 1) is the right-bottom corner of the right-bottom pixel.

        # For converting to pixel space, we need to scale and shift the normalized coordinates
        # such that (-1/2, -1/2) is the left-top corner of the left-top pixel, 
        # and (H-1/2, W-1/2) is the right-bottom corner of the right-bottom pixel.
        
        height, width = cameras[0].image_height, cameras[0].image_width
        image_size = torch.tensor([[width, height]], device=points.device)
        
        # proj_points = (1. + proj_points) * image_size / 2
        proj_points = (1. + proj_points) / 2 * image_size - 1./2.

        if keep_float:
            return proj_points        
        else:
            return torch.round(proj_points).long()


# the following functions are adopted from RaDe-GS: 
def depths_to_points(view, depthmap1, depthmap2=None):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins_inv = torch.tensor(
        [[1/fx, 0.,-W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy),],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float().cuda()
    rays_d = intrins_inv @ points
    points1 = depthmap1.reshape(1,-1) * rays_d
    if depthmap2 is not None:
        points2 = depthmap2.reshape(1,-1) * rays_d
        return points1.reshape(3,H,W), points2.reshape(3,H,W)
    else:
        return points1.reshape(3,H,W)


def point_to_normal(view, points1, points2=None):
    points = (
        points1[None] if points2 is None 
        else torch.stack([points1, points2],dim=0)
    )
    output = torch.zeros_like(points)
    dx = points[...,2:, 1:-1] - points[...,:-2, 1:-1]
    dy = points[...,1:-1, 2:] - points[...,1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=1), dim=1)
    output[...,1:-1, 1:-1] = normal_map
    return (
        output[0] if points2 is None 
        else output
    )


def depth_to_normal(view, depth1, depth2=None):
    points = depths_to_points(view, depth1, depth2)
    points = points[None] if depth2 is None else points
    return point_to_normal(view, *points)


def is_in_view_frustum(
    points:torch.Tensor,
    camera:Camera,
) -> torch.Tensor:
    """_summary_

    Args:
        points (torch.Tensor): Tensor with shape (N, 3)
        cameras (List[Camera]): _description_
    """
    H, W = camera.image_height, camera.image_width
    
    view_points = transform_points_world_to_view(
        points.view(1, -1, 3),
        cameras=[camera],
    )[0]  # (N, 3)
    
    pix_pts = transform_points_to_pixel_space(
        view_points.view(1, -1, 3),
        points_are_already_in_view_space=True,
        cameras=[camera],
    )[0]  # (N, 2)
    
    pix_x, pix_y, pix_z = pix_pts[..., 0], pix_pts[..., 1], view_points[..., 2]
    
    valid_mask = (
        (pix_x >= 0) & (pix_x <= W-1) 
        & (pix_y >= 0) & (pix_y <= H-1) 
        & (pix_z > camera.znear) & (pix_z < camera.zfar)
    )  # (N,)
    
    return valid_mask