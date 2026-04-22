# copy from nerfstudio and 2DGS
import torch
from matplotlib import cm
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from utils.depth_utils import depth_to_normal, central_diff
from torchvision.utils import save_image
from diff_gaussian_rasterization import DebugVisualizationType

def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
    depth,
    accumulation,
    near_plane = 2.0,
    far_plane = 6.0,
    cmap="turbo",
):
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image

def save_points(path_save, pts, colors=None, normals=None, BRG2RGB=False):
    """save points to point cloud using open3d"""
    assert len(pts) > 0
    if colors is not None:
        assert colors.shape[1] == 3
    assert pts.shape[1] == 3

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        # Open3D assumes the color values are of float type and in range [0, 1]
        if np.max(colors) > 1:
            colors = colors / np.max(colors)
        if BRG2RGB:
            colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(path_save, cloud)
    

def colormap(img, cmap='jet'):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    if img.shape[1:] != (H, W):
        img = torch.nn.functional.interpolate(img[None], (W, H), mode='bilinear', align_corners=False)[0]
    return img

def gui_visualize(
    render_cam,
    alpha,
    distortion,
    depth,
    normal,
    render,
    confidence,
    color_variance,
    normal_variance,
    other_args,
):
    indict = lambda key: key in other_args and str(other_args[key]).lower() == "true"
    
    render_alpha = indict("render_alpha")
    render_distortion = indict("render_distortion")
    render_depth = indict("render_depth")
    render_depth_normal = indict("render_depth_normal")
    render_normal = indict("render_normal")
    render_depth_normal_loss = indict("render_depth_normal_loss")
    render_confidence = indict("render_confidence")
    render_color_variance = indict("render_color_variance")
    render_normal_variance = indict("render_normal_variance")
    if render_alpha:
        # magma colormap
        image = alpha.squeeze()
        cmap = matplotlib.colormaps.get_cmap('magma')
        image = torch.tensor(cmap(image.cpu().detach().numpy()), device="cuda").float().permute(-1,0,1)[:3]
        return image
    elif render_distortion:
        image = distortion.squeeze()
        # lets normalize the distortion so we can see it, huh?
        image = image / image.max()
        cmap = matplotlib.colormaps.get_cmap('magma')
        image = torch.tensor(cmap(image.cpu().detach().numpy()), device="cuda").float().permute(-1,0,1)[:3]
        return image
    elif render_depth:
        cmap = matplotlib.colormaps.get_cmap('turbo')
        image = depth.squeeze()
        
        if other_args["manual_normalization"]:
            d_min_max = other_args["depth_min_max"]
            image = torch.clamp((image - d_min_max[0]) / (d_min_max[1] - d_min_max[0]), min=0, max=1.0)
        else:
            image = torch.clamp((image - image.min()) / (image.max() - image.min()), min=0, max=1.0)
        image = torch.tensor(cmap(image.cpu().detach().numpy()), device="cuda").float().permute(-1,0,1)[:3]
        return image
    elif render_depth_normal_loss:
        cmap = matplotlib.colormaps.get_cmap('magma')
        # TODO: update
        depth_normal, _ = depth_to_normal(render_cam, depth)
        depth_normal = depth_normal.permute(2, 0, 1)
        
        render_normal = torch.nn.functional.normalize(normal, p=2, dim=0)
        
        c2w = (render_cam.world_view_transform)
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
        
        normal_error = (1 - (render_normal_world * depth_normal).sum(dim=0))
        
        nabla_I = central_diff(render.permute(1,2,0))
        if indict("image_informed_depthnormal"):
            I = (nabla_I.max() - nabla_I) / (nabla_I.max() - nabla_I.min() + 1e-12)
            normal_error = (I * normal_error)
        return torch.tensor(cmap(normal_error.cpu().detach().numpy()), device="cuda").float().permute(-1,0,1)[:3]
        
    elif render_depth_normal:               
        # depth to normal
        depth_normal, _ = depth_to_normal(render_cam, depth)
        
        # to view space
        w2c = render_cam.world_view_transform[:3,:3]
        depth_normal = depth_normal @ w2c
        
        depth_normal = depth_normal.permute(2, 0, 1)
        
        return (depth_normal + 1) / 2
    elif render_normal:
        normals_normalized = -torch.nn.functional.normalize(normal, p=2, dim=0)
        return (normals_normalized + 1) / 2
    elif render_confidence:
        image = confidence.squeeze()

        # map to [0,1] with the max-value (should be consistent)
        image = image.clamp(min=0, max=5)
        import matplotlib.colors as mcolors
        norm = mcolors.TwoSlopeNorm(vcenter=1, vmin=0, vmax=5)

        cmap = matplotlib.colormaps.get_cmap('RdBu')
        image = torch.tensor(cmap(norm(image.cpu().detach().numpy())), device="cuda").float().permute(-1,0,1)[:3]
        return image
    elif render_color_variance:
        image = color_variance.squeeze()
        # lets normalize the distortion so we can see it, huh?
        image = image / image.max()
        cmap = matplotlib.colormaps.get_cmap('magma')
        image = torch.tensor(cmap(image.cpu().detach().numpy()), device="cuda").float().permute(-1,0,1)[:3]
        return image
    elif render_normal_variance:
        image = normal_variance.squeeze()
        # lets normalize the distortion so we can see it, huh?
        image = image / image.max()
        cmap = matplotlib.colormaps.get_cmap('magma')
        image = torch.tensor(cmap(image.cpu().detach().numpy()), device="cuda").float().permute(-1,0,1)[:3]
        return image
    else:
        return render
    
@torch.no_grad
def export_image(image, path, normalize=True, cmap='magma'):
    if normalize:
        image = torch.clamp((image - image.min()) / (image.max() - image.min() + 1e-9), min=0, max=1.0)
    cmap = matplotlib.colormaps.get_cmap(cmap)
    image = torch.tensor(cmap(image.squeeze().cpu().detach().numpy()), device="cuda").permute(-1,0,1)[:3]
    save_image(image, path)