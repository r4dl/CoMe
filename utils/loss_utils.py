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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from decoupled_fused_ssim import fused_ssim as decoupled_fused_ssim

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map

def _ssim_original(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    eps = 1e-12
    sigma1_sq = torch.clamp(sigma1_sq, min=eps)
    sigma2_sq = torch.clamp(sigma2_sq, min=eps)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2/2

    luminance_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    structure_map = (sigma12 + C3) / (torch.sqrt(sigma1_sq ) * torch.sqrt(sigma2_sq) + C3)

    return luminance_map, contrast_map, structure_map

def _ssim_v2(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    eps = 1e-12
    sigma1_sq = torch.clamp(sigma1_sq, min=eps)
    sigma2_sq = torch.clamp(sigma2_sq, min=eps)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2/2

    luminance_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast_structure_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    return luminance_map, contrast_structure_map

class CompiledSSIMV2(nn.Module):
    # TODO: fuse
    def __init__(self, window_size, channel, **compile_kwargs):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer(
            "window",
            create_window(window_size, channel),
        )

        def _fn(img1, img2):
            return _ssim_v2(img1, img2, self.window, self.window_size, self.channel)

        self._compiled = torch.compile(_fn, **compile_kwargs)

    def forward(self, img1, img2):
        return self._compiled(img1, img2)

class SinglePassFusedSSIM(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, gt_image, gs_image, gs_image_mapped):
        gt_image = gt_image.unsqueeze(0)
        gs_image = gs_image.unsqueeze(0)
        gs_image_mapped = gs_image_mapped.unsqueeze(0)
        lum_map, con_map = decoupled_fused_ssim(gt_image, gs_image, gs_image_mapped)
        return lum_map.squeeze(0), con_map.squeeze(0)

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1 = torch.rand(1, 3, 233, 233, device=device)
    img2 = torch.rand(1, 3, 233, 233, device=device)
    window = create_window(11, 3).to(device=device, dtype=img1.dtype)

    ssim_map = _ssim(img1, img2, window, 11, 3, size_average=False)
    dissim_ssim = 1.0 - ssim_map.mean()

    lum_o, con_o, str_o = _ssim_original(img1, img2, window, 11, 3)
    ssim_map_original = lum_o * con_o * str_o
    dissim_original = 1.0 - ssim_map_original.mean()

    lum_v2, con_str_v2 = _ssim_v2(img1, img2, window, 11, 3)
    ssim_map_v2 = lum_v2 * con_str_v2
    dissim_v2 = 1.0 - ssim_map_v2.mean()

    F_SSIM = SinglePassFusedSSIM()
    luminance_map, contrast_structure_map = F_SSIM(img1, img2, img2)
    dissim_fused = 1.0 - (luminance_map * contrast_structure_map).mean()

    print("D-SSIM of ssim():", dissim_ssim.item())
    print("D-SSIM of ssim_original():", dissim_original.item())
    print("D-SSIM of ssim_v2():", dissim_v2.item())
    print("D-SSIM of decoupled_fused_ssim():", dissim_fused.item())
    
    assert torch.allclose(dissim_v2, dissim_fused, rtol=1e-3, atol=1e-5)
    assert torch.allclose(luminance_map, lum_v2, rtol=1e-3, atol=1e-5)
    assert torch.allclose(contrast_structure_map, con_str_v2, rtol=1e-3, atol=1e-5)
    
    # Build two independent graphs for decoupled SSIM:
    # luminance uses gs_image_mapped, contrast-structure uses gs_image.
    img3 = img2.detach().clone()
    img1_ref = img1.detach().clone().requires_grad_(True)
    img2_ref = img2.detach().clone().requires_grad_(True)
    img3_ref = img3.detach().clone().requires_grad_(True)

    lum_ref, _ = _ssim_v2(img1_ref, img3_ref, window, 11, 3)
    _, cs_ref = _ssim_v2(img1_ref, img2_ref, window, 11, 3)
    loss_ref = 1.0 - (lum_ref * cs_ref).mean()
    g1_ref, g2_ref, g3_ref = torch.autograd.grad(loss_ref, (img1_ref, img2_ref, img3_ref), allow_unused=True)

    img1_fused = img1.detach().clone().requires_grad_(True)
    img2_fused = img2.detach().clone().requires_grad_(True)
    img3_fused = img3.detach().clone().requires_grad_(True)

    lum_fused, cs_fused = F_SSIM(img1_fused, img2_fused, img3_fused)
    loss_fused = 1.0 - (lum_fused * cs_fused).mean()
    g1_fused, g2_fused, g3_fused = torch.autograd.grad(
        loss_fused,
        (img1_fused, img2_fused, img3_fused),
        allow_unused=True,
    )

    assert torch.allclose(g1_ref, g1_fused, rtol=1e-3, atol=1e-5)
    if g2_fused is not None:
        assert torch.allclose(g2_ref, g2_fused, rtol=1e-3, atol=1e-5)
    if g3_fused is not None:
        assert torch.allclose(g3_ref, g3_fused, rtol=1e-3, atol=1e-5)

    print(
        "GRADIENTS MATCH:",
        torch.allclose(g1_ref, g1_fused, rtol=1e-3, atol=1e-5),
    )

    try:
        from fused_ssim import fused_ssim
        fused_ssim_map = fused_ssim(img1, img2)
        print("D-SSIM of fused_ssim():", (1 - fused_ssim_map).mean().item())
    except Exception as e:
        pass