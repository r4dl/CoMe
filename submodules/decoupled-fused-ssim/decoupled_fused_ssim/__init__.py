from typing import NamedTuple
import torch.nn as nn
import torch

if torch.cuda.is_available():
    from decoupled_fused_ssim_cuda import fusedssim, fusedssim_backward, fusedssim3d, fusedssim_backward3d
    is_3D_supported = True
elif torch.mps.is_available():
    from decoupled_fused_ssim_mps import fusedssim, fusedssim_backward
    is_3D_supported = False
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    from decoupled_fused_ssim_xpu import fusedssim, fusedssim_backward
    is_3D_supported = False


allowed_padding = ["same", "valid"]

    
class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, gt_image, gs_image, gs_image_mapped=None, padding="same", train=True, spatial_dims=2):
        if gs_image_mapped is None:
            raise ValueError("gs_image_mapped must be provided for decoupled fused SSIM.")
        ctx.is_cuda = gt_image.is_cuda
        if spatial_dims == 2:
            if gt_image.is_cuda:
                luminance_map, contrast_structure_map, dl_dmu1, dl_dmu3, dcs_dmu1, dcs_dmu2, dcs_dsigma1_sq, dcs_dsigma12 = fusedssim(
                    C1, C2, gt_image, gs_image, gs_image_mapped, train
                )
            else:
                luminance_map, contrast_structure_map, dl_dmu1, dcs_dmu1, dcs_dsigma1_sq, dcs_dsigma12 = fusedssim(
                    C1, C2, gt_image, gs_image, train
                )
        elif spatial_dims == 3:
            luminance_map, contrast_structure_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim3d(
                C1, C2, gt_image, gs_image, train
            )

        if spatial_dims == 2:
            if gt_image.is_cuda:
                ctx.save_for_backward(
                    gt_image.detach(),
                    gs_image,
                    gs_image_mapped,
                    dl_dmu1,
                    dl_dmu3,
                    dcs_dmu1,
                    dcs_dmu2,
                    dcs_dsigma1_sq,
                    dcs_dsigma12,
                )
            else:
                ctx.save_for_backward(gt_image.detach(), gs_image, dl_dmu1, dcs_dmu1, dcs_dsigma1_sq, dcs_dsigma12)
        else:
            ctx.save_for_backward(gt_image.detach(), gs_image, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding
        ctx.spatial_dims = spatial_dims

        return luminance_map, contrast_structure_map

    @staticmethod
    def backward(ctx, dL_dluminance_map, dL_dcontrast_structure_map):
        saved_tensors = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding

        if ctx.spatial_dims == 2:
            if ctx.is_cuda:
                gt_image, gs_image, gs_image_mapped, dl_dmu1, dl_dmu3, dcs_dmu1, dcs_dmu2, dcs_dsigma1_sq, dcs_dsigma12 = saved_tensors

                # Skip expensive gradient computations when not needed by passing empty tensors.
                need_gt = ctx.needs_input_grad[2]
                need_gs = ctx.needs_input_grad[3]
                need_gs_mapped = ctx.needs_input_grad[4]
                empty = dl_dmu1.new_empty((0,))
                if not need_gt:  # gt_image
                    dl_dmu1 = empty
                    dcs_dmu1 = empty
                if not need_gs:  # gs_image
                    dcs_dmu2 = empty
                if not need_gs_mapped:  # gs_image_mapped
                    dl_dmu3 = empty

                grad_gt, grad_gs, grad_gs_mapped = fusedssim_backward(
                    C1, C2, gt_image, gs_image, gs_image_mapped,
                    dL_dluminance_map, dL_dcontrast_structure_map,
                    dl_dmu1, dl_dmu3, dcs_dmu1, dcs_dmu2, dcs_dsigma1_sq, dcs_dsigma12
                )
                if not need_gt:
                    grad_gt = None
                if not need_gs:
                    grad_gs = None
                if not need_gs_mapped:
                    grad_gs_mapped = None
            else:
                gt_image, gs_image, dl_dmu1, dcs_dmu1, dcs_dsigma1_sq, dcs_dsigma12 = saved_tensors
                grad_gt = fusedssim_backward(
                    C1, C2, gt_image, gs_image,
                    dL_dluminance_map, dL_dcontrast_structure_map,
                    dl_dmu1, dcs_dmu1, dcs_dsigma1_sq, dcs_dsigma12
                )
                grad_gs = None
                grad_gs_mapped = None
        elif ctx.spatial_dims == 3:
            gt_image, gs_image, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = saved_tensors
            grad_gt = fusedssim_backward3d(
                C1, C2, gt_image, gs_image, dL_dluminance_map, dL_dcontrast_structure_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12
            )
            grad_gs = None
            grad_gs_mapped = None

        return None, None, grad_gt, grad_gs, grad_gs_mapped, None, None, None

def fused_ssim(gt_image, gs_image, gs_image_mapped, padding="same", train=True):
    """
    Decoupled SSIM maps in one call.

    Inputs:
      - gt_image: static ground-truth image (anchor image in both branches)
      - gs_image: raw GS render (used in contrast-structure branch)
      - gs_image_mapped: appearance-mapped GS render (used in luminance branch)

    Returns:
      - luminance_map = l(gt_image, gs_image_mapped)
      - contrast_structure_map = cs(gt_image, gs_image)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    gt_image = gt_image.contiguous()
    luminance_map, contrast_structure_map = FusedSSIMMap.apply(
        C1, C2, gt_image, gs_image, gs_image_mapped, padding, train, 2
    )
    return luminance_map, contrast_structure_map

