import argparse
import os
import sys
from typing import Dict, Tuple

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from arguments import ModelParams, PipelineParams, SplattingSettings, get_combined_args  # noqa: E402
from diff_gaussian_rasterization import ExtendedSettings  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.appearance_network import AppearanceEmbedding  # noqa: E402


def load_appearance_embedding(model_path: str, iteration: int) -> torch.nn.Module:
    if iteration < 0:
        point_cloud_dir = os.path.join(model_path, "point_cloud")
        iterations = [d for d in os.listdir(point_cloud_dir) if d.startswith("iteration_")]
        if not iterations:
            raise FileNotFoundError(f"No iteration folders in {point_cloud_dir}")
        iteration = max(int(d.split("_")[-1]) for d in iterations)
    embed_path = os.path.join(
        model_path, "point_cloud", f"iteration_{iteration}", "appearance_embedding.pth"
    )
    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"Missing appearance embedding: {embed_path}")
    params = torch.load(embed_path, weights_only=True)
    return AppearanceEmbedding.load_from_capture(params)


def compute_loss_and_grads(
    appearance_embedding: torch.nn.Module,
    image: torch.Tensor,
    gt_image: torch.Tensor,
    view_idx: int,
    use_fused: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if not hasattr(appearance_embedding, "appearance_mapping"):
        raise RuntimeError("Appearance embedding does not expose appearance_mapping.")
    if not hasattr(appearance_embedding, "ssim_v2"):
        raise RuntimeError("Appearance embedding does not expose ssim_v2.")
    if not hasattr(appearance_embedding, "l1_loss"):
        raise RuntimeError("Appearance embedding does not expose l1_loss.")

    from utils.loss_utils import CompiledSSIMV2
    ssim_v2 = CompiledSSIMV2(11, 3).to(device=image.device).train()
    from decoupled_fused_ssim import fused_ssim

    appearance_embedding.zero_grad(set_to_none=True)
    if image.grad is not None:
        image.grad.zero_()

    image_from_app_embed = appearance_embedding.appearance_mapping(image, view_idx)
    mult = torch.ones_like(image)

    Ll1 = appearance_embedding.l1_loss(image_from_app_embed, gt_image) * mult

    ssim_module = appearance_embedding.ssim_v2
    window = getattr(ssim_module, "window", None)
    window_size = getattr(ssim_module, "window_size", 11)
    channel = getattr(ssim_module, "channel", image.shape[0])
    if window is None:
        raise RuntimeError("ssim_v2 module does not expose window buffer.")

    if use_fused:
        # fused_ssim CUDA path expects batched tensors (N,C,H,W).
        l, cs = fused_ssim(
            gt_image=gt_image.unsqueeze(0),
            gs_image=image.unsqueeze(0),
            gs_image_mapped=image_from_app_embed.unsqueeze(0),
        )
        l = l.squeeze(0)
        cs = cs.squeeze(0)
    else:
        _, cs = ssim_v2(image, gt_image)
        l, _ = ssim_v2(image_from_app_embed, gt_image)

    l = l * mult
    LSSIM = (1 - l * cs)
    loss = (1 - appearance_embedding.lambda_ssim) * Ll1 + appearance_embedding.lambda_ssim * LSSIM
    loss = loss.mean()
    loss.backward()

    img_grad = image.grad.detach().clone()
    param_grads: Dict[str, torch.Tensor] = {}
    for name, param in appearance_embedding.named_parameters():
        if param.grad is None:
            continue
        param_grads[name] = param.grad.detach().clone()
    return loss.detach(), img_grad, param_grads


def compare_grads(
    grads_a: Dict[str, torch.Tensor],
    grads_b: Dict[str, torch.Tensor],
) -> Tuple[float, str]:
    max_diff = 0.0
    max_name = ""
    for name, grad_a in grads_a.items():
        grad_b = grads_b.get(name)
        if grad_b is None:
            continue
        diff = (grad_a - grad_b).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            max_name = name
    return max_diff, max_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare manual vs fused SSIM on a real scene.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = get_combined_args(parser)

    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA.")

    torch.manual_seed(args.seed)
    splat_args: ExtendedSettings = ss.get_settings(args)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, skip_test=True, skip_train=False)
    views = scene.getTrainCameras()
    try:
        appearance_embedding = load_appearance_embedding(dataset.model_path, scene.loaded_iter)
    except Exception:
        from scene.appearance_network import SSIMDecoupledAppearanceEmbedding

        appearance_embedding = SSIMDecoupledAppearanceEmbedding(num_views=len(views)).to(
            gaussians._opacity.device
        ).train()
    appearance_embedding = appearance_embedding.to(gaussians._opacity.device).train()
    

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    
    if args.max_views > 0:
        views = views[: args.max_views]

    for i, view in enumerate(views):
        render_pkg = render(view, gaussians, pipe, background, splat_args=splat_args)
        rendering = render_pkg["render"]
        image = rendering[:3].detach().clone().requires_grad_(True)
        gt_image = view.original_image.cuda()
        view_idx = getattr(view, "idx", getattr(view, "uid", 0))

        loss_manual, img_grad_manual, param_grads_manual = compute_loss_and_grads(
            appearance_embedding, image, gt_image, view_idx, use_fused=False
        )
        loss_fused, img_grad_fused, param_grads_fused = compute_loss_and_grads(
            appearance_embedding, image, gt_image, view_idx, use_fused=True
        )

        loss_diff = (loss_manual - loss_fused).abs().item()
        img_grad_diff = (img_grad_manual - img_grad_fused).abs().max().item()
        param_grad_diff, param_name = compare_grads(param_grads_manual, param_grads_fused)

        print(
            f"[view {i}] loss_diff={loss_diff:.6e} "
            f"img_grad_max_diff={img_grad_diff:.6e} "
            f"param_grad_max_diff={param_grad_diff:.6e} ({param_name})"
        )


if __name__ == "__main__":
    main()
