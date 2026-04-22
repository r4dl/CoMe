import argparse
from math import exp
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from decoupled_fused_ssim import fused_ssim


LossWeights = Tuple[float, float, float]
Shape4D = Tuple[int, int, int, int]


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma**2)) for x in range(window_size)],
        dtype=torch.float32,
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w1 = gaussian(window_size, 1.5).unsqueeze(1)
    w2 = w1.mm(w1.t()).unsqueeze(0).unsqueeze(0)
    return w2.expand(channel, 1, window_size, window_size).contiguous().to(device=device, dtype=dtype)


def ssim_v2_reference(
    img1: torch.Tensor,
    img2: torch.Tensor,
    img3: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu3 = F.conv2d(img3, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu3_sq = mu3.pow(2)
    mu1_mu2 = mu1 * mu2
    mu1_mu3 = mu1 * mu3

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    eps = 1e-12
    sigma1_sq = torch.clamp(sigma1_sq, min=eps)
    sigma2_sq = torch.clamp(sigma2_sq, min=eps)

    c1 = 0.01**2
    c2 = 0.03**2
    luminance_map = (2 * mu1_mu3 + c1) / (mu1_sq + mu3_sq + c1)
    contrast_structure_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    return luminance_map, contrast_structure_map


def decoupled_loss(luminance_map: torch.Tensor, contrast_structure_map: torch.Tensor, w: LossWeights) -> torch.Tensor:
    a, b, c = w
    return (a * luminance_map + b * contrast_structure_map + c * luminance_map * contrast_structure_map).mean()


def assert_close(name: str, a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> None:
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        abs_err = (a - b).abs().max().item()
        rel_den = b.abs().clamp_min(1e-12)
        rel_err = ((a - b).abs() / rel_den).max().item()
        raise AssertionError(f"{name} mismatch: max_abs_err={abs_err:.6e}, max_rel_err={rel_err:.6e}")


def run_forward_parity(
    device: torch.device,
    seeds: Sequence[int],
    shapes: Sequence[Shape4D],
    rtol: float,
    atol: float,
) -> None:
    print("[1/3] Forward parity checks")
    for shape in shapes:
        b, ch, h, w = shape
        window = create_window(11, ch, device, torch.float32)
        for seed in seeds:
            torch.manual_seed(seed)
            img1 = torch.rand(b, ch, h, w, device=device)
            img2 = torch.rand(b, ch, h, w, device=device)
            img3 = torch.rand(b, ch, h, w, device=device)

            lum_ref, cs_ref = ssim_v2_reference(img1, img2, img3, window, 11, ch)
            lum_fused, cs_fused = fused_ssim(img1, img2, img3, train=False)
            assert_close(f"forward/luminance/{shape}/seed={seed}", lum_fused, lum_ref, rtol, atol)
            assert_close(f"forward/contrast_structure/{shape}/seed={seed}", cs_fused, cs_ref, rtol, atol)

            img1_nc = img1.transpose(2, 3)
            img2_nc = img2.transpose(2, 3)
            img3_nc = img3.transpose(2, 3)
            lum_ref_nc, cs_ref_nc = ssim_v2_reference(img1_nc, img2_nc, img3_nc, window, 11, ch)
            lum_fused_nc, cs_fused_nc = fused_ssim(img1_nc, img2_nc, img3_nc, train=False)
            assert_close(f"forward-nc/luminance/{shape}/seed={seed}", lum_fused_nc, lum_ref_nc, rtol, atol)
            assert_close(
                f"forward-nc/contrast_structure/{shape}/seed={seed}", cs_fused_nc, cs_ref_nc, rtol, atol
            )
    print("  OK")


def run_grad_parity(
    device: torch.device,
    seeds: Sequence[int],
    shapes: Sequence[Shape4D],
    weight_sets: Iterable[LossWeights],
    rtol: float,
    atol: float,
) -> None:
    print("[2/3] Gradient parity checks (d loss / d img1)")
    for shape in shapes:
        b, ch, h, w = shape
        window = create_window(11, ch, device, torch.float32)
        for seed in seeds:
            torch.manual_seed(seed)
            img1_base = torch.rand(b, ch, h, w, device=device)
            img2_base = torch.rand(b, ch, h, w, device=device)
            img3_base = torch.rand(b, ch, h, w, device=device)

            for weights in weight_sets:
                img1_ref = img1_base.detach().clone().requires_grad_(True)
                img2_ref = img2_base.detach().clone().requires_grad_(True)
                img3_ref = img3_base.detach().clone().requires_grad_(True)
                lum_ref, cs_ref = ssim_v2_reference(img1_ref, img2_ref, img3_ref, window, 11, ch)
                loss_ref = decoupled_loss(lum_ref, cs_ref, weights)
                g1_ref, g2_ref, g3_ref = torch.autograd.grad(
                    loss_ref, (img1_ref, img2_ref, img3_ref), allow_unused=True
                )

                img1_fused = img1_base.detach().clone().requires_grad_(True)
                img2_fused = img2_base.detach().clone().requires_grad_(True)
                img3_fused = img3_base.detach().clone().requires_grad_(True)
                lum_fused, cs_fused = fused_ssim(img1_fused, img2_fused, img3_fused, train=True)
                loss_fused = decoupled_loss(lum_fused, cs_fused, weights)
                g1_fused, g2_fused, g3_fused = torch.autograd.grad(
                    loss_fused, (img1_fused, img2_fused, img3_fused), allow_unused=True
                )

                assert_close(
                    f"grad-img1/{shape}/seed={seed}/weights={weights}",
                    g1_fused,
                    g1_ref,
                    rtol,
                    atol,
                )

                if g2_fused is not None:
                    assert_close(
                        f"grad-img2/{shape}/seed={seed}/weights={weights}",
                        g2_fused,
                        g2_ref,
                        rtol,
                        atol,
                    )
                if g3_fused is not None:
                    assert_close(
                        f"grad-img3/{shape}/seed={seed}/weights={weights}",
                        g3_fused,
                        g3_ref,
                        rtol,
                        atol,
                    )
    print("  OK")


def run_finite_difference_check(
    device: torch.device,
    seed: int,
    eps: float,
    sample_count: int,
    fd_rtol: float,
    fd_atol: float,
) -> None:
    print("[3/3] Finite-difference gradient checks (sampled pixels)")
    torch.manual_seed(seed)

    shape = (1, 1, 21, 23)
    _, ch, _, _ = shape
    img1 = torch.rand(*shape, device=device, requires_grad=True)
    img2 = torch.rand(*shape, device=device)
    img3 = torch.rand(*shape, device=device)
    weights: LossWeights = (0.2, 0.4, 0.8)

    lum, cs = fused_ssim(img1, img2, img3, train=True)
    loss = decoupled_loss(lum, cs, weights)
    (g_autograd,) = torch.autograd.grad(loss, (img1,))

    flat = img1.detach().flatten()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 17)
    sample_idx = torch.randperm(flat.numel(), generator=gen, device=device)[:sample_count]

    for i in sample_idx.tolist():
        x_plus = img1.detach().clone()
        x_minus = img1.detach().clone()
        x_plus.view(-1)[i] += eps
        x_minus.view(-1)[i] -= eps

        lum_p, cs_p = fused_ssim(x_plus, img2, img3, train=False)
        lum_m, cs_m = fused_ssim(x_minus, img2, img3, train=False)
        l_plus = decoupled_loss(lum_p, cs_p, weights)
        l_minus = decoupled_loss(lum_m, cs_m, weights)
        g_num = (l_plus - l_minus) / (2.0 * eps)
        g_ana = g_autograd.view(-1)[i]

        if not torch.allclose(g_num, g_ana, rtol=fd_rtol, atol=fd_atol):
            raise AssertionError(
                f"finite-diff mismatch at flat_idx={i}: "
                f"num={g_num.item():.6e}, ana={g_ana.item():.6e}, "
                f"abs_err={abs((g_num - g_ana).item()):.6e}"
            )
    print("  OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive decoupled fused SSIM checks")
    parser.add_argument("--rtol", type=float, default=1e-3, help="rtol for forward and analytic grad parity")
    parser.add_argument("--atol", type=float, default=1e-5, help="atol for forward and analytic grad parity")
    parser.add_argument("--fd-rtol", type=float, default=2e-2, help="rtol for finite-difference checks")
    parser.add_argument("--fd-atol", type=float, default=2e-3, help="atol for finite-difference checks")
    parser.add_argument("--fd-eps", type=float, default=1e-3, help="finite-difference epsilon")
    parser.add_argument("--fd-samples", type=int, default=32, help="number of sampled pixels for FD checks")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="random seeds to test")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA because decoupled-fused-ssim CUDA extension is being validated.")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")
    shapes: List[Shape4D] = [
        (1, 1, 17, 19),
        (1, 3, 31, 29),
        (1, 3, 233, 233),
    ]
    weight_sets: List[LossWeights] = [
        (1.0, 0.0, 0.0),  # luminance-only
        (0.0, 1.0, 0.0),  # contrast-structure-only
        (0.3, 0.7, 0.0),  # weighted additive
        (0.2, 0.4, 0.8),  # mixed additive+product
    ]

    run_forward_parity(device, args.seeds, shapes, args.rtol, args.atol)
    run_grad_parity(device, args.seeds, shapes, weight_sets, args.rtol, args.atol)
    run_finite_difference_check(
        device,
        seed=args.seeds[0],
        eps=args.fd_eps,
        sample_count=args.fd_samples,
        fd_rtol=args.fd_rtol,
        fd_atol=args.fd_atol,
    )
    print("All decoupled fused SSIM checks passed.")


if __name__ == "__main__":
    main()
