import re
import traceback
import numpy as np
from PIL import Image
import matplotlib
import torch
import torch.nn

from splatviz_utils.dict_utils import EasyDict
from utils.loss_utils import create_window, _ssim, _ssim_original


class Renderer:
    def __init__(self):
        self._device = torch.device("cuda")
        self._pinned_bufs = dict()
        self._is_timing = False
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def render(self, **args):
        self._is_timing = True
        self._start_event.record(torch.cuda.current_stream(self._device))
        res = EasyDict()
        try:
            with torch.no_grad():
                self._render_impl(res, **args)
        except Exception as e:
            res.error = "".join(traceback.format_exception(e))
            res.error += str(e)
        self._end_event.record(torch.cuda.current_stream(self._device))
        if "image" in res:
            def _load_gt_tensor(target_h, target_w, device):
                image_path = args["current_camera"]["img_path"]
                gt_img = Image.open(image_path).convert("RGB")
                src_w, src_h = gt_img.size
                scale = min(target_w / src_w, target_h / src_h)
                new_w = max(1, int(round(src_w * scale)))
                new_h = max(1, int(round(src_h * scale)))
                gt_img = gt_img.resize((new_w, new_h), Image.BILINEAR)
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                mask = np.zeros((target_h, target_w), dtype=np.uint8)
                offset_x = (target_w - new_w) // 2
                offset_y = (target_h - new_h) // 2
                canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = np.asarray(gt_img)
                mask[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = 1
                gt_tensor = torch.from_numpy(canvas).to(device=device)
                mask_tensor = torch.from_numpy(mask).to(device=device)
                return gt_tensor, mask_tensor

            show_loss = any(
                args.get(key)
                for key in (
                    "show_l1_loss",
                    "show_dssim_loss",
                    "show_ssim_luminance_loss",
                    "show_ssim_contrast_loss",
                    "show_ssim_structure_loss",
                )
            )

            if args.get("show_gt_image") or show_loss:
                try:
                    target_h, target_w = res.image.shape[:2]
                    gt_tensor, valid_mask = _load_gt_tensor(target_h, target_w, res.image.device)
                    if show_loss:
                        render = res.image.to(torch.float32) / 255.0
                        render = render.permute(2, 0, 1).unsqueeze(0)
                        gt = gt_tensor.to(torch.float32) / 255.0
                        gt = gt.permute(2, 0, 1).unsqueeze(0)

                        channel = render.shape[1]
                        window_size = 11
                        window = create_window(window_size, channel).to(device=render.device, dtype=render.dtype)

                        if args.get("show_l1_loss"):
                            loss_map = (render - gt).abs().mean(dim=1, keepdim=False)
                        elif args.get("show_dssim_loss"):
                            ssim_map = _ssim(render, gt, window, window_size, channel, size_average=False)
                            loss_map = (1.0 - ssim_map.mean(dim=1, keepdim=False)) * 0.5
                        else:
                            lum, con, struct = _ssim_original(render, gt, window, window_size, channel)
                            if args.get("show_ssim_luminance_loss"):
                                loss_map = 1.0 - lum.mean(dim=1, keepdim=False)
                            elif args.get("show_ssim_contrast_loss"):
                                loss_map = 1.0 - con.mean(dim=1, keepdim=False)
                            else:
                                loss_map = 1.0 - struct.mean(dim=1, keepdim=False)

                        loss_map = loss_map.clamp(min=0)
                        valid = valid_mask.to(dtype=loss_map.dtype).unsqueeze(0)
                        loss_map = loss_map * valid
                        loss_mean = loss_map.mean().item()
                        loss_max = loss_map.max().item()
                        res.loss_stats = {"mean": loss_mean, "max": loss_max}
                        max_val = loss_map.max().clamp(min=1e-6)
                        loss_map = (loss_map / max_val).squeeze(0)
                        loss_np = loss_map.detach().cpu().numpy()
                        cmap = matplotlib.colormaps.get_cmap("turbo")
                        color = cmap(loss_np)[..., :3]
                        color_uint8 = (color * 255).astype(np.uint8)
                        res.image = torch.from_numpy(color_uint8).to(device=res.image.device)
                    else:
                        res.image = gt_tensor
                except Exception as e:
                    print(f"Error loading GT image: {e}")

            res.image = res.image.cpu().detach().numpy()

        if "stats" in res:
            res.stats = res.stats.cpu().detach().numpy()
        if "error" in res:
            res.error = str(res.error)
        if self._is_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    @staticmethod
    def sanitize_command(edit_text):
        command = re.sub(";+", ";", edit_text.replace("\n", ";"))
        while command.startswith(";"):
            command = command[1:]
        return command

    def _render_impl(self, **args):
        raise NotImplementedError

    def _load_model(self, path):
        raise NotImplementedError

    @staticmethod
    def _return_image(
        images,
        res: dict,
        normalize: bool,
        use_splitscreen: bool = False,
        highlight_border: bool = False,
        on_top: bool = False,
    ) -> None:

        if not isinstance(images, list):
            images = [images]

        if use_splitscreen:
            img = torch.zeros_like(images[0])
            split_size = img.shape[-1] // len(images)
            offset = 0
            for i in range(len(images)):
                img[..., offset : offset + split_size] = images[i][..., offset : offset + split_size]
                offset += split_size
                if highlight_border and i != len(images) - 1:
                    img[..., offset - 1 : offset] = 1

        elif on_top:
            mask = torch.mean(images[1], dim=0)
            img = images[0] * (1 - mask) + images[1] * mask
        else:
            img = torch.concat(images, dim=2)

        res.stats = torch.stack([img.mean(), img.std()])

        # Scale and convert to uint8.
        if normalize:
            img = img / img.norm(float("inf"), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
        img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        res.image = img
