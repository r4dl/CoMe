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

# This code based on VastGaussian (https://arxiv.org/abs/2402.17427), and modified from GOF (https://github.com/autonomousvision/gaussian-opacity-fields)
# https://github.com/autonomousvision/gaussian-opacity-fields/blob/5245b20e5d11acd6d1ff5af4b890dc2bedd99693/scene/appearance_network.py#L5


import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from functools import partial
from utils.loss_utils import create_window, ssim, SinglePassFusedSSIM


class UpsampleBlock(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(UpsampleBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(num_input_channels // (2 * 2), num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class AppearanceNetwork(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(AppearanceNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(num_input_channels, 256, 3, stride=1, padding=1)
        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)
        self.up4 = UpsampleBlock(32, 16)
        
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # bilinear interpolation
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x
    
class AppearanceNetv2(AppearanceNetwork):
    def __init__(self, num_input_channels, num_output_channels):
        super(AppearanceNetv2, self).__init__(num_input_channels, num_output_channels)
        
        # store the function (do not call it here)
        # use a non-Module attribute to avoid nn.Module checks
        self.activation = torch.exp
        
        # 0-init for the last layers
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # bilinear interpolation
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.activation(x)
        return x

class AppearanceEmbedding(nn.Module):
    def __init__(self, num_views: int, lambda_ssim: float = 0.2):
        super().__init__()
        self.num_view = num_views
        self.l1_loss = nn.L1Loss(reduction='none')
        self.ssim_loss = partial(ssim, size_average=False)
        self.lambda_ssim = lambda_ssim
        self._init_kwargs = {"num_views": num_views, "lambda_ssim": lambda_ssim}

    def forward(self, image, gt_image, view_idx):
        return (1 - self.lambda_ssim) * self.l1_loss(image, gt_image) + self.lambda_ssim * (1 - self.ssim_loss(image, gt_image))

    def capture(self):
        return {
            "class_path": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "init_kwargs": dict(self._init_kwargs),
            "state_dict": self.state_dict(),
        }

    def restore(self, capture):
        if not isinstance(capture, dict):
            raise ValueError("Expected a dict capture with 'state_dict' and 'class_path'.")
        self.load_state_dict(capture.get("state_dict", {}))

    @staticmethod
    def load_from_capture(capture: dict):
        """
        The 'Universal Entry Point'. 
        It uses the class_path to find the right class and initialize it.
        """
        # 1. Parse the path (e.g., "my_project.models.AppearanceEmbedding")
        class_path = capture["class_path"]
        module_path, class_name = class_path.rsplit(".", 1)
        
        # 2. Dynamic Import
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        
        # 3. Initialize & Load Weights
        # Because 'cls' is the class we just imported, this works for any subclass
        instance = cls(**capture["init_kwargs"])
        instance.load_state_dict(capture["state_dict"])
        
        return instance

class VastGaussianAppearanceEmbedding(AppearanceEmbedding):
    def __init__(self, num_views, lambda_ssim: float = 0.2):
        super().__init__(num_views, lambda_ssim)
        
        STD = 1e-4
        
        # TODO: for optimization purposes, might make sense to have this close to 0 mean
        self._appearance_embeddings = nn.Parameter(torch.empty(num_views, 64).cuda())
        self._appearance_embeddings.data.normal_(0, std=STD)
        
        self.appearance_network = AppearanceNetwork(3+64, 3).cuda()
        
    def get_apperance_embedding(self, idx):
        return self._appearance_embeddings[idx]

    def appearance_mapping(self, image, view_idx):
        appearance_embedding = self.get_apperance_embedding(idx=view_idx)
        # center crop the image
        origH, origW = image.shape[1:]
        H = origH // 32 * 32
        W = origW // 32 * 32
        left = origW // 2 - W // 2
        top = origH // 2 - H // 2
        # store last crop window for viewer/masking use
        self._last_crop = (top, left, H, W)
        crop_image = image[:, top:top+H, left:left+W]

        # down sample the image
        crop_image_down = torch.nn.functional.interpolate(
            crop_image[None],
            size=(H // 32, W // 32),
            mode="bilinear",
            align_corners=True
        )[0]

        # TODO: it might make sense to detach this image to avoid backpropagation through the mapping
        crop_image_down = torch.cat(
            [crop_image_down, appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)],
            dim=0
        )[None]
        mapping_image = self.appearance_network(crop_image_down)
        transformed_image = mapping_image * crop_image

        # place the mapped crop back into full resolution without in-place ops
        output = image.clone()
        output[:, top:top+H, left:left+W] = transformed_image
        return output
    
    def forward(self, image, gt_image, view_idx):
        transformed_image = self.appearance_mapping(image, view_idx)
        top, left, H, W = self._last_crop
        
        mult = torch.zeros_like(image)
        mult[:, top:top+H, left:left+W] = 1.0
        
        Ll1 = self.l1_loss(transformed_image, gt_image) * mult
        LSSIM = (1 - self.ssim_loss(image, gt_image))
        
        # TODO: might make sense to not crop the l1-loss but simply lets gradients backprop
        return (1 - self.lambda_ssim) * Ll1 + self.lambda_ssim * LSSIM.mean()


# Decoupled luminance/structure SSIM (SinglePassFusedSSIM) + final mapping:
# 1. detaching the gradients from the image through the mapping
# 2. using reflection padding instead of cropping
# 3. adding a low-frequency grid embedding to the appearance embedding (for vignetting)
class SSIMDecoupledAppearanceEmbedding(VastGaussianAppearanceEmbedding):
    def __init__(self, num_views, lambda_ssim: float = 0.2):
        super().__init__(num_views, lambda_ssim)
        self.register_buffer("window", create_window(11, 3))
        self.ssim_v2 = SinglePassFusedSSIM()
        self.appearance_network = AppearanceNetv2(3 + 64 + 3, 3).cuda()
    
    # this one detaches inbetween the mapping and the loss
    def appearance_mapping(self, image, view_idx):
        appearance_embedding = self.get_apperance_embedding(idx=view_idx)
        # center crop the image
        origH, origW = image.shape[1:]
        
        # compute the padding
        pad_H = (32 - (origH % 32)) % 32
        pad_W = (32 - (origW % 32)) % 32
        
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        
        # pad the image
        image_pad = F.pad(image[None],  (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        padded_H, padded_W = image_pad.shape[2:]
        down_H, down_W = padded_H // 32, padded_W // 32

        # down sample the paddedimage
        image_down = torch.nn.functional.interpolate(
            image_pad,
            size=(down_H, down_W),
            mode="bilinear",
            align_corners=True
        )[0]
        
        # generate spatial coordinates for the padded grid
        y_coords = torch.linspace(-1, 1, steps=down_H, device=image.device)
        x_coords = torch.linspace(-1, 1, steps=down_W, device=image.device)
        y_coords, x_coords = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid_r = torch.sqrt(x_coords**2 + y_coords**2)
        grid_coords = torch.stack([x_coords, y_coords, grid_r], dim=0)

        # TODO: it might make sense to detach this image to avoid backpropagation through the mapping
        image_down_w_embedding = torch.cat(
            [
                image_down.detach(), 
                appearance_embedding[:, None, None].expand(-1, down_H, down_W),
                grid_coords
            ],
            dim=0
        )[None]
        mapping_image = self.appearance_network(image_down_w_embedding)
        transformed_image = mapping_image * image_pad

        return transformed_image[0, :, pad_top : pad_top + origH, pad_left : pad_left + origW]
    
    def forward(self, image, gt_image, view_idx):
        transformed_image = self.appearance_mapping(image, view_idx)
        
        Ll1 = self.l1_loss(transformed_image, gt_image)

        l, cs = self.ssim_v2(gt_image, image, transformed_image)
        LSSIM = (1 - l * cs)
        
        # TODO: might make sense to not crop the l1-loss but simply lets gradients backprop
        return (1 - self.lambda_ssim) * Ll1 + self.lambda_ssim * LSSIM
