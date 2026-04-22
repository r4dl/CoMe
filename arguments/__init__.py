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

from argparse import ArgumentParser, Namespace
from diff_gaussian_rasterization import ExtendedSettings, GlobalSortOrder, SortMode
import json
import sys
import os
from distutils.util import strtobool

from enum import IntEnum
from dataclasses import dataclass

class GroupParams:
    pass

class BoundingSetting(IntEnum):
    SIGMA_3 = 0
    SIGMA_333 = 1
    STP = 2
    
    def __str__(self):
        return self.name

@dataclass
class MeshingSettings:
    near : float
    far : float
    texture_mesh : bool
    bounding : BoundingSetting
    mesh_name : str
    near_far_culling : bool
    opacity_cutoff_tetra : float
    load_cells : bool

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        if parser is None:
            return
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t, help=f'Default = {value}')
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t, help=f'Default = {value}')

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.alpha_mask = False
        self.init_type = "sfm"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.convert_SBs_python = False
        self.compute_filter3D_python = False
        self.debug = False
        self.compute_view2gaussian_python = False
        super().__init__(parser, "Pipeline Parameters")

class SplattingSettings():
    
    group_config = None
    group_settings = None
    settings = ExtendedSettings()
    parser = None
    render = False
    
    def __init__(self, parser=None, render=False):
        self.parser = parser
        self.render = render
        if not render:
            self.group_config = parser.add_argument_group("Splatting Config")
            self.group_config.add_argument("--splatting_config", type=str, default='configs/hierarchical.json')
            
        bool_ = lambda x: bool(strtobool(x))

        # TODO: remove EWA Scaling (does not make sense with 3D Eval)
        if parser is not None:
            self.group_settings = parser.add_argument_group("Splatting Settings")
            self.group_settings.add_argument("--sort_mode", type=lambda sortmode: SortMode[sortmode], choices=list(SortMode))
            self.group_settings.add_argument("--sort_order", type=lambda sortorder: GlobalSortOrder[sortorder], choices=list(GlobalSortOrder))
            self.group_settings.add_argument("--tile_4x4", type=int, choices=[64], help='only needed if using sort_mode HIER')
            self.group_settings.add_argument("--tile_2x2", type=int, choices=[8,12,20], help='only needed if using sort_mode HIER')
            self.group_settings.add_argument("--per_pixel", type=int, choices=[1,2,4,8,12,16,20,24], help='if using sort_mode HIER, only {4,8,16} are valid')
            self.group_settings.add_argument("--rect_bounding", type=bool_, choices=[True, False], help="Bound 2D Gaussians with a rectangle instead of a circle")
            self.group_settings.add_argument("--tight_opacity_bounding", type=bool_, choices=[True, False], help="Bound 2D Gaussians by considering their opacity")
            self.group_settings.add_argument("--tile_based_culling", type=bool_, choices=[True, False], help="Cull complete tiles based on opacity")
            self.group_settings.add_argument("--hierarchical_4x4_culling", type=bool_, choices=[True, False], help="Cull Gaussians for 4x4 subtiles, only when using sort_mode HIER")
            self.group_settings.add_argument("--load_balancing", type=bool_, choices=[True, False], help=f"Perform per-tile computations cooperatively (e.g. duplication) (default={ExtendedSettings().load_balancing})")
            self.group_settings.add_argument("--proper_ewa_scaling", type=bool_, choices=[True, False], help=f'Dilation of 2D Gaussians as proposed by Yu et al. ("Mip-Splatting") (default={ExtendedSettings().proper_ewa_scaling})')
            # new arguments
            self.group_settings.add_argument("--exact_depth", type=bool_, choices=[True, False], help=f'Exact Depth computation (better 0.5 level set approximation) (default={ExtendedSettings().exact_depth})')
            self.group_settings.add_argument("--detach_alpha", type=bool_, choices=[True, False], help=f'Detach Alpha Gradient for Depth Distortion (not working currently) (default={ExtendedSettings().detach_alpha})')
            self.group_settings.add_argument("--far_plane", type=float, help=f'Far plane for distortion computations, 2DGS/GOF use 100.0 (default={ExtendedSettings().far_plane})')
            self.group_settings.add_argument("--detach_alpha_extent", type=bool_, choices=[True, False], help=f'Exact Depth computation (better 0.5 level set approximation) (default={ExtendedSettings().detach_alpha_extent})')
            self.group_settings.add_argument("--include_alpha", type=bool_, choices=[True, False], help=f'Exact Depth computation (better 0.5 level set approximation) (default={ExtendedSettings().include_alpha})')

    def get_settings(self, arguments):
        # get valid choices from configargparse
        config = None
        
        # load default dict, if passed
        if self.render:
            cmdlne_string = sys.argv[1:]
            args_cmdline = self.parser.parse_args(cmdlne_string)
            cfgfilepath = os.path.join(args_cmdline.model_path, "config.json")
            print("Looking for splatting config file in", cfgfilepath)
            if os.path.exists(cfgfilepath):
                print("Config file found: {}".format(cfgfilepath))
                self.settings = ExtendedSettings.from_json(cfgfilepath)
            else:
                print("No config file found, assuming default values")
        else:
            for arg in vars(arguments).items():
                if any([arg[0] in z.option_strings[0] for z in self.group_config._group_actions]):
                    # json passed, load it
                    if arg[1] is None:
                        continue
                    with open(arg[1], 'r') as json_file:
                        config = json.load(json_file)
                        self.settings = ExtendedSettings.from_dict(config)
                    
        for arg in vars(arguments).items():
            if any([arg[0] in z.option_strings[0] for z in self.group_settings._group_actions]):
                # pass any options which were not given
                if arg[1] is None:
                    continue
                self.settings.set_value(arg[0], arg[1])
                
        return self.settings
    
    def get_settings_from_path(self, path):
        # unmount the path until we get a config file we can parse
        path_to_start = path[0]
        
        while path_to_start:
            config_path = os.path.join(path_to_start, "config.json")
            if os.path.exists(config_path):
                return ExtendedSettings.from_json(config_path)
            path_to_start = os.path.dirname(path_to_start)
                
        import warnings
        warnings.warn("Did not find a splatting settings file; default initialization")
        return ExtendedSettings()

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.confidence_lr = 0.00025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        
        super().__init__(parser, "Optimization Parameters")

class MeshingParams(ParamGroup):
    """
    Currently, we do not need these for rendering at all, this is just a timestamp for training
    """
    def __init__(self, parser):
        # appearance embedding
        self.use_vastgaussian_appearance = False
        self.use_ssimdecoupled_appearance = False
        self.appearance_lr_init = 0.001
        self.appearance_lr_final = 0.001

        # color confidence (default settings)
        self.color_confidence = False
        self.color_confidence_max = 0.075
        self.color_confidence_from_iter = 500

        # custom variance losses
        self.lambda_variance = 0.0
        self.variance_from_iter = 15000
        self.lambda_normal_variance = 0.0
        self.normal_variance_from_iter = 15000

        # distortion/depthnormal
        self.lambda_distortion = 1000.
        self.lambda_opacity_field = 0.004
        self.lambda_extent = 0.1
        self.lambda_depth_normal = 0.05
        self.distortion_from_iter =  15000
        self.depth_normal_from_iter = 15000
        
        # densification
        self.abs_grad_for_densification = True
        self.clone_with_sampling = True
        self.prune_threshold = 0.05
        self.opacity_decay = 0.0
        # TODO: make this similar to what we have for appearance
        self.use_msv2_simplification = False
        # normal regularization
        self.lambda_smoothness = 0.01
        # MCMC
        self.scale_reg = 0.0
        self.opacity_reg = 0.0
        self.min_scale_reg = 0.0
        self.cap_max = -1
        self.noise_lr = 5e5
        self.min_opacity = 1./255.
        
        super().__init__(parser, "Meshing Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    # try to load the cfg_args
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print(f"Config file not found at {cfgfilepath}")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
