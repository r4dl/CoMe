import torch
import math

def params_to_sb_features(sb_params):
    """Organizes raw SB parameters into structured components"""
    # sb_params shape: [N, 15]
    return {
        'c0': sb_params[:, 0:3],               # Diffuse color [N, 3]
        'lights': [
            {                                   # First light
                'theta': sb_params[:, 3],       # Polar angle
                'phi': sb_params[:, 4],         # Azimuthal angle
                'b': sb_params[:, 5],           # Sharpness control
                'c': sb_params[:, 6:9]          # Specular color [N, 3]
            },
            {                                   # Second light
                'theta': sb_params[:, 9],
                'phi': sb_params[:, 10],
                'b': sb_params[:, 11],
                'c': sb_params[:, 12:15]
            }
        ]
    }

def spherical_to_cartesian(theta, phi):
    """Convert spherical coordinates to unit vector"""
    sin_theta = torch.sin(theta)
    return torch.stack([
        sin_theta * torch.cos(phi),
        sin_theta * torch.sin(phi),
        torch.cos(theta)
    ], dim=-1)

def sb_eval(tensor, view_dir):
    R = spherical_to_cartesian(tensor[:,0], tensor[:,1])  # [N, 3]
        
    # Compute dot product between reflection and view directions
    dot_product = torch.sum(R * view_dir, dim=-1)  # [N]
    
    # Apply Beta kernel (1 - R·V)^(4e^b)
    beta = (torch.clamp(dot_product, 0.0, 1.0)) ** (4 * torch.exp(tensor[:,2]))
        
        # Add specular contribution
    return beta.unsqueeze(-1) * tensor[:,3:]

def eval_sb(sb_params, view_dir):
    """
    Evaluate Spherical Beta color model
    :param sb_params: [N, 15] tensor containing SB parameters
    :param view_dir: [N, 3] normalized view direction vectors
    :return: [N, 3] RGB colors
    """
    
    SH_DEGREES = (sb_params.shape[-1] - 3) / 6
    
    # Base diffuse color
    color = sb_params[:, :3]
    
    if SH_DEGREES > 0:
        color = color + sb_eval(sb_params[:, 3:9], view_dir)
    if SH_DEGREES > 1:
       color = color + sb_eval(sb_params[:, 9:15], view_dir)
    if SH_DEGREES > 2:
        color = color + sb_eval(sb_params[:, 15:21], view_dir)
    if SH_DEGREES > 3:
        color = color + sb_eval(sb_params[:, 21:27], view_dir)
    if SH_DEGREES > 4:
        color = color + sb_eval(sb_params[:, 27:33], view_dir)
    
    return torch.clamp_min(color, 0.0)