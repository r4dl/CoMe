# training script NVS datasets

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import constants as C
from constants import dispatch_jobs

scenes = C.SCENES_NVS
factors = C.FACTORS_NVS
TRAIN_DATA = f'{C.DATA_DIR}/mip360'

ITERATIONS = 30000
STD_ARGS = f'--iterations {ITERATIONS} --lambda_distortion 100 --far_plane 100. --detach_alpha False'
OUT_DIR = 'output/NVS_ABLATION_1'

DRY_RUN = False

configs = {
    "CoMe": "--splatting_config configs/hierarchical.json \
        --use_ssimdecoupled_appearance \
        --color_confidence --color_confidence_max 0.075 --color_confidence_from_iter 500 \
        --lambda_variance 0.5 --lambda_normal_variance 0.005",
}

# jobs as a cross product of scenes and configs
jobs = [
    (scenes[idx], factors[idx], f'{OUT_DIR}/{config_name}', config_args) 
    for idx,_ in enumerate(scenes) 
    for config_name, config_args in configs.items()
]

def train_scene(gpu, scene, factor, out_dir, args):
    # only run if the point cloud does not exist
    if not os.path.exists(f'{out_dir}/{scene}/point_cloud/iteration_{ITERATIONS}/point_cloud.ply'):
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
                python train.py -s {TRAIN_DATA}/{scene} \
                --eval -i images_{factor} \
                -m {out_dir}/{scene}  \
                {STD_ARGS} {args} \
                --port {6049+gpu}"
        os.system(cmd)
        
    if not os.path.exists(f'{out_dir}/{scene}/results_full.json'):
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
                python render.py -m {out_dir}/{scene} \
                -s {TRAIN_DATA}/{scene} --init_type sfm \
                --data_device cpu --skip_train"
        os.system(cmd)
        
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {out_dir}/{scene}"
        os.system(cmd)
        
        # at the end, remove the ground truth images (they need to much disk space)
        os.system(f"rm -rf {out_dir}/{scene}/test/ours_{ITERATIONS}/gt/")
    
    # marching tets
    if not os.path.exists(f'{out_dir}/{scene}/test/ours_{ITERATIONS}/mesh_faster_binary_search_7.ply'):
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
                python extract_mesh_tets.py -m {out_dir}/{scene} \
                --iteration {ITERATIONS} \
                --data_device cpu"
        # by default, not run
        os.system(cmd)
    
    return True


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor, train_scene)