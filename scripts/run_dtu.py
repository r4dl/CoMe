# training script for DTU dataset

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import constants as C
from constants import dispatch_jobs

scenes = C.SCENES_DTU
factors = C.FACTORS_DTU

ITERATIONS = 30000
STD_ARGS = f'--iterations {ITERATIONS} --lambda_distortion 1000. --eval --far_plane 100. --detach_alpha True --lambda_opacity_field 0.0 --lambda_smoothness 0.0 --lambda_extent 0.0'
OUT_DIR = 'output/DTU_ABLATION_1'
# not the same directory.
DTU_GT_DATA = f'{C.DATA_DIR}/DTU_GT/SampleSet/MVSData/'
DTU_TRAIN_DATA = f'{C.DATA_DIR}/DTU'

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
    if not os.path.exists(f'{out_dir}/scan{scene}/point_cloud/iteration_{ITERATIONS}/point_cloud.ply'):
        cmd = f" CUDA_VISIBLE_DEVICES={gpu} \
                python train.py -s {DTU_TRAIN_DATA}/scan{scene} \
                -m {out_dir}/scan{scene} \
                -r {factor} \
                {STD_ARGS} {args} \
                --port {6099+gpu}"
        os.system(cmd)
    
    # tsdf fusion
    if not os.path.exists(f'{out_dir}/scan{scene}/test/ours_{ITERATIONS}/tsdf.ply'):
        cmd = f" CUDA_VISIBLE_DEVICES={gpu} \
                python extract_mesh_tsdf.py \
                -m {out_dir}/scan{scene}"
        os.system(cmd)
    
    # evaluate
    if not os.path.exists(f'{out_dir}/scan{scene}/TSDF/results.json'):
        cmd = f" CUDA_VISIBLE_DEVICES={gpu} \
                python mesh_utils/eval_DTU.py \
                --instance_dir {DTU_TRAIN_DATA}/scan{scene} \
                --input_mesh {out_dir}/scan{scene}/test/ours_{ITERATIONS}/tsdf.ply \
                --dataset_dir {DTU_GT_DATA} \
                --vis_out_dir {out_dir}/scan{scene}/TSDF"
        os.system(cmd)
    
    return True


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor, train_scene)