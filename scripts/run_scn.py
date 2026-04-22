# training script for SCN dataset

import os
from concurrent.futures import ThreadPoolExecutor
import constants as C
from constants import dispatch_jobs

scenes = C.SCENES_SCN
factors = C.FACTORS_SCN

excluded_gpus = set([])

ITERATIONS = 30000
STD_ARGS = f'--iterations {ITERATIONS} --lambda_distortion 100 --eval --far_plane 100. --detach_alpha False'
OUT_DIR = 'output/SCN_ABLATION_1'
# gt data is assumed to be in the same directory as train data
SCN_GT_DATA = f'{C.DATA_DIR}/SCN/'
SCN_TRAIN_DATA = f'{C.DATA_DIR}/SCN/'

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
    for idx, _ in enumerate(scenes)
    for config_name, config_args in configs.items()
]


def train_scene(gpu, scene, factor, out_dir, args):
    # optimization
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
            python train.py -s {SCN_TRAIN_DATA}/{scene} \
            -m {out_dir}/{scene} \
            -r {factor} \
            {STD_ARGS} {args} \
            --port {6009 + gpu}"
    os.system(cmd)

    # marching tets
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
            python extract_mesh_tets.py -m {out_dir}/{scene} \
            --iteration {ITERATIONS} \
            --data_device cpu"
    os.system(cmd)

    # evaluate
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
            python mesh_utils/eval_SCN.py \
            --dataset-dir {SCN_GT_DATA}/{scene} \
            --ply-path {out_dir}/{scene}/test/ours_{ITERATIONS}/mesh_faster_binary_search_7.ply \
            --out-dir {out_dir}/{scene}/eval"
    os.system(cmd)

    return True


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor, train_scene)

# Run reports after all scenes/configs have finished.
for config_name in configs:
    out_dir = f"{OUT_DIR}/{config_name}"
    cmd = f"python report.py --input_dir {out_dir}"
    os.system(cmd)
