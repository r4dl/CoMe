# training script for TNT dataset

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import constants as C
from constants import dispatch_jobs

scenes = C.SCENES_TNT
factors = C.FACTORS_TNT

ITERATIONS = 30000
STD_ARGS = f'--iterations {ITERATIONS} --lambda_distortion 100 --eval --far_plane 100. --detach_alpha False'
OUT_DIR = 'output/TNT_ABLATION_1'
# gt data is assumed to be in the same directory as train data
TNT_GT_DATA = f'{C.DATA_DIR}/TNT_GOF/'
TNT_TRAIN_DATA = f'{C.DATA_DIR}/TNT_GOF'

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

def run_command(cmd):
    status = os.system(cmd)
    exit_code = os.waitstatus_to_exitcode(status)
    if exit_code != 0:
        print(f"Command failed with exit code {exit_code}:\n{cmd}")
    return exit_code == 0

def train_scene(gpu, scene, factor, out_dir, args):
    out_path = Path(out_dir) / scene
    out_path.mkdir(parents=True, exist_ok=True)
    mesh_path = out_path / f"test/ours_{ITERATIONS}/mesh_faster_binary_search_7.ply"

    # optimization
    if not (out_path / f"point_cloud/iteration_{ITERATIONS}/point_cloud.ply").exists():
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
                python train.py -s {TNT_TRAIN_DATA}/{scene} \
                -m {out_dir}/{scene} \
                -r {factor} \
                {STD_ARGS} {args} \
                --port {6009+gpu} > {out_path}/train.log 2>&1"
        if not run_command(cmd):
            print(f"Skipping marching tets/eval for {scene} because training failed.")
            return False

    # marching tets
    if not mesh_path.exists():
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
                python extract_mesh_tets.py -m {out_dir}/{scene} \
                --iteration {ITERATIONS} \
                --data_device cpu"
        if not run_command(cmd):
            print(f"Skipping eval for {scene} because marching tets failed.")
            return False
    
    # evaluate
    results_path = out_dir / Path(scene) / "eval/results.json"
    if not results_path.exists():
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
                python mesh_utils/eval_TNT.py \
                --dataset-dir {TNT_GT_DATA}/{scene} \
                --ply-path {mesh_path} \
                --traj-path {TNT_GT_DATA}/{scene}/{scene}_traj_path.log \
                --out-dir {out_dir}/{scene}/eval"
        return run_command(cmd)
    return True
    
# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor, train_scene)

# Run reports after all scenes/configs have finished.
for config_name in configs:
    out_dir = f"{OUT_DIR}/{config_name}"
    cmd = f"python report.py --input_dir {out_dir}"
    os.system(cmd)