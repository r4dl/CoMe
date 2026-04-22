import json
import numpy as np
import constants as C
import subprocess

output_dirs = [
    "output/NVS_ABLATION_1/CoMe",
]
KEYS = ["PSNR", "SSIM", "LPIPS", "FLIPS"]
SCENES = C.SCENES_NVS

FULL_RESULTS = []

def eval_for_scenes(o):
    all_metrics = {"PSNR": [], "SSIM": [], "LPIPS": [], "FLIPS": []}

    num_gaussians = []

    for scene in SCENES:
        
        json_file = f"{o}/{scene}/results_full.json"
        data = json.load(open(json_file))

        for k in KEYS:
            all_metrics[k].append(data['ours_30000'][k])
            
        # also load the number of gaussians
        result = subprocess.run(['head', '-n', '3', f'{o}/{scene}/point_cloud/iteration_30000/point_cloud.ply'], stdout=subprocess.PIPE, text=True)
        num_gaussians += [int(result.stdout.split('\n')[-2][15:])]

    print(f'\t{C.YELLOW}{o}{C.RESET}')
    print(" & ".join([str(z) for z in SCENES]))
    
    for z in KEYS:
        latex = []
        for k in KEYS:
            numbers = np.asarray(all_metrics[k]).mean(axis=0).tolist()
            
            numbers = all_metrics[k] + [numbers]
            
            numbers = [f"{x:.3f}" for x in numbers]
            if k == z:
                latex.extend(numbers)
        print(f'{C.RED}{z}{C.RESET}: ' + " & ".join(latex))
        
    num_gaussians += [int(np.asarray(num_gaussians).mean())]
    
    formatted = [C.human_format(int(n)) for n in num_gaussians]
    print(f'{C.RED}NUM{C.RESET}: ' + " & ".join(formatted))
    
    print()

for o in output_dirs:
    print('')
    eval_for_scenes(o)