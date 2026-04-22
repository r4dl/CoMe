import json
import numpy as np
import constants as C
import subprocess
from latex_utils import print_latex_table

scenes = C.SCENES_SCN

output_dirs = [
    "output/SCN_ABLATION_1/CoMe",
]
KEYS = ["precision", "recall", "fscore"]

RESULTS = {}

def show_results(o):
    all_metrics = {"precision": [], "recall": [], "fscore": []}
    for scene in scenes:

        json_file = f"{o}/{scene}/eval/results.json"
        import os
        if not os.path.exists(json_file):
            [all_metrics[k].append(0.0) for k in KEYS]
            continue
        data = json.load(open(json_file))
        
        for k in KEYS:
            all_metrics[k].append(data[k])

    print(f'\t{C.YELLOW}{o}{C.RESET}')
    for z in KEYS:
        latex = []
        for k in KEYS:
            numbers = np.asarray(all_metrics[k]).mean(axis=0).tolist()
            
            numbers = all_metrics[k] + [numbers]
            
            numbers = [f"{x:.3f}" for x in numbers]
            if k == z:
                latex.extend(numbers)
            
        
        print(f'{C.RED}{z}:{C.RESET} ' + " & ".join([str(s) for s in scenes]))
        print(" & ".join(latex))

    # print the number of gaussians as well
    num_gaussians = []
    for scene in scenes:
        fpath = f'{o}/{scene}/point_cloud/iteration_30000/point_cloud.ply'
        if not os.path.exists(fpath):
            num_gaussians += [0]
            continue
        else:
            # the 3rd line always contains the number of gaussians
            result = subprocess.run(['head', '-n', '3', fpath], stdout=subprocess.PIPE, text=True)
            # get 3rd line, remove 'element vertex ' and convert to int
            num_gaussians += [int(result.stdout.split('\n')[2][15:])]
        
    print(f'{C.RED}primitives{C.RESET}: ' + " & ".join([str(s) for s in scenes]))
    
    # add average
    num_gaussians += [int(np.asarray(num_gaussians).mean())]
    
    formatted = [C.human_format(int(n)) for n in num_gaussians]
    print(" & ".join(formatted))

    # add to dict for table printing
    from pathlib import Path
    scene_name = str(Path(o).name).replace("_", "\_")
    RESULTS[scene_name] = all_metrics["fscore"]


for o in output_dirs:
    print('')
    show_results(o)

print_latex_table(RESULTS, scenes)
