import GPUtil
import time

RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'
YELLOW = '\033[33m'
excluded_gpus = set([])

# TODO: modify this when the data is somewhere else
DATA_DIR = '../data'

# all DTU scenes, downsample by a factor of 2
SCENES_DTU = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
FACTORS_DTU = [2] * len(SCENES_DTU)

# all MipNeRF360 scenes, downsample by a factor of 2 (indoor) or 4 (outdoor)
SCENES_NVS = ["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]
FACTORS_NVS = [4, 2, 2, 4, 4, 4, 4, 2, 2]

# all TNT scenes, downsample by a factor of 2
SCENES_TNT = ["Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck"]
FACTORS_TNT = [2] * len(SCENES_TNT)

# all small ScanNet++ scenes, downsample by a factor of 2
SCENES_SCN = ["5a269ba6fe", "08bbbdcc3d", "39f36da05b", "dc263dfbf0", "ef18cf0708", "fb564c935d"]
FACTORS_SCN = [2] * len(SCENES_SCN)

def human_format(num):
    for unit in ['', 'K']:
        if abs(num) < 1000:
            return f"{num:.0f}{unit}"
        num /= 1000
    return f"{num:.2f}M"

def worker(gpu, f, scene, factor, out_dir, args):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    try:
        f(gpu, scene, factor, out_dir, args)
    except Exception as e:
        print(e)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.
    
def dispatch_jobs(jobs, executor, f):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, excludeID=[]))
        # all_available_gpus = set([0,1,2,3])
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
        
        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, f, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(1)
        
    print("All jobs have been processed.")