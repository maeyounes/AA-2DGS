import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import argparse
import sys

dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']
output_path = "aa_output/benchmark_dtu_mipfiltering"
dtu_dataset_path = "datasets/2DGS_data/DTU"
dtu_official_path = "datasets/dtu_gt/SampleSet/MVS_Data"



SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EVAL_DTU_SCRIPT = os.path.join(SCRIPT_DIR, "eval_dtu", "evaluate_single_scene.py")

def process_dtu_scene(gpu, scene, args):
    """
    Processes a single DTU scene: training, rendering, and metrics.
    """
    scene_output_path = os.path.join(output_path, scene)
    scene_dataset_path = os.path.join(dtu_dataset_path, scene)
    
    print(f"[GPU {gpu}] Processing scene: {scene}")

    # 1. Training
    if not args.skip_training:
        cmd_train = (
            f"OMP_NUM_THREADS={args.omp_num_threads} CUDA_VISIBLE_DEVICES={gpu} python train.py "
            f"-s {scene_dataset_path} -m {scene_output_path} --quiet --test_iterations -1 "
            f"--depth_ratio 1.0 -r 2 --lambda_dist 1000 --port {6000 + gpu + os.getpid() % 100}" 
            f" --kernel_size 0.1"
        )
        print(f"[GPU {gpu}][{scene}] Training command: {cmd_train}")
        if not args.dry_run:
            os.makedirs(scene_output_path, exist_ok=True)
            if os.system(cmd_train) != 0:
                print(f"[GPU {gpu}][{scene}] Training failed for scene {scene}. Skipping subsequent steps for this scene.")
                return False
        print(f"[GPU {gpu}][{scene}] Training finished.")
    else:
        print(f"[GPU {gpu}][{scene}] Skipping training.")

    # 2. Rendering (Mesh Extraction)
    if not args.skip_rendering:
        cmd_render = (
            f"OMP_NUM_THREADS={args.omp_num_threads} CUDA_VISIBLE_DEVICES={gpu} python render_mesh.py "
            f"--iteration {args.iteration} -s {scene_dataset_path} -m {scene_output_path} "
            f"--quiet --skip_train --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 "
            f"--sdf_trunc 0.016 --depth_trunc 3.0"
        )
        print(f"[GPU {gpu}][{scene}] Rendering command: {cmd_render}")
        if not args.dry_run:
            if os.system(cmd_render) != 0:
                print(f"[GPU {gpu}][{scene}] Rendering failed for scene {scene}. Skipping metrics.")
                return False
        print(f"[GPU {gpu}][{scene}] Rendering finished.")
    else:
        print(f"[GPU {gpu}][{scene}] Skipping rendering.")

    # 3. Metrics Evaluation
    if not args.skip_metrics:

        scan_id = scene[4:]
        ply_output_dir = os.path.join(scene_output_path, "train", f"ours_{args.iteration}")
        input_mesh_path = os.path.join(ply_output_dir, "fuse_post.ply")

        if not args.dry_run and not os.path.exists(input_mesh_path):
            print(f"[GPU {gpu}][{scene}] Mesh file {input_mesh_path} not found. Skipping metrics.")
            return False

        cmd_metrics = (
            f"OMP_NUM_THREADS={args.omp_num_threads} CUDA_VISIBLE_DEVICES={gpu} python {EVAL_DTU_SCRIPT} "
            f"--input_mesh {input_mesh_path} "
            f"--scan_id {scan_id} --output_dir {ply_output_dir} "
            f"--mask_dir {dtu_dataset_path} " # mask_dir should be base dtu_dataset_path, not scene specific for eval script
            f"--DTU {dtu_official_path}"
        )
        print(f"[GPU {gpu}][{scene}] Metrics command: {cmd_metrics}")
        if not args.dry_run:
            if os.system(cmd_metrics) != 0:
                print(f"[GPU {gpu}][{scene}] Metrics evaluation failed for scene {scene}.")
                return False
        print(f"[GPU {gpu}][{scene}] Metrics evaluation finished.")
    else:
        print(f"[GPU {gpu}][{scene}] Skipping metrics.")
    
    return True

def worker(gpu, scene, args_namespace):
    print(f"Starting job on GPU {gpu} for scene {scene}\n")
    success = process_dtu_scene(gpu, scene, args_namespace)
    if success:
        print(f"Successfully finished all stages for job on GPU {gpu} with scene {scene}\n")
    else:
        print(f"Job on GPU {gpu} with scene {scene} encountered errors or was incomplete.\n")
    return gpu # Return GPU to signal it's free

def dispatch_jobs(jobs_to_run, executor, args, eligible_gpu_ids):
    future_to_job_info = {}
    reserved_gpus = set() 
    job_queue = list(jobs_to_run) # Make a mutable copy

    max_concurrent_jobs = len(eligible_gpu_ids)
    if args.max_gpus is not None:
        max_concurrent_jobs = min(max_concurrent_jobs, args.max_gpus)
    
    print(f"Eligible GPUs for processing: {eligible_gpu_ids}")
    print(f"Maximum concurrent jobs: {max_concurrent_jobs}")


    while job_queue or future_to_job_info:
        # Check for completed jobs and release GPUs
        done_futures = [f for f in future_to_job_info if f.done()]
        for future in done_futures:
            finished_gpu, finished_scene = future_to_job_info.pop(future)
            reserved_gpus.discard(finished_gpu)
            try:
                future.result() # To raise exceptions if any occurred in the worker
                print(f"Job for scene {finished_scene} on GPU {finished_gpu} completed and GPU released.")
            except Exception as e:
                print(f"Job for scene {finished_scene} on GPU {finished_gpu} failed with error: {e}")


        # Try to launch new jobs if there are job_queue and available GPUs and we are below concurrency limit
        if len(future_to_job_info) < max_concurrent_jobs:
            # Get truly available GPUs (not just eligible, but also free according to GPUtil and not reserved by us)
            # Order by 'first' (usually lowest ID), 'load', 'memory'
            try:
                gputil_available_gpus = set(GPUtil.getAvailable(order='first', limit=len(eligible_gpu_ids), maxLoad=0.8, maxMemory=0.8))
            except Exception as e:
                print(f"Warning: Could not query GPUtil for available GPUs: {e}. Assuming all eligible GPUs are available if not reserved.")
                gputil_available_gpus = set(eligible_gpu_ids)


            # Filter by our eligible list and ensure they are not already reserved
            candidate_gpus = list(gputil_available_gpus.intersection(set(eligible_gpu_ids)) - reserved_gpus)
            
            while candidate_gpus and job_queue and len(future_to_job_info) < max_concurrent_jobs:
                gpu_to_assign = candidate_gpus.pop(0)
                current_scene = job_queue.pop(0)
                
                print(f"Submitting job for scene {current_scene} to GPU {gpu_to_assign}")
                future = executor.submit(worker, gpu_to_assign, current_scene, args)
                future_to_job_info[future] = (gpu_to_assign, current_scene)
                reserved_gpus.add(gpu_to_assign)

        if not job_queue and not future_to_job_info:
            break 
            
        time.sleep(args.check_interval)
        
    print("All DTU jobs have been processed or attempted.")

def main():
    parser = argparse.ArgumentParser(description="Run multi-GPU DTU training, rendering, and evaluation.")
    
    parser.add_argument("--skip_training", action="store_true", help="Skip the training stage.")
    parser.add_argument("--skip_rendering", action="store_true", help="Skip the rendering/mesh extraction stage.")
    parser.add_argument("--skip_metrics", action="store_true", help="Skip the metrics evaluation stage.")
    
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration number for rendering and metrics.")
    parser.add_argument("--excluded_gpus", type=str, default="", help="Comma-separated list of GPU IDs to exclude (e.g., '0,1').")
    parser.add_argument("--max_gpus", type=int, default=None, help="Maximum number of GPUs to use concurrently.")
    parser.add_argument("--omp_num_threads", type=int, default=4, help="Number of OMP threads to use for each job.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--check_interval", type=int, default=10, help="Interval in seconds to check for job completion and dispatch new jobs.")


    args = parser.parse_args()

    if not os.path.exists(EVAL_DTU_SCRIPT) and not args.skip_metrics:
        print(f"Error: DTU evaluation script not found at {EVAL_DTU_SCRIPT}. Please ensure it's in the 'eval_dtu' subdirectory or adjust SCRIPT_DIR.")
        sys.exit(1)

    # Determine eligible GPUs
    try:
        all_physical_gpu_ids = [gpu.id for gpu in GPUtil.getGPUs()]
        if not all_physical_gpu_ids and not args.dry_run: # Allow dry run without GPUs
             print("Error: No GPUs detected by GPUtil. If you have GPUs, ensure drivers and GPUtil are correctly set up.")
             sys.exit(1)
    except Exception as e:
        if not args.dry_run:
            print(f"Error getting GPU list: {e}. Please ensure NVIDIA drivers and GPUtil are installed and working.")
            sys.exit(1)
        all_physical_gpu_ids = [] # For dry run

    parsed_excluded_gpus = set()
    if args.excluded_gpus:
        try:
            parsed_excluded_gpus = set(map(int, args.excluded_gpus.split(',')))
        except ValueError:
            print("Error: Invalid format for --excluded_gpus. Expected comma-separated integers (e.g., '0,1').")
            sys.exit(1)
            
    eligible_gpu_ids = sorted(list(set(all_physical_gpu_ids) - parsed_excluded_gpus))

    if not eligible_gpu_ids and not args.dry_run:
        print("Error: No eligible GPUs available after exclusions. Exiting.")
        sys.exit(1)
    
    num_worker_threads = len(eligible_gpu_ids)
    if args.max_gpus is not None:
        if args.max_gpus <= 0:
            print("Error: --max_gpus must be a positive integer.")
            sys.exit(1)
        num_worker_threads = min(num_worker_threads, args.max_gpus)
    
    if num_worker_threads == 0 and not args.dry_run:
        print("Error: Number of worker threads is zero. No GPUs to run on. Exiting.")
        sys.exit(1)
    
    if args.dry_run:
        print("Dry run mode enabled. Commands will be printed but not executed.")
        if not eligible_gpu_ids: # For dry run, if no GPUs, assign dummy IDs for command generation
            num_dummy_gpus = args.max_gpus if args.max_gpus else 1
            eligible_gpu_ids = list(range(num_dummy_gpus))
            num_worker_threads = len(eligible_gpu_ids)
            print(f"Dry run: Using dummy GPU IDs {eligible_gpu_ids} for command generation.")


    jobs_to_process = list(dtu_scenes) # All scenes by default

    print(f"Starting DTU processing for {len(jobs_to_process)} scenes.")
    print(f"Output path: {output_path}")
    print(f"DTU dataset path: {dtu_dataset_path}")
    print(f"DTU official path: {dtu_official_path}")
    
    # Using ThreadPoolExecutor to manage the thread pool
    # num_worker_threads can be 0 in dry_run if no GPUs and no max_gpus specified.
    # Ensure max_workers is at least 1 for the executor to function, even if no jobs run.
    actual_max_workers = max(1, num_worker_threads) 
    with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
        if num_worker_threads > 0 or args.dry_run: # Only dispatch if there are workers or it's a dry run
             dispatch_jobs(jobs_to_process, executor, args, eligible_gpu_ids)
        else:
            print("No GPUs available or configured for work. Exiting.")

if __name__ == "__main__":
    main()