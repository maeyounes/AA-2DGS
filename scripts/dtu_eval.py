import os
from argparse import ArgumentParser

dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", required=True)
parser.add_argument('--dtu', "-dtu", required=True, type=str)
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(dtu_scenes)
iteration = 30000

if not args.skip_metrics:
    parser.add_argument('--DTU_Official', "-DTU", required=True, type=str)
    args = parser.parse_args()


if not args.skip_training:
    common_args = " --quiet --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 1000"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)


if not args.skip_rendering:
    all_sources = []
    common_args = " --quiet --skip_train --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print(f"python render_mesh.py --iteration {iteration} -s " + source + " -m" + args.output_path + "/" + scene + common_args)
        os.system(f"python render_mesh.py --iteration {iteration} -s " + source + " -m" + args.output_path + "/" + scene + common_args)


if not args.skip_metrics:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for scene in dtu_scenes:
        scan_id = scene[4:]
        ply_file = f"{args.output_path}/{scene}/train/ours_{iteration}/"
        string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
            f"--input_mesh {args.output_path}/{scene}/train/ours_{iteration}/fuse_post.ply " + \
            f"--scan_id {scan_id} --output_dir {ply_file} " + \
            f"--mask_dir {args.dtu} " + \
            f"--DTU {args.DTU_Official}"
        
        os.system(string)