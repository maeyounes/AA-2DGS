from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim

import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)

        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, scale):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ f"gt_{scale}"
                renders_dir = method_dir / f"test_preds_{scale}"
                renders, gts, image_names = readImages(renders_dir, gt_dir)
                
                # The images in gts have different resolutions: full res, 1/2, 1/4, 1/8
                # We want to have metrics for separate resolutions then the average
                ssims_full, ssims_half, ssims_quarter, ssims_eighth = [], [], [], []
                psnrs_full, psnrs_half, psnrs_quarter, psnrs_eighth = [], [], [], []
                lpipss_full, lpipss_half, lpipss_quarter, lpipss_eighth = [], [], [], []
                
                # find the base resolution
                base_res = 0
                for idx in range(len(renders)):
                    current_res = renders[idx].shape[2]
                    if current_res > base_res:
                        base_res = current_res
                
                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssim_val = ssim(renders[idx], gts[idx])
                    psnr_val = psnr(renders[idx], gts[idx])
                    lpips_val = lpips_fn(renders[idx], gts[idx]).detach()
                    if renders[idx].shape[2] == base_res:
                        ssims_full.append(ssim_val)
                        psnrs_full.append(psnr_val)
                        lpipss_full.append(lpips_val)
                    elif renders[idx].shape[2] == base_res // 2:
                        ssims_half.append(ssim_val)
                        psnrs_half.append(psnr_val)
                        lpipss_half.append(lpips_val)
                    elif renders[idx].shape[2] == base_res // 4:
                        ssims_quarter.append(ssim_val)
                        psnrs_quarter.append(psnr_val)
                        lpipss_quarter.append(lpips_val)
                    elif renders[idx].shape[2] == base_res // 8:
                        ssims_eighth.append(ssim_val)
                        psnrs_eighth.append(psnr_val)
                        lpipss_eighth.append(lpips_val)
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                # assert that all metrics are the same size across all resolutions
                assert len(ssims_full) == len(ssims_half) == len(ssims_quarter) == len(ssims_eighth)
                assert len(psnrs_full) == len(psnrs_half) == len(psnrs_quarter) == len(psnrs_eighth)
                assert len(lpipss_full) == len(lpipss_half) == len(lpipss_quarter) == len(lpipss_eighth)
                assert len(ssims) == len(psnrs) == len(lpipss)
                
                print(" x1   Res: PSNR : {:>12.7f} | SSIM : {:>12.7f} | LPIPS : {:>12.7f}".format(torch.tensor(psnrs_full).mean(), torch.tensor(ssims_full).mean(), torch.tensor(lpipss_full).mean()))
                print(" x1/2 Res: PSNR : {:>12.7f} | SSIM : {:>12.7f} | LPIPS : {:>12.7f}".format(torch.tensor(psnrs_half).mean(), torch.tensor(ssims_half).mean(), torch.tensor(lpipss_half).mean()))
                print(" x1/4 Res: PSNR : {:>12.7f} | SSIM : {:>12.7f} | LPIPS : {:>12.7f}".format(torch.tensor(psnrs_quarter).mean(), torch.tensor(ssims_quarter).mean(), torch.tensor(lpipss_quarter).mean()))
                print(" x1/8 Res: PSNR : {:>12.7f} | SSIM : {:>12.7f} | LPIPS : {:>12.7f}".format(torch.tensor(psnrs_eighth).mean(), torch.tensor(ssims_eighth).mean(), torch.tensor(lpipss_eighth).mean()))
                print("")
                print(" Average PSNR : {:>12.7f} | Average SSIM : {:>12.7f} | Average LPIPS : {:>12.7f}".format(torch.tensor(psnrs).mean(), torch.tensor(ssims).mean(), torch.tensor(lpipss).mean()))
                print("")

                full_dict[scene_dir][method].update({"SSIM_avg": torch.tensor(ssims).mean().item(),
                                                        "PSNR_avg": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS_avg": torch.tensor(lpipss).mean().item(),
                                                        "SSIM_full": torch.tensor(ssims_full).mean().item(),
                                                        "PSNR_full": torch.tensor(psnrs_full).mean().item(),
                                                        "LPIPS_full": torch.tensor(lpipss_full).mean().item(),
                                                        "SSIM_half": torch.tensor(ssims_half).mean().item(),
                                                        "PSNR_half": torch.tensor(psnrs_half).mean().item(),
                                                        "LPIPS_half": torch.tensor(lpipss_half).mean().item(),
                                                        "SSIM_quarter": torch.tensor(ssims_quarter).mean().item(),
                                                        "PSNR_quarter": torch.tensor(psnrs_quarter).mean().item(),
                                                        "LPIPS_quarter": torch.tensor(lpipss_quarter).mean().item(),
                                                        "SSIM_eighth": torch.tensor(ssims_eighth).mean().item(),
                                                        "PSNR_eighth": torch.tensor(psnrs_eighth).mean().item(),
                                                        "LPIPS_eighth": torch.tensor(lpipss_eighth).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--resolution', '-r', type=int, default=-1)
    
    args = parser.parse_args()
    evaluate(args.model_paths, args.resolution)