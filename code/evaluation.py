import os
import re
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

merged_dir = os.path.join(ROOT_DIR, "merged_outputs")
gt_dir = os.path.join(ROOT_DIR, "groundtruths")
excel_path = os.path.join(ROOT_DIR, "results", "results.xlsx")

os.makedirs(os.path.dirname(excel_path), exist_ok=True)


def extract_number(filename):
    match = re.search(r'\d+', filename)
    return match.group() if match else None


merged_files = {extract_number(f): f for f in os.listdir(merged_dir)}
gt_files = {extract_number(f): f for f in os.listdir(gt_dir)}

common_keys = sorted(set(merged_files.keys()) & set(gt_files.keys()))

print("Matched images:", common_keys)


results = []

psnr_list = []
ssim_list = []


for key in common_keys:

    merged_path = os.path.join(merged_dir, merged_files[key])
    gt_path = os.path.join(gt_dir, gt_files[key])

    img = cv2.imread(merged_path)
    gt = cv2.imread(gt_path)

    if img is None or gt is None:
        print("Skipping:", key)
        continue

    img = cv2.resize(img, (gt.shape[1], gt.shape[0]))

    img_norm = img.astype(np.float32) / 255.0
    gt_norm = gt.astype(np.float32) / 255.0

    p = psnr(gt_norm, img_norm, data_range=1.0)
    s = ssim(gt_norm, img_norm, channel_axis=2, data_range=1.0)

    psnr_list.append(p)
    ssim_list.append(s)

    results.append({
        "Image_ID": key,
        "PSNR": p,
        "SSIM": s
    })


df = pd.DataFrame(results)

df.to_excel(excel_path, index=False)

print("\nResults saved to:", excel_path)

print("\nAverage Metrics")
print("Average PSNR :", np.mean(psnr_list))
print("Average SSIM :", np.mean(ssim_list))
