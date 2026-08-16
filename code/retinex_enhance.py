import cv2
import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_dir = os.path.join(ROOT_DIR, "inputs")
output_dir = os.path.join(ROOT_DIR, "retinex_outputs")

os.makedirs(output_dir, exist_ok=True)


def single_scale_retinex(img, sigma):

    blur = cv2.GaussianBlur(img, (0, 0), sigma)

    retinex = np.log1p(img) - np.log1p(blur)

    return retinex


def multi_scale_retinex(img, scales):

    ret = np.zeros_like(img, dtype=np.float32)

    for s in scales:
        ret += single_scale_retinex(img, s)

    ret = ret / len(scales)

    return ret


def adaptive_retinex(img):

    mean_val = img.mean()

    if mean_val < 80:
        scales = [5, 30, 100]
    else:
        scales = [15, 80, 250]

    img = img.astype(np.float32) + 1

    ret = multi_scale_retinex(img, scales)

    ret = cv2.normalize(ret, None, 0, 255, cv2.NORM_MINMAX)

    ret = ret.astype(np.uint8)

    ret = cv2.detailEnhance(ret, sigma_s=10, sigma_r=0.15)

    ret = cv2.bilateralFilter(ret, 5, 50, 50)

    return ret


for f in os.listdir(input_dir):

    if f.lower().endswith((".png", ".jpg", ".jpeg")):

        img_path = os.path.join(input_dir, f)

        img = cv2.imread(img_path)

        if img is None:
            print("Skipping unreadable image:", img_path)
            continue

        print("Processing:", f)

        enhanced = adaptive_retinex(img)

        name = os.path.splitext(f)[0]

        save_path = os.path.join(output_dir, f"{name}_retinex.png")

        cv2.imwrite(save_path, enhanced)


print("Retinex processing complete")
