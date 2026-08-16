import os
import re
import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

retinex_dir = os.path.join(ROOT_DIR, "retinex_outputs")
waternet_dir = os.path.join(ROOT_DIR, "waternet", "output", "batch_run")
output_dir = os.path.join(ROOT_DIR, "merged_outputs")

os.makedirs(output_dir, exist_ok=True)


def extract_number(filename):
    match = re.search(r'\d+', filename)
    return match.group() if match else None


ret_files = {extract_number(f): f for f in os.listdir(retinex_dir)}
wat_files = {extract_number(f): f for f in os.listdir(waternet_dir)}

common_keys = set(ret_files.keys()) & set(wat_files.keys())


def edge_aware_fusion(ret, wat):

    gray = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 150)

    edges = cv2.GaussianBlur(edges, (7, 7), 0)

    edges = edges.astype(np.float32) / 255.0
    edges = np.expand_dims(edges, axis=2)

    ret_norm = ret.astype(np.float32) / 255.0
    wat_norm = wat.astype(np.float32) / 255.0

    alpha = 0.8

    merged = alpha * wat_norm + (1 - alpha) * ret_norm

    merged = merged + 0.15 * edges * (ret_norm - wat_norm)

    merged = np.clip(merged, 0, 1)

    return (merged * 255).astype(np.uint8)


for key in sorted(common_keys):

    print("Processing image:", key)

    ret_path = os.path.join(retinex_dir, ret_files[key])
    wat_path = os.path.join(waternet_dir, wat_files[key])

    ret = cv2.imread(ret_path)
    wat = cv2.imread(wat_path)

    if ret is None or wat is None:
        print("Skipping", key)
        continue

    ret = cv2.resize(ret, (wat.shape[1], wat.shape[0]))

    merged = edge_aware_fusion(ret, wat)

    cv2.imwrite(
        os.path.join(output_dir, f"{key}_merged.png"),
        merged
    )


print("Fusion complete")
