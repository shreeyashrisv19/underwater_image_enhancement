import cv2
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_folder = os.path.join(ROOT_DIR, "inputs")
output_folder = os.path.join(ROOT_DIR, "waternet_inputs")

os.makedirs(output_folder, exist_ok=True)


def adaptive_color_balance(img):
    img = img.astype(np.float32)

    for c in range(3):
        channel = img[:, :, c]
        low = np.percentile(channel, 1)
        high = np.percentile(channel, 99)

        channel = (channel - low) * 255 / (high - low + 1e-6)
        img[:, :, c] = np.clip(channel, 0, 255)

    return img.astype(np.uint8)


def apply_CLAHE(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])

    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)

    result = np.clip(result, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def gamma_correction(img, gamma=1.2):
    inv = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv) * 255 for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(img, table)


for filename in os.listdir(input_folder):

    if filename.endswith((".png", ".jpg")):

        path = os.path.join(input_folder, filename)
        img = cv2.imread(path)

        name = os.path.splitext(filename)[0]

        img = adaptive_color_balance(img)

        img_ce = apply_CLAHE(img)
        img_wb = white_balance(img)
        img_gc = gamma_correction(img)

        cv2.imwrite(f"{output_folder}/{name}.jpg", img)
        cv2.imwrite(f"{output_folder}/{name}_ce.jpg", img_ce)
        cv2.imwrite(f"{output_folder}/{name}_wb.jpg", img_wb)
        cv2.imwrite(f"{output_folder}/{name}_gc.jpg", img_gc)

print("Preprocessing complete")
