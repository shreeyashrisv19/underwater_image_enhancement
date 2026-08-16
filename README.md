# Underwater Image Enhancement

## Adaptive Contrast-Aware Retinex and Deep Learning Framework with Edge-Aware Fusion for Underwater Image Enhancement

This project presents a hybrid underwater image enhancement framework that combines adaptive image preprocessing, deep-learning-based enhancement using WaterNet, adaptive multi-scale Retinex enhancement, and edge-aware fusion.

The objective is to improve underwater image quality by addressing common issues such as color distortion, low contrast, non-uniform illumination, and loss of fine details.

---

## Project Overview

Underwater images often suffer from color casts, reduced contrast, uneven illumination, haze, and loss of details because light is absorbed and scattered differently under water.

This project explores a hybrid enhancement pipeline that combines complementary enhancement approaches:

- Adaptive preprocessing for color and contrast correction
- WaterNet for deep-learning-based enhancement
- Adaptive multi-scale Retinex for illumination and contrast enhancement
- Edge-aware fusion for combining the complementary outputs
- PSNR and SSIM for quantitative evaluation

The final system combines the strengths of the deep-learning and Retinex branches while preserving image details through edge-aware fusion.

---

## Pipeline

```text
Input Underwater Image
          |
          v
Adaptive Preprocessing
          |
          +----------------------+
          |                      |
          v                      v
      WaterNet          Adaptive Retinex
          |                      |
          |                      |
          +----------+-----------+
                     |
                     v
             Edge-Aware Fusion
                     |
                     v
            Enhanced Image
                     |
                     v
              PSNR / SSIM
```

---

## Methodology
### 1. Adaptive Preprocessing

The input image is first processed using adaptive color and contrast correction.

The preprocessing stage includes:

Percentile-based adaptive color correction
CLAHE (Contrast Limited Adaptive Histogram Equalization)
White balance
Gamma correction

The percentile-based correction uses the lower and upper intensity percentiles to reduce extreme illumination and color variations before subsequent enhancement.

### 2. WaterNet Enhancement

WaterNet is used as the deep-learning enhancement branch of the framework.

The WaterNet implementation and pretrained model are used as an existing enhancement component rather than being developed from scratch in this project.

The WaterNet output provides a learned enhancement representation that complements the traditional Retinex-based branch.

Reference implementation:

https://github.com/Li-Chongyi/Water-Net_Code

### 3. Adaptive Multi-Scale Retinex

A separate Retinex enhancement branch is used to improve illumination and local contrast.

The Retinex scales are selected adaptively according to the mean brightness of the input image.

For darker images, smaller scales are used, while brighter images use larger scales.

The Retinex output is subsequently normalized and refined using detail enhancement and bilateral filtering.

### 4. Edge-Aware Fusion

The WaterNet and Retinex outputs contain complementary information.

An edge-aware fusion stage combines the two outputs.

Canny edge detection is used to identify important structural regions, followed by Gaussian smoothing of the edge map.

The normalized WaterNet and Retinex outputs are then combined using a weighted fusion strategy with an additional edge-dependent detail term.

This helps retain structural details while combining the enhancement characteristics of both branches.

---

## My Contribution

The project uses WaterNet as an existing deep-learning enhancement component.

The main contribution of this work is the development and integration of the hybrid enhancement pipeline, including:

Adaptive preprocessing
Adaptive Retinex scale selection
Retinex-based contrast enhancement
Edge-aware fusion of WaterNet and Retinex outputs
End-to-end pipeline integration
Quantitative evaluation using PSNR and SSIM

The project therefore focuses on integrating complementary enhancement techniques rather than developing the WaterNet model itself.

---

## Dataset

The project uses the Underwater Image Enhancement Benchmark (UIEB) dataset.

UIEB contains 950 real-world underwater images, including 890 images with corresponding reference images.

The dataset was used for enhancement experiments and quantitative evaluation.

Dataset and benchmark reference:

C. Li, C. Guo, W. Ren, R. Cong, J. Hou, S. Kwong, and D. Tao,
"An Underwater Image Enhancement Benchmark Dataset and Beyond,"
IEEE Transactions on Image Processing, vol. 29, pp. 4376–4389, 2020.

DOI: 10.1109/TIP.2019.2955241

---

## Evaluation

The final enhanced images are evaluated using two full-reference image quality metrics:

### PSNR

Peak Signal-to-Noise Ratio measures the pixel-level reconstruction quality between the enhanced image and its reference image.

### SSIM

Structural Similarity Index measures structural similarity between the enhanced image and the corresponding reference image.
| Metric | Average |
| ------ | ------- |
| PSNR   | 19.9579 |
| SSIM   | 0.8478  |

Detailed evaluation results are available in:

results/evaluation_results.xlsx

---

## Visual Comparison

The following comparison shows representative results from the pipeline:

Input → Retinex → WaterNet → Fusion → Ground Truth

---

## Repository Structure
```text
underwater_image_enhancement/
│
├── code/
│   ├── preprocessing.py
│   ├── retinex_enhance.py
│   ├── run_waternet.py
│   ├── fusion.py
│   └── evaluation.py
│
├── results/
│   ├── final_comparison.png
│   └── evaluation_results.xlsx
│
└── README.md
```
---

## Technologies Used
Python
OpenCV
NumPy
Pandas
Scikit-image
Matplotlib
WaterNet
Multi-Scale Retinex
Canny Edge Detection
CLAHE
PSNR
SSIM

---

## Code Description
### preprocessing.py

Performs adaptive preprocessing including color correction, CLAHE, white balancing, and gamma correction.

### retinex_enhance.py

Implements adaptive multi-scale Retinex enhancement based on image brightness.

### run_waternet.py

Runs the WaterNet enhancement process using the existing WaterNet implementation and pretrained model.

### fusion.py

Combines WaterNet and Retinex outputs using edge-aware fusion.

### evaluation.py

Calculates PSNR and SSIM between the enhanced images and reference images.

---

## Running the Project

The repository contains the main processing scripts used in the project.

The complete execution requires the appropriate Python environment, dependencies, input images, reference images, and the externally provided WaterNet implementation and pretrained model.

The WaterNet model and UIEB dataset are not redistributed in this repository.

The general processing sequence is:
1. Preprocess input images
2. Generate WaterNet-enhanced images
3. Generate Retinex-enhanced images
4. Fuse WaterNet and Retinex outputs
5. Evaluate the final results using PSNR and SSIM

---

## References
### UIEB and WaterNet

C. Li, C. Guo, W. Ren, R. Cong, J. Hou, S. Kwong, and D. Tao,
"An Underwater Image Enhancement Benchmark Dataset and Beyond,"
IEEE Transactions on Image Processing, vol. 29, pp. 4376–4389, 2020.
DOI: 10.1109/TIP.2019.2955241

### WaterNet Source Code

Li-Chongyi, Water-Net Code:

https://github.com/Li-Chongyi/Water-Net_Code

---

## Acknowledgement

This project uses the publicly available WaterNet implementation and the UIEB dataset as external resources.

The hybrid preprocessing, adaptive Retinex enhancement, edge-aware fusion, pipeline integration, and evaluation were developed as part of this project.
