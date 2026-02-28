# 🛰️ SAR Image Despeckling using U-Net

An interactive web application powered by **TensorFlow** and **Streamlit** that utilizes deep learning to remove speckle noise from Synthetic Aperture Radar (SAR) imagery in real-time.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](INSERT_YOUR_STREAMLIT_LINK_HERE)

## 🚀 Overview

Synthetic Aperture Radar (SAR) is essential for remote sensing due to its all-weather, day-and-night imaging capabilities. However, SAR images are inherently degraded by **speckle noise**, a granular interference that obscures fine details. 

This project implements a **U-Net Convolutional Neural Network** to effectively despeckle radar data while preserving critical structural edges. This tool serves as a foundational preprocessing step for advanced tasks like **SAR and Hyperspectral (HSI) Image Fusion**.



## 🖥️ App Preview & Datasets

### 1. Mendeley SAR Data
* **Source:** [SAR despeckling filters dataset by Mendeley Data](https://data.mendeley.com/datasets/2xf5v5pwkr/1)
* **Description:** Used for primary training and validation to establish a robust baseline.

| Original SAR Input | Deployment Screenshot 1 | Deployment Screenshot 2 |
| :--- | :---: | :---: |
| ![Mendeley Original](mendeley_original.png) | ![Mendeley Deploy 1](mendeley_deploy_1.png) | ![Mendeley Deploy 2](mendeley_deploy_2.png) |

---

### 2. Chandrayaan-2 DFSAR Dataset
* **Source:** ISRO Pradan / Chandrayaan-2 DFSAR
* **Target File:** `(  )` *(Insert specific filename here)*
* **Description:** High-resolution orbital radar data capturing the lunar surface.

| Original SAR Input | Deployment Screenshot 1 | Deployment Screenshot 2 |
| :--- | :---: | :---: |
| ![Chandrayaan Original](chandrayaan_original.png) | ![Chandrayaan Deploy 1](chandrayaan_deploy_1.png) | ![Chandrayaan Deploy 2](chandrayaan_deploy_2.png) |

---

## ✨ Features

* **Real-time Deep Inference:** Upload `.tiff` files and process them instantly through a fine-tuned U-Net model.
* **Dual Dataset Validation:** Benchmarked on standard terrestrial SAR and lunar orbital radar.
* **Advanced Loss Function:** Optimized using a composite of **SSIM**, **MAE**, and **Total Variation (TV)** loss to prevent blurring.
* **Residual Analysis:** Visualizes the "speckle pattern" removed by the AI to verify data preservation.
* **Export Ready:** Download processed images directly for use in GIS or further research pipelines.

## 🛠️ Tech Stack

* **Python:** Core logic and matrix manipulation.
* **TensorFlow/Keras:** Deep learning framework for U-Net implementation.
* **Streamlit:** Web interface and cloud deployment.
* **Rasterio:** Handling of complex geospatial `.tiff` metadata.
* **NumPy & Scikit-Image:** Mathematical operations and image metrics.

## 📂 Repository Structure

```text
├── streamlit_app.py        # Streamlit Web Application
├── best_sar_model.h5       # Fine-tuned U-Net weights
├── requirements.txt        # Python Dependencies
├── sar_unet_diagram.png    # Architecture Visualization
├── SAR_Notebook.ipynb      # Training & Evaluation Logic
└── README.md               # Project Documentation