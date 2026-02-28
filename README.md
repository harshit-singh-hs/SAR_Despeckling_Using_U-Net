# 🛰️ SAR Image Despeckling using U-Net

An interactive web application powered by **TensorFlow** and **Streamlit** that utilizes deep learning to remove speckle noise from Synthetic Aperture Radar (SAR) imagery in real-time.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sar-despeckling.streamlit.app/)

## 🚀 Overview

Synthetic Aperture Radar (SAR) is essential for remote sensing due to its all-weather, day-and-night imaging capabilities. However, SAR images are inherently degraded by **speckle noise**, a granular interference that obscures fine details. 

This project implements a **U-Net Convolutional Neural Network** to effectively despeckle radar data while preserving critical structural edges.



## 🖥️ App Preview & Datasets

### 1. Mendeley SAR Data
* **Source:** [SAR despeckling filters dataset by Mendeley Data](https://data.mendeley.com/datasets/2xf5v5pwkr/1)
* **Description:** Then U-Net model trained on this dataset only.

<img width="1919" height="912" alt="image" src="https://github.com/user-attachments/assets/5ed6aee8-6768-4767-a77d-2169d55cd637" />

<img width="1918" height="912" alt="image" src="https://github.com/user-attachments/assets/09136a33-5679-4d7c-aa62-11d2cd96c033" />

---

### 2. Chandrayaan-2 DFSAR Dataset
* **Source:** ISRO Pradan / Chandrayaan-2 DFSAR
* **Target File:** *ch2_sar_ncxl_20201003t155249412_d_sri_in_fp_xx_d18*
* **Description:** High-resolution orbital radar data.
<img width="1919" height="907" alt="image" src="https://github.com/user-attachments/assets/07c8be93-a636-45fe-a12b-0ca73983f838" />

<img width="1917" height="906" alt="image" src="https://github.com/user-attachments/assets/50465274-e891-4abf-8bd3-d53cc7c6c99f" />


---

## ✨ Features

* **Real-time Deep Inference:** Upload `.tiff` files and process them instantly through a fine-tuned U-Net model.
* **Advanced Loss Function:** Optimized using a composite of **SSIM**, **MAE**, and **Total Variation (TV)** loss to prevent blurring.
* **Residual Analysis:** Visualizes the "speckle pattern" removed by the AI to verify data preservation.
* **Output IMage:** Download processed images in .png format directly.

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
