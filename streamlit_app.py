import streamlit as st
import tensorflow as tf
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
import io

# --- Page Configuration ---
st.set_page_config(page_title="SAR Despeckling AI", layout="wide")
st.title("SAR Image Despeckling using U-Net")
st.write("Upload a noisy Synthetic Aperture Radar (SAR) .tiff image to remove speckle noise.")

# --- Define Custom Objects for Model Loading ---
@st.cache_resource
def load_sar_model(model_path):
    # Setting compile=False bypasses the need to load the custom loss functions.
    # Since we are only doing inference (prediction) and not training, 
    # the optimizer and loss functions are not required.
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Load the model into memory
try:
    # Ensure this matches the exact name of your uploaded weights file
    model = load_sar_model('best_sar_model.h5')
    st.sidebar.success("Model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading model. Check file path. Details: {e}")
    st.stop()

# --- Image Processing Helper ---
def process_uploaded_tiff(uploaded_file):
    # Use .getvalue() to extract the raw bytes from the Streamlit UploadedFile object
    with MemoryFile(uploaded_file.getvalue()) as memfile:
        with memfile.open() as src:
            img = src.read(1)
            
            # Normalize to 0-1 range to match the U-Net training pipeline
            img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
            
            # Resize to 128x128 and add the channel dimension
            img_resized = tf.image.resize(img_normalized[..., np.newaxis], [128, 128]).numpy()
            return img_resized

# --- Main Application Logic ---
uploaded_file = st.file_uploader("Choose a .tiff file", type=["tiff", "tif"])

if uploaded_file is not None:
    st.write("Image Processed")
    
    # Preprocess
    input_image = process_uploaded_tiff(uploaded_file)
    
    # Run Inference
    # The U-Net expects a batch dimension, so we add one: shape becomes (1, 128, 128, 1)
    input_batch = np.expand_dims(input_image, axis=0)
    prediction = model.predict(input_batch)
    
    # Format for visualization by removing the batch and channel dimensions
    denoised_image = prediction.squeeze()
    original_image = input_image.squeeze()
    
    # Calculate Residual (what the AI removed)
    residual = np.abs(original_image - denoised_image)
    
    # --- Display Results ---
    st.subheader("Analysis Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### 1. Noisy Input")
        st.image(original_image, clamp=True, use_container_width=True, output_format="PNG")
        
    with col2:
        st.write("### 2. Denoised Output")
        st.image(denoised_image, clamp=True, use_container_width=True, output_format="PNG")
        
    with col3:
        st.write("### 3. Residual Noise")
        st.image(residual, clamp=True, use_container_width=True, output_format="PNG")
        st.write("The extracted speckle pattern.")

    # Download Button for the cleaned image
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(denoised_image, cmap='gray')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    st.download_button(
        label="Download Cleaned Image",
        data=buf,
        file_name="denoised_sar.png",
        mime="image/png"
    )