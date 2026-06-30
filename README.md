---
title: TRANSIT — Exoplanet Detector
emoji: 🪐
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🪐 TRANSIT — AI-Powered Exoplanet Transit Detection

A deep-learning pipeline designed for detecting and classifying exoplanetary transits from TESS (Transiting Exoplanet Survey Satellite) light curves.

This project is part of **BAH 2026 · Problem Statement 7 · Team Bharat Ka Khazana**.

## 🚀 Features

- **Automated Data Retrieval**: Downloads SPOC 2-minute cadence light curves directly from MAST using the TESS Input Catalog (TIC) ID.
- **Detrending & Cleaning**: Leverages `wotan` (biweight detrending) and iterative sigma clipping to remove stellar variability and instrument noise.
- **Transit Search**: Runs Transit Least Squares (TLS) to look for periodic transit events.
- **Hybrid Neural Network**: Classifies the light curves using a hybrid **CNN-Transformer** model (implemented in PyTorch).
- **Interactive Visualization**: Plots raw/detrended light curves, attention maps, and phase-folded fits with `batman` models.

## 🛠️ Hugging Face Spaces Deployment Instructions

To deploy this model to Hugging Face Spaces, follow the steps below:

### Step 1: Create a Space on Hugging Face
1. Log in to your [Hugging Face account](https://huggingface.co/).
2. Click on your profile picture in the top-right and select **New Space** (or go to [huggingface.co/new-space](https://huggingface.co/new-space)).
3. Set the details:
   - **Space name**: Choose a name (e.g., `transit-exoplanet-detector`).
   - **License**: Choose a license (e.g., `mit` or `apache-2.0`).
   - **SDK**: Select **Streamlit**.
   - **Space Hardware**: Keep it on the free **CPU basic** (since the model is light, ~790 KB, a CPU is more than enough).
   - **Visibility**: Set it to **Public** or **Private**.
4. Click **Create Space**.

### Step 2: Push your Code to the Space
You can upload the files in one of two ways:

#### Option A: Using Git CLI (Recommended)
In your terminal, navigate to your workspace directory (`c:\Users\HP\Downloads\TRANSIT`) and run:
```bash
# Initialize git if you haven't already
git init

# Add the Hugging Face Space repository as a remote
# (Replace USERNAME and SPACE_NAME with your actual Hugging Face details)
git remote add origin https://huggingface.co/spaces/USERNAME/SPACE_NAME

# Stage all files (make sure to include the models folder)
git add .

# Commit your changes
git commit -m "Initial commit for TRANSIT exoplanet detector space"

# Push to Hugging Face (it will ask for your Hugging Face username and password/Token)
# Note: Use your Hugging Face Access Token (with 'write' permission) as your password.
git branch -M main
git push -u origin main --force
```

#### Option B: Using the Hugging Face Web Interface
1. Open your created Space on Hugging Face.
2. Click on the **Files and versions** tab.
3. Click **Add file** -> **Upload files**.
4. Drag and drop the following files/folders from your local workspace:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `pipeline/` (include all folder contents)
   - `models/` (include the whole `transit_model` folder with weight/scalers files)
5. Add a commit message and click **Commit changes to main**.

### Step 3: Wait for Building
Hugging Face will automatically detect `requirements.txt` and start building the Space using the configuration in `README.md`. You can watch the build progress in the **Logs** tab. It will take a few minutes to install the dependencies and launch the app.
