# AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves
**Project Report**  
*BAH 2026 · Problem Statement 7 · Team Bharat Ka Khazana*

---

## 1. Executive Summary & Objectives
Exoplanet transit detection requires identifying minuscule periodic brightness dips in stellar light curves. For stars in crowded fields, these dips are heavily obscured by instrumental noise, stellar spots, eclipsing binary systems, and blending from nearby foreground/background sources. 

The objective of this project is to develop and deploy an automated, AI-driven data analysis pipeline (**TRANSIT**) capable of:
1. Downloading and cleaning raw TESS light curve data.
2. Identifying periodic dips using transit search algorithms.
3. Classifying dips into three categories: **Transit** (planetary candidates), **EB** (Eclipsing Binaries), and **Other** (stellar activity, noise, or blended background sources).
4. Estimating physical transit parameters (period, depth, duration) and significance levels (SDE, SNR).
5. Exposing the pipeline via an interactive web interface on Hugging Face Spaces.

---

## 2. Pipeline Methodology
The pipeline consists of a sequential flow from raw astronomical data to deep learning classification and parameter refinement:

```
[ TESS TIC ID ] 
       │
       ▼ (lightkurve)
[ Raw Light Curve ]
       │
       ▼ (wotan biweight detrending + MAD-based sigma clipping)
[ Cleaned / Detrended Light Curve ]
       │
       ▼ (Transit Least Squares - TLS)
[ Periodic Dip Detection & Parameter Extraction ] ──► [ SNR & SDE Significance ]
       │
 ┌─────┴────────────────────────┐
 │                              │
 ▼ (Phase-Folding & Binning)    ▼ (Feature Vector: SDE, SNR, Period, Depth...)
[ Phase-Folded Profile (200,) ] [ Scalar Features (8,) ]
       │                              │
       └──────────────┬───────────────┘
                      ▼
        [ CNN-Transformer PyTorch Model ]
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  [ Classification ]          [ Parameter Refinement ]
  - Class Name (Transit/EB/Other) - Refined Period, Depth, Duration
  - Softmax Confidence        - Analytical batman fit comparison
```

### A. Data Retrieval & Preprocessing
- **Data Source**: Using `lightkurve`, the pipeline queries the MAST archive for TESS high-cadence (2-minute SPOC) data using a star's TESS Input Catalog (TIC) ID.
- **Cleaning & Detrending**: 
  - Systematic outliers and instrumental anomalies are removed using a 3-step iterative **median absolute deviation (MAD)** sigma clipping method (4.0$\sigma$ lower, 5.0$\sigma$ upper bounds).
  - Stellar variability (e.g., starspots or rotation) is detrended using a robust **biweight filter** via the `wotan` package with a default window length of 0.5 days. This preserves transit features while flattening background trends.

### B. Periodic Dip Search (TLS)
Instead of the standard Box Least Squares (BLS) algorithm which fits a square box, the pipeline utilizes **Transit Least Squares (TLS)**. TLS searches for periodic dips by fitting a physical transit template that accounts for stellar limb-darkening and ingress/egress profiles. This yields a much higher sensitivity for shallow exoplanetary transits in noisy regimes and produces:
- **SDE (Signal Detection Efficiency)**: A metric indicating the significance of the detection peak in the periodogram.
- **SNR (Signal-to-Noise Ratio)**: The strength of the dip signal relative to residual noise.
- **Transit Parameters**: Initial period, depth, and duration estimates.

### C. Hybrid CNN-Transformer Architecture
Dips detected by TLS are forwarded to a PyTorch neural network that combines convolutional and attention-based structures:
1. **CNN Branch**: Processes the 200-bin phase-folded transit profile to extract local spatial features (shape of the dip, asymmetry).
2. **Positional Encoding & Transformer Attention Block**: Applies Multi-Head Self-Attention on the CNN feature sequence. This captures global temporal dependencies across the transit profile, allowing the model to focus on critical sections of the curve (generating the attention maps displayed in the app).
3. **MLP Branch**: Encodes 8 key scalar features from the TLS results: SDE, SNR, period, duration, transit count, primary depth, secondary depth, and secondary eclipse flags.
4. **Fusion & Multi-Task Heads**: The profile features and scalar features are concatenated and fed into:
   - **Classification Head**: Classifies the signal into `Transit`, `EB`, or `Other` using a Softmax activation layer.
   - **Regression Head**: Outputs refined estimates of orbital period, transit depth (ppm), and transit duration (hours).

---

## 3. Assumptions Made
1. **Circular Orbits**: The transit search assumes circular orbits ($e=0$) to simplify the initial parameter space during TLS.
2. **Quadratic Limb Darkening**: In generating the validation overlays (using `batman`), a quadratic limb-darkening law is assumed with standard stellar coefficients ($u = [0.3, 0.2]$).
3. **SPOC Data Quality**: It is assumed that the MAST archive contains 2-minute SPOC cadence data for the target; targets without SPOC light curves will trigger fallback errors in the download module.

---

## 4. Tools & Libraries Used
- **Data Access**: `lightkurve` (queries MAST, handles metadata and target pixel files).
- **Signal Processing**: `wotan` (biweight detrending), `transitleastsquares` (TLS periodogram analysis).
- **Transit Modeling**: `batman-package` (analytical light curve calculation for visual validation).
- **Machine Learning**: `PyTorch` (neural network definition, inference), `scikit-learn` (input scaler preprocessing).
- **Visualization**: `matplotlib` (graph plotting).
- **Interface**: `Streamlit` (interactive UI), deployed using the **Hugging Face Docker SDK** for reliable C-extension builds.

---

## 5. Uncertainty & Significance Estimation
Exoplanet searches are highly prone to false positives. To quantify uncertainties, the pipeline implements three methods:
1. **Astrophysical Signal Significance (SDE & SNR)**:
   - **SDE**: Calculated via TLS. An $SDE > 7.0$ is generally accepted in exoplanet literature as a significant detection.
   - **SNR**: Measures the ratio of the transit depth to the standard deviation of out-of-transit data.
2. **Classification Confidence**:
   - The neural network's classification head uses a Softmax function, yielding a probability distribution over the three classes. The confidence score represents the model's posterior probability of the predicted label, indicating how distinct the dip's features are.
3. **Physical Parameter Refinement**:
   - The neural network's regression head acts as a learned corrector, taking the raw TLS parameters alongside the shape profiles and refining them. The comparison between the raw TLS output and the AI-refined output acts as a cross-verification check.
4. **Generalization Metrics (5-Fold Cross-Validation)**:
   - The model was trained using 5-fold cross-validation. This ensures that the reported training accuracy (~95% classification accuracy) generalizes across different sectors, limiting over-fitting.

---

## 6. Project Coverage Checklist vs. Requirements
- [x] **Identify datasets with periodic dips**: Implemented using TLS in `tess_pipeline.py`.
- [x] **Classification framework**: Hybrid CNN-Transformer in PyTorch classifying into `Transit`, `EB`, and `Other`.
- [x] **Apply classifier on science datasets**: Live query system in the app downloads and evaluates raw MAST data.
- [x] **Provide SNR/SDE significance**: Physical SDE and SNR metrics are extracted and displayed in the UI.
- [x] **Estimate transit parameters**: Period, Depth, and Duration are estimated both physically (TLS) and refined by the AI.
- [x] **TESS raw light curves downloaded from repository**: Integrated via standard MAST queries in `lightkurve`.
- [x] **Visualization**: Visualizes raw light curve, detrended curve, attention weights (Transformer heatmap), and data overlaid with a physical `batman` analytical model fit.
- [x] **Project Report**: Completed (this document, `report.md`).
