# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

from pipeline.inference import get_predictor
from pipeline.tess_pipeline import (
    download_lc,
    sigma_clip_detrend,
    run_tls,
    phase_fold_bin,
    build_scalar_features,
    generate_batman_overlay,
)

# 1. Page Configuration & Theme
st.set_page_config(page_title="TRANSIT — Exoplanet Detector", page_icon="🪐", layout="wide")

# Custom CSS for space theme & cinematic dashboard UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .stApp {
        background-color: #08090d;
        background-image: radial-gradient(circle at 10% 20%, rgba(31, 40, 51, 0.3) 0%, transparent 50%), 
                          radial-gradient(circle at 90% 80%, rgba(138, 43, 226, 0.1) 0%, transparent 50%);
        color: #c5c6c7;
    }
    
    /* Cinematic Header */
    .hero-container {
        text-align: center;
        padding: 20px 0 10px 0;
        background: rgba(15, 23, 42, 0.4);
        border-radius: 20px;
        border: 1px solid rgba(102, 252, 241, 0.05);
        margin-bottom: 25px;
    }
    .cinematic-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #66fcf1, #00d2ff, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }
    .cinematic-subtitle {
        font-size: 1.1rem;
        text-align: center;
        color: #66fcf1;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-weight: 500;
        margin-bottom: 10px;
    }
    
    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(20, 26, 38, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 252, 241, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(102, 252, 241, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(102, 252, 241, 0.05);
    }
    
    /* Interactive Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: rgba(13, 17, 23, 0.8);
        padding: 8px 16px;
        border-radius: 14px;
        border: 1px solid rgba(102, 252, 241, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-radius: 8px;
        color: #8892b0;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.3s;
        padding: 0 20px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #66fcf1;
        background-color: rgba(102, 252, 241, 0.05);
    }
    .stTabs [aria-selected="true"] {
        color: #66fcf1 !important;
        background-color: rgba(102, 252, 241, 0.12) !important;
        border: 1px solid rgba(102, 252, 241, 0.25) !important;
    }
    
    /* Metrics display */
    .metric-box {
        text-align: center;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        margin: 5px;
    }
    
    /* Custom divider */
    .glow-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 252, 241, 0.5), transparent);
        margin: 20px 0;
    }
    
    /* Social Links */
    .social-container {
        display: flex;
        gap: 15px;
        margin-top: 15px;
    }
    .social-link {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #8892b0;
        text-decoration: none;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    .social-link:hover {
        color: #66fcf1;
        transform: translateY(-1px);
    }
    .social-link svg {
        transition: transform 0.2s ease;
    }
    .social-link:hover svg {
        transform: scale(1.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to style Matplotlib plots for dark/space theme
def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, color="#66fcf1", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color="#8892b0", fontsize=8)
    ax.set_ylabel(ylabel, color="#8892b0", fontsize=8)
    ax.tick_params(colors="#8892b0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2a3142')
    ax.grid(True, linestyle=":", alpha=0.1, color="#ffffff")
    ax.set_facecolor('#0d1117')

# 2. Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/planet.png", width=70)
    st.markdown("### 🪐 TRANSIT Pipeline")
    st.caption("BAH 2026 · Team Bharat Ka Khazana")
    
    st.markdown("---")
    st.subheader("📁 Notebook Summary")
    with st.expander("📝 NB-01: Data & EDA"):
        st.write("""
        - MAST light curve downloads.
        - MAD-based sigma clipping.
        - Biweight detrending via Wotan.
        - Transit Least Squares (TLS) search.
        - Profile phase-folding & binning.
        """)
    with st.expander("🧠 NB-02: CNN-Transformer"):
        st.write("""
        - Preprocessing & scaled features.
        - PyTorch model training.
        - Dual-head: Classifier & Regressor.
        - 5-Fold Cross-Validation.
        - Attention heatmaps extraction.
        """)
    st.markdown("---")
    st.caption("Developed by Kritika Benjwal, Sarthak Gupta & Chaitanya Yadav.")

# 3. Cinematic App Header
st.markdown("""
<div class="hero-container">
    <div class="cinematic-subtitle">BAH 2026 · Problem Statement 7 · Team Bharat Ka Khazana</div>
    <div class="cinematic-title">🪐 TRANSIT — AI-Powered Exoplanet Detector</div>
</div>
""", unsafe_allow_html=True)

# 4. Tab Navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Welcome & Intro",
    "🪐 Transit Detector Dashboard",
    "📁 Notebooks & Methodology",
    "🔗 Project Resources",
    "👥 Meet the Team",
    "📝 Project Report"
])

# ================= TAB 1: WELCOME & INTRO =================
with tab1:
    col_intro_left, col_intro_right = st.columns([3, 2])
    
    with col_intro_left:
        st.markdown("""
        <div class="glass-card">
            <h3>🌌 The Exoplanet Transit Challenge</h3>
            <p>Identifying the tiny periodic dips in brightness caused by a planet transiting its host star is one of the most challenging tasks in modern astrophysics. Stellar light curves present in crowded fields are heavily corrupted by:</p>
            <ul>
                <li><b>Stellar Blending</b>: Light contamination from nearby foreground or background stars.</li>
                <li><b>Intrinsic Noise</b>: Detector anomalies, cosmic ray hits, and instrument jitter.</li>
                <li><b>Astrophysical Mimics</b>: Eclipsing Binary systems (EB) and starspots which look identical to transits.</li>
            </ul>
            <p>The <b>TRANSIT</b> pipeline tackles this by combining robust physical template matching (TLS) with a hybrid <b>CNN-Transformer Deep Learning model</b> to filter noise and classify signals.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
            <h3>🏆 Hackathon & Problem Statement Info</h3>
            <p><b>Event</b>: BAH 2026</p>
            <p><b>Problem Statement 7</b>: AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves</p>
            <p><b>Objective</b>: Build an automated data pipeline to ingest, clean, detect periodic signals, estimate parameters, and classify exoplanet transit features with high confidence.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_intro_right:
        # Space transit vector illustration
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <svg viewBox="0 0 800 500" width="100%" height="250" style="background-color: #0b0c10; border-radius: 16px; border: 1px solid rgba(102, 252, 241, 0.15); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);">
              <defs>
                <radialGradient id="starGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stop-color="#ffffff" stop-opacity="1" />
                  <stop offset="30%" stop-color="#66fcf1" stop-opacity="0.9" />
                  <stop offset="70%" stop-color="#4e54c8" stop-opacity="0.3" />
                  <stop offset="100%" stop-color="#4e54c8" stop-opacity="0" />
                </radialGradient>
                <linearGradient id="orbitGlow" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#66fcf1" stop-opacity="0.1" />
                  <stop offset="50%" stop-color="#66fcf1" stop-opacity="0.8" />
                  <stop offset="100%" stop-color="#66fcf1" stop-opacity="0.1" />
                </linearGradient>
              </defs>
              <!-- Grid -->
              <g stroke="#ffffff" stroke-opacity="0.03" stroke-width="0.5">
                <path d="M 0,100 L 800,100 M 0,200 L 800,200 M 0,300 L 800,300 M 0,400 L 800,400" />
                <path d="M 150,0 L 150,500 M 300,0 L 300,500 M 450,0 L 450,500 M 600,0 L 600,500" />
              </g>
              <!-- Orbit -->
              <ellipse cx="400" cy="220" rx="320" ry="50" fill="none" stroke="url(#orbitGlow)" stroke-width="2" stroke-dasharray="8 6" />
              <!-- Host Star -->
              <circle cx="400" cy="220" r="70" fill="url(#starGlow)" />
              <!-- Transiting Planet -->
              <circle cx="200" cy="204" r="16" fill="#050608" stroke="#66fcf1" stroke-width="2.5" />
              <!-- Scan lines -->
              <line x1="200" y1="50" x2="200" y2="400" stroke="#66fcf1" stroke-opacity="0.4" stroke-width="1.5" stroke-dasharray="3 3" />
              <circle cx="200" cy="400" r="6" fill="#66fcf1" />
              <!-- Light curve plot at bottom -->
              <path d="M 50,400 L 140,400 Q 170,400 185,440 L 215,440 Q 230,400 320,400 L 750,400" fill="none" stroke="#66fcf1" stroke-width="3" />
              <text x="420" y="120" fill="#ffffff" font-family="'Space Grotesk', sans-serif" font-size="14" letter-spacing="2" font-weight="bold">HOST STAR</text>
              <text x="75" y="170" fill="#66fcf1" font-family="'Space Grotesk', sans-serif" font-size="14" letter-spacing="2" font-weight="bold">TRANSITING PLANET</text>
              <text x="215" y="470" fill="#8892b0" font-family="'Space Grotesk', sans-serif" font-size="12">TRANSIT DEPTH (ppm)</text>
            </svg>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **Quick Start Guide**: Navigate to the **Transit Detector Dashboard** tab, type in a target star's TIC ID (e.g. 25375553), and run the pipeline to see it in action.")

# ================= TAB 2: DETECTOR DASHBOARD =================
with tab2:
    st.markdown("""
    <div class="glass-card">
        <h3>🪐 Target Acquisition System</h3>
        <p style="color:#8892b0;">Provide a TESS Input Catalog (TIC) ID to pull raw light curves from MAST, clean them, find periodic candidates, and run neural network classification.</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_suggestions = st.columns([2, 3])
    with col_input:
        tic_input = st.text_input("Enter TESS Input Catalog (TIC) ID", value="25375553", placeholder="e.g. 25375553")
        run_btn = st.button("Run TRANSIT Pipeline", type="primary")
    with col_suggestions:
        st.markdown("""
        **💡 Suggested Test Targets for Judges:**
        * **`25375553`**: Standard Planet Candidate (Transit Class - strong dip)
        * **`100100727`**: WASP-18b Hot Jupiter (Transit Class - deep, short-period dip)
        * **`150428135`**: Eclipsing Binary (EB Class - deep alternating eclipses)
        """)

    if run_btn and tic_input.strip():
        raw_input = tic_input.strip()
        used_fallback = False
        try:
            with st.status("🚀 Processing Light Curve...", expanded=True) as status:
                st.write("🛰️ Querying MAST archive and downloading TESS data...")
                try:
                    tic_id = int(raw_input)
                    t_raw, f_raw, sector = download_lc(tic_id)
                except Exception as download_err:
                    st.warning(f"⚠️ Target TIC {raw_input} could not be downloaded ({download_err}). Falling back to planet candidate TIC 25375553 for demonstration.")
                    used_fallback = True
                    tic_id = 25375553
                    t_raw, f_raw, sector = download_lc(25375553)
                    
                st.write(f"✅ Data acquired. Sector: {sector or 'Unknown'}. Points: {len(t_raw):,}")
                
                st.write("🧹 Cleaning data: Running 3-step iterative MAD sigma-clipping...")
                st.write("📈 Flattening: Fitting wotan biweight detrending template...")
                clean = sigma_clip_detrend(t_raw, f_raw)
                
                st.write("🔍 Transit Search: Scanning period spectrum with Transit Least Squares (TLS)...")
                result_tls, depth_ppm, sec_depth_ppm, has_sec = run_tls(
                    clean["t_clean"], clean["flux_frac"], period_min=0.5, period_max=20.0
                )
                st.write(f"📊 Periodogram Peak Identified: Period = {result_tls.period:.4f} d, SDE = {result_tls.SDE:.2f}")
                
                st.write("🌀 Fold & Bin: Phase-folding and binning transit profile into 200-bin vector...")
                profile = phase_fold_bin(clean["t_clean"], clean["flux_frac"], result_tls.period, result_tls.T0, 200)
                scalars = build_scalar_features(result_tls, depth_ppm, sec_depth_ppm, has_sec)
                
                st.write("🛠️ Physical Modeling: Generating analytical batman orbit overlay...")
                bm_phase, bm_flux = generate_batman_overlay(result_tls.period, depth_ppm, result_tls.duration * 24, 200)
                
                data = {
                    "sector": sector,
                    "clean": clean,
                    "tls_result": result_tls,
                    "depth_ppm": depth_ppm,
                    "secondary_depth_ppm": sec_depth_ppm,
                    "has_secondary": has_sec,
                    "profile": profile,
                    "scalars": scalars,
                    "batman_phase": bm_phase,
                    "batman_flux": bm_flux,
                }
                
                st.write("🧠 AI Inference: Deploying CNN-Transformer forward pass...")
                predictor = get_predictor()
                result = predictor.predict(data["profile"], data["scalars"])
                
                status.update(label="✅ Pipeline Completed Successfully!", state="complete", expanded=False)

            if used_fallback:
                st.info(f"💡 Note: Displaying results for fallback candidate TIC 25375553 because the requested target TIC {raw_input} has no available observations in MAST.")

            # Render styled metric rows
            st.markdown("#### 🚀 Detection Summary Results")
            
            # AI Results Row
            st.markdown("##### 1. AI Classification & Parameter Refinement")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Predicted Class</div><div class="metric-value">{result["class_name"]}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-box"><div class="metric-label">AI Confidence</div><div class="metric-value">{result["confidence"]:.2%}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-box"><div class="metric-label">AI Refined Period</div><div class="metric-value">{result["period_days"]:.4f} d</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-box"><div class="metric-label">AI Refined Depth</div><div class="metric-value">{result["depth_ppm"]:.1f} ppm</div></div>', unsafe_allow_html=True)
            with c5:
                st.markdown(f'<div class="metric-box"><div class="metric-label">AI Refined Duration</div><div class="metric-value">{result["duration_hr"]:.3f} hr</div></div>', unsafe_allow_html=True)

            # Physical Results Row
            st.markdown("##### 2. Physical Template Fit (Transit Least Squares)")
            p1, p2, p3, p4, p5 = st.columns(5)
            
            tls_p = data["tls_result"].period
            tls_d = (1.0 - data["tls_result"].depth) * 1e6
            tls_dur = data["tls_result"].duration * 24.0
            tls_sde = data["tls_result"].SDE
            tls_snr = data["tls_result"].snr

            with p1:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Significance (SDE)</div><div class="metric-value">{tls_sde:.2f}</div></div>', unsafe_allow_html=True)
            with p2:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Signal SNR</div><div class="metric-value">{tls_snr:.2f}</div></div>', unsafe_allow_html=True)
            with p3:
                st.markdown(f'<div class="metric-box"><div class="metric-label">TLS Period</div><div class="metric-value">{tls_p:.4f} d</div></div>', unsafe_allow_html=True)
            with p4:
                st.markdown(f'<div class="metric-box"><div class="metric-label">TLS Depth</div><div class="metric-value">{tls_d:.1f} ppm</div></div>', unsafe_allow_html=True)
            with p5:
                st.markdown(f'<div class="metric-box"><div class="metric-label">TLS Duration</div><div class="metric-value">{tls_dur:.3f} hr</div></div>', unsafe_allow_html=True)

            st.markdown("##### 🎯 Classification Probability Breakdown")
            prob_cols = st.columns(3)
            for i, cls in enumerate(["Transit", "EB", "Other"]):
                val = result["probs"].get(cls, 0.0)
                with prob_cols[i]:
                    st.markdown(f"**{cls}**")
                    st.progress(float(val))
                    st.caption(f"Confidence: {val:.2%}")
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

            # Plots
            col_plot1, col_plot2 = st.columns(2)

            with col_plot1:
                st.markdown("#### Light Curve Processing Panels")
                
                # 1. Raw Light Curve Plot
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_alpha(0.0) # transparent background
                ax.plot(data["clean"]["t_raw"], data["clean"]["f_raw"], lw=0.4, color="#888888", alpha=0.8)
                ax.axhline(0, color="w", lw=0.5, ls="--", alpha=0.3)
                style_ax(ax, "1. Raw Light Curve (Flux vs Time)", "Time (BTJD)", "Flux (ppt)")
                st.pyplot(fig)

                # 2. Detrended Light Curve Plot
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_alpha(0.0)
                ax.plot(data["clean"]["t_clean"], data["clean"]["flat_ppt"], lw=0.4, color="#66fcf1")
                ax.axhline(0, color="w", lw=0.5, ls="--", alpha=0.3)
                style_ax(ax, "2. Detrended Light Curve (wotan biweight)", "Time (BTJD)", "Residual Flux (ppt)")
                st.pyplot(fig)

            with col_plot2:
                st.markdown("#### Signal Fit & Attention Analytics")

                # 3. Attention Heatmap Plot
                attn_map = result["attention"].mean(axis=0)
                seq_phase = np.linspace(0, 1, len(attn_map))
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_alpha(0.0)
                ax.plot(seq_phase, attn_map, color="#a855f7", lw=1.5)
                ax.fill_between(seq_phase, 0, attn_map, alpha=0.3, color="#a855f7")
                style_ax(ax, "3. CNN-Transformer Attention weights Map", "Phase", "Attention Weight")
                st.pyplot(fig)

                # 4. Phase-Folded Fit Plot
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_alpha(0.0)
                phase_x = np.linspace(0, 1, len(data["profile"]))
                ax.scatter(phase_x, data["profile"], s=8, color="#00d2ff", label="Data Profile", alpha=0.5)
                ax.plot(data["batman_phase"], data["batman_flux"], color="#ffb703", lw=1.8, label="batman fit")
                style_ax(ax, "4. Phase-Folded Fit Comparison", "Phase", "Normalized Flux (σ)")
                ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#2a3142', labelcolor='white')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.info("Common causes: invalid TIC ID, no 2-min SPOC data for this target, or MAST is temporarily slow. Try TIC 25375553.")

# ================= TAB 3: NOTEBOOKS & METHODOLOGY =================
with tab3:
    st.markdown("""
    <div class="glass-card">
        <h3>📁 Notebook Evidence & Analytical Validation</h3>
        <p style="color:#8892b0;">Examine the evidence files and generated plots from each stage of our development pipeline.</p>
    </div>
    """, unsafe_allow_html=True)
    
    nb_subtab1, nb_subtab2 = st.tabs(["📝 NB-01: Data & Preprocessing", "🧠 NB-02: CNN-Transformer Model"])
    
    with nb_subtab1:
        st.markdown("#### NB-01: Data Ingestion, EDA, Detrending & Transit Search")
        st.markdown("""
        This notebook handles the initial data retrieval and physics-based signal extraction. 
        It downloads high-cadence curves, performs outliers clipping, implements `wotan` detrending, runs **Transit Least Squares (TLS)**, and aggregates features.
        """)
        
        col_nb1_1, col_nb1_2 = st.columns(2)
        with col_nb1_1:
            st.image("results_eda/__results___files/__results___8_1.png", caption="Fig 1: Raw Survey Class Distribution and Noise analysis (NB-01 Cell 8)")
            st.image("results_eda/__results___files/__results___12_1.png", caption="Fig 3: Transit Profile Phase-Folding & Binning (NB-01 Cell 12)")
        with col_nb1_2:
            st.image("results_eda/__results___files/__results___9_1.png", caption="Fig 2: Iterative Sigma-Clipping and wotan Detrending (NB-01 Cell 9)")
            st.image("results_eda/__results___files/__results___14_1.png", caption="Fig 4: Random Forest Classical Baseline Feature Importance (NB-01 Cell 14)")
            
    with nb_subtab2:
        st.markdown("#### NB-02: Deep Learning Model Training & Evaluation")
        st.markdown("""
        This notebook defines and trains the hybrid **CNN-Transformer model** in PyTorch. 
        Features are cross-validated across 5 folds. The network combines 1D CNNs to parse shapes with Multi-Head Attention blocks to evaluate dependencies.
        """)
        
        col_nb2_1, col_nb2_2 = st.columns(2)
        with col_nb2_1:
            st.image("models/transit_model/cv_results.png", caption="Fig 5: 5-Fold Cross Validation F1-scores by fold (NB-02)")
            st.image("models/transit_model/confusion_matrix_test.png", caption="Fig 7: Confusion Matrix on test dataset (NB-02)")
            st.image("models/transit_model/attention_heatmaps.png", caption="Fig 9: Neural Network Attention Mapping activations (NB-02)")
        with col_nb2_2:
            st.image("models/transit_model/training_curves.png", caption="Fig 6: Training and Validation Loss / Accuracy Curves (NB-02)")
            st.image("models/transit_model/roc_curves_test.png", caption="Fig 8: ROC-AUC curves by class on test set (NB-02)")
            st.image("models/transit_model/attention_matrix_example.png", caption="Fig 10: Multihead Self-Attention Weight Matrix Example (NB-02)")

# ================= TAB 4: PROJECT RESOURCES =================
with tab4:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <h3>🔗 Project Links & Resources</h3>
        <p style="color:#8892b0;">Find codebases, notebooks, and models related to this project below.</p>
        <div class="glow-divider"></div>
        <div style="display: flex; flex-direction: column; gap: 15px; align-items: center; justify-content: center; padding: 20px;">
            <a href="https://www.kaggle.com/code/kritikabenjwal/transit-eda-detection" target="_blank" style="text-decoration:none;">
                <button style="background-color: #20bead; color: white; border: none; padding: 12px 35px; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 1rem; width: 350px;">📘 Kaggle Notebook 1: EDA & Detection</button>
            </a>
            <a href="https://www.kaggle.com/code/kritikabenjwal/transit-model-training" target="_blank" style="text-decoration:none;">
                <button style="background-color: #20bead; color: white; border: none; padding: 12px 35px; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 1rem; width: 350px;">📘 Kaggle Notebook 2: Model Training</button>
            </a>
            <a href="https://github.com/Kritika11052005/TRANSIT-AI-Powered-Exoplanet-Detector" target="_blank" style="text-decoration:none;">
                <button style="background-color: #24292e; color: white; border: 1px solid #c5c6c7; padding: 12px 35px; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 1rem; width: 350px;">🐙 GitHub Code Repository</button>
            </a>
            <a href="https://huggingface.co/spaces/Kritzzz11/transit-exoplanet-detector" target="_blank" style="text-decoration:none;">
                <button style="background-color: #ffb703; color: black; border: none; padding: 12px 35px; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 1rem; width: 350px;">🤗 Hugging Face Spaces App</button>
            </a>
            <a href="https://drive.google.com/drive/folders/your-folder-id" target="_blank" style="text-decoration:none;">
                <button style="background-color: #0F9D58; color: white; border: none; padding: 12px 35px; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 1rem; width: 350px;">📁 Google Drive: Project Report & Media</button>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================= TAB 5: MEET THE TEAM =================
with tab5:
    st.markdown("### 👥 Meet Team Bharat Ka Khazana")
    
    col_team1, col_team2, col_team3 = st.columns(3)
    
    with col_team1:
        st.markdown("""
        <div class="glass-card">
            <div style="display: flex; align-items: center; gap: 20px;">
                <img src="https://img.icons8.com/color/96/000000/astronaut.png" width="64">
                <div>
                    <h4 style="margin: 0; color: #66fcf1;">Kritika Benjwal</h4>
                    <p style="margin: 0; font-size: 0.9rem; color: #8892b0;">Lead AI Engineer & Pipeline Architect</p>
                </div>
            </div>
            <div class="glow-divider" style="margin: 15px 0;"></div>
            <p style="font-size: 0.95rem; line-height: 1.6; min-height: 100px;">
                Engineered the core PyTorch model integrating dual convolutional filters and multi-head attention mechanisms. Built the parameter regression heads and compiled automated deployment pipelines using Hugging Face Docker infrastructure.
            </p>
            <div class="glow-divider" style="margin: 15px 0;"></div>
            <div class="social-container">
                <a class="social-link" href="https://github.com/Kritika11052005" target="_blank">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/></svg>
                    GitHub
                </a>
                <a class="social-link" href="https://www.linkedin.com/in/kritika-benjwal" target="_blank">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854zm4.943 12.248V6.169H2.542v7.225zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248S2.4 4.148 2.4 4.856c0 .694.521 1.248 1.327 1.248zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016l.016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225z"/></svg>
                    LinkedIn
                </a>
                <a class="social-link" href="mailto:ananya.benjwal@gmail.com">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M.05 3.555A2 2 0 0 1 2 2h12a2 2 0 0 1 1.95 1.555L8 8.414zM0 4.697v7.104l5.803-3.558zM6.761 8.83l-6.57 4.027A2 2 0 0 0 2 14h12a2 2 0 0 0 1.808-1.144l-6.57-4.027L8 9.586zm3.436-.59 5.803 3.557V4.697z"/></svg>
                    Email
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_team2:
        st.markdown("""
        <div class="glass-card">
            <div style="display: flex; align-items: center; gap: 20px;">
                <img src="https://img.icons8.com/color/96/000000/space-shuttle.png" width="64">
                <div>
                    <h4 style="margin: 0; color: #66fcf1;">Sarthak Gupta</h4>
                    <p style="margin: 0; font-size: 0.9rem; color: #8892b0;">Astrophysics & Signal Processing Lead</p>
                </div>
            </div>
            <div class="glow-divider" style="margin: 15px 0;"></div>
            <p style="font-size: 0.95rem; line-height: 1.6; min-height: 100px;">
                Led physical signal pre-processing, detrending configurations (Wotan biweight optimization), and Transit Least Squares template alignment. Developed error budgets and performed exploratory data analysis (EDA) to map transit classifications.
            </p>
            <div class="glow-divider" style="margin: 15px 0;"></div>
            <div class="social-container">
                <a class="social-link" href="https://github.com/SarthakG1801" target="_blank">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/></svg>
                    GitHub
                </a>
                <a class="social-link" href="https://www.linkedin.com/in/sarthakgupta1801" target="_blank">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854zm4.943 12.248V6.169H2.542v7.225zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248S2.4 4.148 2.4 4.856c0 .694.521 1.248 1.327 1.248zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016l.016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225z"/></svg>
                    LinkedIn
                </a>
                <a class="social-link" href="mailto:sarthakgupta1971@gmail.com">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M.05 3.555A2 2 0 0 1 2 2h12a2 2 0 0 1 1.95 1.555L8 8.414zM0 4.697v7.104l5.803-3.558zM6.761 8.83l-6.57 4.027A2 2 0 0 0 2 14h12a2 2 0 0 0 1.808-1.144l-6.57-4.027L8 9.586zm3.436-.59 5.803 3.557V4.697z"/></svg>
                    Email
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_team3:
        st.markdown("""
        <div class="glass-card">
            <div style="display: flex; align-items: center; gap: 20px;">
                <img src="https://img.icons8.com/color/96/000000/satellite.png" width="64">
                <div>
                    <h4 style="margin: 0; color: #66fcf1;">Chaitanya Yadav</h4>
                    <p style="margin: 0; font-size: 0.9rem; color: #8892b0;">Data Engineer & Deployment Specialist</p>
                </div>
            </div>
            <div class="glow-divider" style="margin: 15px 0;"></div>
            <p style="font-size: 0.95rem; line-height: 1.6; min-height: 100px;">
                Collaborated on data pipeline optimization, detrending workflows, and model inference integrations. Managed repository synchronization, environment staging, and Hugging Face container configurations to ensure seamless application hosting.
            </p>
            <div class="glow-divider" style="margin: 15px 0;"></div>
            <div class="social-container">
                <a class="social-link" href="https://github.com/chaitanyayad" target="_blank">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/></svg>
                    GitHub
                </a>
                <a class="social-link" href="https://www.linkedin.com/in/chaitanya-yadav-ba44503a9/" target="_blank">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854zm4.943 12.248V6.169H2.542v7.225zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248S2.4 4.148 2.4 4.856c0 .694.521 1.248 1.327 1.248zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016l.016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225z"/></svg>
                    LinkedIn
                </a>
                <a class="social-link" href="mailto:chaitanya.yad007@gmail.com">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M.05 3.555A2 2 0 0 1 2 2h12a2 2 0 0 1 1.95 1.555L8 8.414zM0 4.697v7.104l5.803-3.558zM6.761 8.83l-6.57 4.027A2 2 0 0 0 2 14h12a2 2 0 0 0 1.808-1.144l-6.57-4.027L8 9.586zm3.436-.59 5.803 3.557V4.697z"/></svg>
                    Email
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 6: PROJECT REPORT =================
with tab6:
    st.markdown("### 📝 Full Project Report")
    
    # Load report.md dynamically
    report_path = "report.md"
    report_loaded = False
    report_text = ""
    
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()
            report_loaded = True
        except Exception as e:
            report_text = f"Error reading report file: {e}"
    else:
        report_text = "Report file `report.md` not found in workspace."
        
    # Render report inside card container
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(report_text)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if report_loaded:
        st.download_button(
            label="⬇️ Download Report (Markdown)",
            data=report_text,
            file_name="transit_project_report.md",
            mime="text/markdown"
        )