"""
Live TESS pipeline: TIC ID -> cleaned light curve -> TLS -> phase-folded
profile + 8 scalar features, matching NB-01 Cells 6, 9, 10, 12 exactly.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import lightkurve as lk
from wotan import flatten
from transitleastsquares import transitleastsquares
import batman


def download_lc(tic_id, quality_bitmask="default"):
    # 1. Try TESS mission first (most specific to our CNN-Transformer model)
    sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
    
    # 2. If no TESS data, try any light curve at all (e.g. Kepler, K2)
    if len(sr) == 0:
        sr = lk.search_lightcurve(f"TIC {tic_id}")
        
    if len(sr) == 0:
        raise ValueError(f"No light curve found in MAST database for TIC {tic_id}")
        
    # Sort search results: prefer SPOC, then TESS-SPOC, then QLP, then others.
    # Also prefer shorter exposure times (smaller values) if available.
    def sort_key(res):
        author = str(res.author).upper()
        # Handle exptime attribute variations
        try:
            exptime = res.exptime.value if hasattr(res.exptime, "value") else float(res.exptime)
        except Exception:
            exptime = 9999.0
            
        author_rank = 0
        if "SPOC" in author and "TESS-SPOC" not in author:
            author_rank = 3
        elif "TESS-SPOC" in author:
            author_rank = 2
        elif "QLP" in author:
            author_rank = 1
        return (-author_rank, exptime)

    results_list = list(sr)
    results_list.sort(key=sort_key)
    best_result = results_list[0]
    
    # Download the best available light curve
    lc = best_result.download(quality_bitmask=quality_bitmask)
    
    # Determine the best flux column
    flux_column = "sap_flux"
    if "sap_flux" not in lc.columns:
        if "pdcsap_flux" in lc.columns:
            flux_column = "pdcsap_flux"
        elif "flux" in lc.columns:
            flux_column = "flux"
        else:
            for col in lc.columns:
                if col != "time" and np.issubdtype(lc[col].dtype, np.number):
                    flux_column = col
                    break
                    
    lc = best_result.download(flux_column=flux_column, quality_bitmask=quality_bitmask)
    t = lc.time.value
    f = lc.flux.value
    mask = np.isfinite(t) & np.isfinite(f)
    return t[mask], f[mask], lc.meta.get("SECTOR", None)


def sigma_clip_detrend(t, f, sigma_lower=4.0, sigma_upper=5.0,
                       window_length=0.5, break_tolerance=0.5):
    """Mirrors NB-01 Cell 9 exactly (minus the EB-specific window widening,
    since at inference time we don't know the class yet)."""
    med = np.median(f)
    f_norm = f / med
    raw_t, raw_f = t.copy(), f_norm.copy()   # keep for the "raw" panel

    for _ in range(3):
        residual = f_norm - np.median(f_norm)
        mad_val  = np.median(np.abs(residual)) * 1.4826
        keep     = (residual > -sigma_lower * mad_val) & (residual < sigma_upper * mad_val)
        t, f_norm = t[keep], f_norm[keep]

    flat, trend = flatten(
        t, f_norm, method="biweight",
        window_length=window_length, break_tolerance=break_tolerance,
        return_trend=True, robust=True,
    )
    flux_frac = flat   # ≈1.0, already in the right scale for TLS

    return {
        "t_raw": raw_t, "f_raw": (raw_f - 1.0) * 1e3,
        "t_clean": t, "flux_frac": flux_frac,
        "trend_ppt": (trend / np.median(trend) - 1.0) * 1e3,
        "flat_ppt": (flux_frac - 1.0) * 1e3,
    }


def run_tls(t, flux_frac, period_min=0.5, period_max=20.0):
    """Mirrors NB-01 Cell 10. Returns the TLS result object plus secondary
    eclipse features (FLAG-21/23)."""
    model = transitleastsquares(t, flux_frac)
    result = model.power(
        period_min=period_min, period_max=period_max,
        oversampling_factor=3, duration_grid_step=1.05,
        show_progress_bar=False,
    )

    assert result.depth < 1.0, "TLS depth should be fractional flux (<1.0)."
    true_depth_ppm = (1.0 - result.depth) * 1e6

    # Secondary eclipse check (FLAG-21/23)
    secondary_depth_ppm, has_secondary = 0.0, 0
    try:
        half_period = result.period / 2.0
        if half_period >= period_min:
            model2 = transitleastsquares(t, flux_frac)
            result2 = model2.power(
                period_min=max(period_min, half_period * 0.9),
                period_max=half_period * 1.1,
                oversampling_factor=3, duration_grid_step=1.05,
                show_progress_bar=False,
            )
            if result2.depth < 1.0:
                secondary_depth_ppm = (1.0 - result2.depth) * 1e6
                has_secondary = int(secondary_depth_ppm > 0.10 * true_depth_ppm and result2.SDE > 5.0)
    except Exception:
        pass

    return result, true_depth_ppm, secondary_depth_ppm, has_secondary


def phase_fold_bin(t, flux, period, t0, n_bins=200):
    """Mirrors NB-01 Cell 12's phase_fold_bin exactly."""
    phase  = ((t - t0) / period) % 1.0
    edges  = np.linspace(0, 1, n_bins + 1)
    binned = np.full(n_bins, np.nan)
    for i in range(n_bins):
        in_bin = (phase >= edges[i]) & (phase < edges[i + 1])
        if in_bin.sum() >= 2:
            binned[i] = np.median(flux[in_bin])
    nans = np.isnan(binned)
    if nans.sum() < n_bins * 0.5:
        x = np.arange(n_bins)
        binned[nans] = np.interp(x[nans], x[~nans], binned[~nans])
    else:
        binned = np.nan_to_num(binned, nan=0.0)
    std = np.std(binned)
    if std > 1e-9:
        binned = (binned - np.mean(binned)) / std
    return binned


def build_scalar_features(result, true_depth_ppm, secondary_depth_ppm, has_secondary):
    """Mirrors NB-01's 8-feature SCALAR_FEATS exactly, in the same order
    the model was trained on."""
    return np.array([
        result.SDE,
        result.snr,
        result.period,
        result.duration * 24,        # duration_hr
        result.transit_count,        # n_transits
        true_depth_ppm,
        secondary_depth_ppm,
        has_secondary,
    ], dtype=np.float32)


def generate_batman_overlay(period_days, depth_ppm, duration_hr, n_bins=200):
    """GAP FIX reuse: generates the batman model curve for the
    'data + model fit' panel, same logic as NB-02's physics-loss metric."""
    rp_over_rs = np.clip(np.sqrt(max(depth_ppm, 1e-6) / 1e6), 1e-4, 0.5)
    dur_days   = max(duration_hr / 24.0, 0.01)
    a_over_rs  = max(3.0, period_days / dur_days)

    params = batman.TransitParams()
    params.t0  = 0.5 * period_days
    params.per = period_days
    params.rp  = rp_over_rs
    params.a   = a_over_rs
    params.inc = 89.0
    params.ecc = 0.0
    params.w   = 90.0
    params.u   = [0.3, 0.2]
    params.limb_dark = "quadratic"

    phase_x = np.linspace(0, 1, n_bins)
    t_model = phase_x * period_days
    m = batman.TransitModel(params, t_model)
    flux = m.light_curve(params)

    std = flux.std()
    flux_norm = (flux - flux.mean()) / std if std > 1e-9 else flux
    return phase_x, flux_norm


def run_full_pipeline(tic_id, period_min=0.5, period_max=20.0, n_bins=200):
    """End-to-end: TIC ID -> everything the model needs + everything the UI needs."""
    t_raw, f_raw, sector = download_lc(tic_id)
    clean = sigma_clip_detrend(t_raw, f_raw)
    result, depth_ppm, sec_depth_ppm, has_sec = run_tls(
        clean["t_clean"], clean["flux_frac"], period_min, period_max
    )
    profile = phase_fold_bin(clean["t_clean"], clean["flux_frac"],
                             result.period, result.T0, n_bins)
    scalars = build_scalar_features(result, depth_ppm, sec_depth_ppm, has_sec)
    bm_phase, bm_flux = generate_batman_overlay(result.period, depth_ppm,
                                                 result.duration * 24, n_bins)

    return {
        "sector": sector,
        "clean": clean,
        "tls_result": result,
        "depth_ppm": depth_ppm,
        "secondary_depth_ppm": sec_depth_ppm,
        "has_secondary": has_sec,
        "profile": profile,
        "scalars": scalars,
        "batman_phase": bm_phase,
        "batman_flux": bm_flux,
    }