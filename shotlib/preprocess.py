"""
Signal preprocessing utilities for shot data.

Includes baseline subtraction, smoothing, normalization, and time alignment.
"""

from enum import Enum
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d


class NormalizationMethod(str, Enum):
    """Normalization methods for signal data."""
    NONE = "none"
    ZSCORE = "zscore"
    DIVIDE_BY_PEAK = "divide_by_peak"


class AlignmentMethod(str, Enum):
    """Time alignment methods for multi-shot comparison."""
    NONE = "none"
    DETECTED_EVENT = "detected_event"
    CROSS_CORRELATION = "cross_correlation"


def subtract_baseline(
    data: np.ndarray,
    time: np.ndarray,
    pre_trigger_end: float = 0.0001,
) -> np.ndarray:
    """
    Subtract baseline using median of pre-trigger window.
    
    Args:
        data: Signal data array
        time: Time array
        pre_trigger_end: End time of pre-trigger window in seconds
    
    Returns:
        Baseline-subtracted signal
    """
    # Handle NaN values
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return data.copy()
    
    # Find pre-trigger region
    pre_mask = (time < pre_trigger_end) & valid_mask
    
    if np.sum(pre_mask) < 2:
        # Not enough pre-trigger data, use first 10% of valid data
        n_valid = np.sum(valid_mask)
        n_baseline = max(2, n_valid // 10)
        valid_indices = np.where(valid_mask)[0][:n_baseline]
        baseline = np.median(data[valid_indices])
    else:
        baseline = np.median(data[pre_mask])
    
    result = data.copy()
    result[valid_mask] = data[valid_mask] - baseline
    
    return result


def smooth_signal(
    data: np.ndarray,
    method: str = "savgol",
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Apply smoothing filter to signal.
    
    Args:
        data: Signal data array
        method: Smoothing method ('savgol' or 'moving_average')
        window_length: Filter window length (must be odd for savgol)
        polyorder: Polynomial order for Savitzky-Golay filter
    
    Returns:
        Smoothed signal
    """
    # Handle NaN values by interpolating first
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return data.copy()
    
    result = data.copy()
    
    # Simple handling: apply filter only to valid segments
    if method == "savgol":
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        if window_length > len(data):
            window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window_length < polyorder + 2:
            return result
        
        # Apply to valid data only
        if np.all(valid_mask):
            result = scipy_signal.savgol_filter(data, window_length, polyorder)
        else:
            # Interpolate NaN regions for filtering
            indices = np.arange(len(data))
            interp_data = np.interp(
                indices, indices[valid_mask], data[valid_mask]
            )
            smoothed = scipy_signal.savgol_filter(interp_data, window_length, polyorder)
            result[valid_mask] = smoothed[valid_mask]
    
    elif method == "moving_average":
        if np.all(valid_mask):
            result = uniform_filter1d(data, size=window_length, mode='nearest')
        else:
            indices = np.arange(len(data))
            interp_data = np.interp(
                indices, indices[valid_mask], data[valid_mask]
            )
            smoothed = uniform_filter1d(interp_data, size=window_length, mode='nearest')
            result[valid_mask] = smoothed[valid_mask]
    
    return result


def normalize_signal(
    data: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.NONE,
) -> np.ndarray:
    """
    Normalize signal data.
    
    Args:
        data: Signal data array
        method: Normalization method
    
    Returns:
        Normalized signal
    """
    if method == NormalizationMethod.NONE:
        return data.copy()
    
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return data.copy()
    
    result = data.copy()
    valid_data = data[valid_mask]
    
    if method == NormalizationMethod.ZSCORE:
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        if std > 1e-10:
            result[valid_mask] = (valid_data - mean) / std
        else:
            result[valid_mask] = valid_data - mean
    
    elif method == NormalizationMethod.DIVIDE_BY_PEAK:
        peak = np.max(np.abs(valid_data))
        if peak > 1e-10:
            result[valid_mask] = valid_data / peak
    
    return result


def detect_event_time(
    data: np.ndarray,
    time: np.ndarray,
    threshold_fraction: float = 0.1,
) -> Optional[float]:
    """
    Detect event onset time based on signal rise.
    
    Uses threshold crossing detection: finds first time the signal
    exceeds a fraction of its peak value.
    
    Args:
        data: Signal data array
        time: Time array
        threshold_fraction: Fraction of peak to use as threshold
    
    Returns:
        Detected event time, or None if detection fails
    """
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return None
    
    valid_data = data[valid_mask]
    valid_time = time[valid_mask]
    
    # Compute threshold
    peak = np.max(np.abs(valid_data))
    threshold = peak * threshold_fraction
    
    # Find first crossing
    above_threshold = np.abs(valid_data) > threshold
    crossings = np.where(above_threshold)[0]
    
    if len(crossings) > 0:
        return float(valid_time[crossings[0]])
    
    return None


def align_by_event(
    time: np.ndarray,
    data: np.ndarray,
    reference_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align signal by shifting time axis so event occurs at t=0.
    
    Args:
        time: Time array
        data: Signal data array
        reference_time: Time to align to t=0
    
    Returns:
        Tuple of (aligned_time, data)
    """
    return time - reference_time, data.copy()


def align_by_cross_correlation(
    time: np.ndarray,
    data: np.ndarray,
    reference_data: np.ndarray,
    reference_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align signal to reference using cross-correlation.
    
    Args:
        time: Time array for signal to align
        data: Signal data to align
        reference_data: Reference signal
        reference_time: Reference time array
    
    Returns:
        Tuple of (aligned_time, data)
    """
    # Handle NaN values
    valid_mask = ~np.isnan(data)
    ref_valid = ~np.isnan(reference_data)
    
    if not np.any(valid_mask) or not np.any(ref_valid):
        return time.copy(), data.copy()
    
    # Interpolate to common grid if needed
    if len(time) != len(reference_time):
        # Use shorter length
        n = min(len(time), len(reference_time))
        data_interp = data[:n]
        ref_interp = reference_data[:n]
    else:
        data_interp = data.copy()
        ref_interp = reference_data.copy()
    
    # Fill NaN with 0 for correlation
    data_filled = np.nan_to_num(data_interp, nan=0.0)
    ref_filled = np.nan_to_num(ref_interp, nan=0.0)
    
    # Compute cross-correlation
    correlation = np.correlate(data_filled, ref_filled, mode='full')
    
    # Find lag of maximum correlation
    lag_indices = np.arange(-len(ref_filled) + 1, len(data_filled))
    max_idx = np.argmax(correlation)
    lag = lag_indices[max_idx]
    
    # Compute time shift
    dt = np.median(np.diff(time)) if len(time) > 1 else 1e-6
    time_shift = lag * dt
    
    return time - time_shift, data.copy()


def preprocess_signal(
    time: np.ndarray,
    data: np.ndarray,
    baseline_subtract: bool = False,
    smooth: bool = False,
    smooth_method: str = "savgol",
    smooth_window: int = 11,
    normalization: NormalizationMethod = NormalizationMethod.NONE,
) -> np.ndarray:
    """
    Apply full preprocessing pipeline to a signal.
    
    Args:
        time: Time array
        data: Signal data array
        baseline_subtract: Whether to subtract baseline
        smooth: Whether to apply smoothing
        smooth_method: Smoothing method
        smooth_window: Smoothing window length
        normalization: Normalization method
    
    Returns:
        Preprocessed signal
    """
    result = data.copy()
    
    if baseline_subtract:
        result = subtract_baseline(result, time)
    
    if smooth:
        result = smooth_signal(result, method=smooth_method, window_length=smooth_window)
    
    result = normalize_signal(result, method=normalization)
    
    return result
