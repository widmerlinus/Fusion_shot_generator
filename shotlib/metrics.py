"""
Metrics computation for shot analysis.

Computes per-shot, per-channel metrics including peak values, timing,
spectral characteristics, and signal quality indicators.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal


@dataclass
class ChannelMetrics:
    """Computed metrics for a single channel."""
    peak: float
    min_val: float
    peak_to_peak: float
    time_of_peak: float
    auc: float
    rise_time_10_90: Optional[float]
    dominant_freq_hz: Optional[float]
    snr_estimate: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "peak": self.peak,
            "min": self.min_val,
            "peak_to_peak": self.peak_to_peak,
            "time_of_peak": self.time_of_peak,
            "auc": self.auc,
            "rise_time_10_90": self.rise_time_10_90,
            "dominant_freq_hz": self.dominant_freq_hz,
            "snr_estimate": self.snr_estimate,
        }


def compute_peak(data: np.ndarray) -> float:
    """Compute peak (maximum) value."""
    valid = data[~np.isnan(data)]
    return float(np.max(valid)) if len(valid) > 0 else np.nan


def compute_min(data: np.ndarray) -> float:
    """Compute minimum value."""
    valid = data[~np.isnan(data)]
    return float(np.min(valid)) if len(valid) > 0 else np.nan


def compute_peak_to_peak(data: np.ndarray) -> float:
    """Compute peak-to-peak value."""
    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        return np.nan
    return float(np.max(valid) - np.min(valid))


def compute_time_of_peak(data: np.ndarray, time: np.ndarray) -> float:
    """Compute time at which peak occurs."""
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return np.nan
    
    # Find index of maximum in valid data using only valid indices
    valid_indices = np.where(valid_mask)[0]
    valid_values = data[valid_mask]
    max_in_valid = np.argmax(valid_values)
    peak_idx = valid_indices[max_in_valid]
    
    return float(time[peak_idx])


def compute_auc(
    data: np.ndarray,
    time: np.ndarray,
    window_start: Optional[float] = None,
    window_end: Optional[float] = None,
) -> float:
    """
    Compute area under curve using trapezoidal integration.
    
    Args:
        data: Signal data
        time: Time array
        window_start: Start of integration window (None = start of data)
        window_end: End of integration window (None = end of data)
    
    Returns:
        Area under curve
    """
    # Apply window
    mask = np.ones(len(data), dtype=bool)
    if window_start is not None:
        mask &= time >= window_start
    if window_end is not None:
        mask &= time <= window_end
    
    t_win = time[mask]
    d_win = data[mask]
    
    # Handle NaN
    valid = ~np.isnan(d_win)
    if np.sum(valid) < 2:
        return np.nan
    
    # Use np.trapezoid (NumPy 2.0+) or fall back to np.trapz (older NumPy)
    try:
        return float(np.trapezoid(d_win[valid], t_win[valid]))
    except AttributeError:
        return float(np.trapz(d_win[valid], t_win[valid]))


def compute_rise_time_10_90(data: np.ndarray, time: np.ndarray) -> Optional[float]:
    """
    Compute 10-90% rise time for signals with clear rise phase.
    
    Returns None if rise time cannot be determined.
    """
    valid_mask = ~np.isnan(data)
    if np.sum(valid_mask) < 10:
        return None
    
    valid_data = data[valid_mask]
    valid_time = time[valid_mask]
    
    # Find baseline and peak
    # Use first 10% as baseline estimate
    n_baseline = max(1, len(valid_data) // 10)
    baseline = np.mean(valid_data[:n_baseline])
    peak = np.max(valid_data)
    
    # Compute thresholds
    amplitude = peak - baseline
    if amplitude <= 0:
        return None
    
    thresh_10 = baseline + 0.1 * amplitude
    thresh_90 = baseline + 0.9 * amplitude
    
    # Find crossing times
    above_10 = valid_data > thresh_10
    above_90 = valid_data > thresh_90
    
    if not np.any(above_10) or not np.any(above_90):
        return None
    
    t_10 = valid_time[np.argmax(above_10)]
    t_90 = valid_time[np.argmax(above_90)]
    
    if t_90 <= t_10:
        return None
    
    return float(t_90 - t_10)


def compute_dominant_frequency(
    data: np.ndarray,
    dt: float,
    exclude_dc: bool = True,
) -> Optional[float]:
    """
    Compute dominant frequency from FFT.
    
    Args:
        data: Signal data
        dt: Time step in seconds
        exclude_dc: Whether to exclude DC component
    
    Returns:
        Dominant frequency in Hz, or None if computation fails
    """
    valid_mask = ~np.isnan(data)
    if np.sum(valid_mask) < 16:
        return None
    
    # Use valid data, zero-pad NaN regions
    signal_data = np.nan_to_num(data, nan=0.0)
    
    # Compute FFT
    n = len(signal_data)
    fft_result = np.fft.fft(signal_data)
    freqs = np.fft.fftfreq(n, dt)
    
    # Consider only positive frequencies
    pos_mask = freqs > 0 if exclude_dc else freqs >= 0
    pos_freqs = freqs[pos_mask]
    pos_power = np.abs(fft_result[pos_mask]) ** 2
    
    if len(pos_power) == 0:
        return None
    
    # Find dominant frequency
    dominant_idx = np.argmax(pos_power)
    
    return float(pos_freqs[dominant_idx])


def compute_snr(
    data: np.ndarray,
    time: np.ndarray,
    event_start: float = 0.0002,
    baseline_end: float = 0.0001,
) -> float:
    """
    Estimate signal-to-noise ratio.
    
    SNR = std(signal in event window) / std(baseline)
    
    Args:
        data: Signal data
        time: Time array
        event_start: Start time of event window
        baseline_end: End time of baseline window
    
    Returns:
        SNR estimate
    """
    valid_mask = ~np.isnan(data)
    
    # Baseline region
    baseline_mask = (time < baseline_end) & valid_mask
    # Event region
    event_mask = (time >= event_start) & valid_mask
    
    if np.sum(baseline_mask) < 3 or np.sum(event_mask) < 3:
        return np.nan
    
    baseline_std = np.std(data[baseline_mask])
    event_std = np.std(data[event_mask])
    
    if baseline_std < 1e-15:
        return np.inf if event_std > 1e-15 else 1.0
    
    return float(event_std / baseline_std)


def compute_channel_metrics(
    data: np.ndarray,
    time: np.ndarray,
    channel_name: str,
    window_start: Optional[float] = None,
    window_end: Optional[float] = None,
) -> ChannelMetrics:
    """
    Compute all metrics for a single channel.
    
    Args:
        data: Signal data array
        time: Time array
        channel_name: Name of the channel (affects which metrics are computed)
        window_start: Start of analysis window
        window_end: End of analysis window
    
    Returns:
        ChannelMetrics object with computed values
    """
    dt = np.median(np.diff(time)) if len(time) > 1 else 1e-6
    
    # Basic metrics (all channels)
    peak = compute_peak(data)
    min_val = compute_min(data)
    peak_to_peak = compute_peak_to_peak(data)
    time_of_peak = compute_time_of_peak(data, time)
    auc = compute_auc(data, time, window_start, window_end)
    snr = compute_snr(data, time)
    
    # Rise time (mainly for photodiode-like channels)
    rise_time = None
    if channel_name in ["photodiode", "interferometer"]:
        rise_time = compute_rise_time_10_90(data, time)
    
    # Dominant frequency (mainly for b_dot)
    dominant_freq = None
    if channel_name in ["b_dot"]:
        dominant_freq = compute_dominant_frequency(data, dt)
    
    return ChannelMetrics(
        peak=peak,
        min_val=min_val,
        peak_to_peak=peak_to_peak,
        time_of_peak=time_of_peak,
        auc=auc,
        rise_time_10_90=rise_time,
        dominant_freq_hz=dominant_freq,
        snr_estimate=snr,
    )


def compute_shot_metrics(
    shot,  # Shot object from io module
    channels: Optional[list[str]] = None,
    window_start: Optional[float] = None,
    window_end: Optional[float] = None,
) -> dict[str, ChannelMetrics]:
    """
    Compute metrics for all channels in a shot.
    
    Args:
        shot: Shot object
        channels: List of channels to compute (None = all)
        window_start: Start of analysis window
        window_end: End of analysis window
    
    Returns:
        Dictionary mapping channel names to ChannelMetrics
    """
    if channels is None:
        channels = shot.channels
    
    time = shot.time
    results = {}
    
    for channel in channels:
        if channel in shot.channels:
            data = shot.get_channel(channel)
            results[channel] = compute_channel_metrics(
                data, time, channel, window_start, window_end
            )
    
    return results
