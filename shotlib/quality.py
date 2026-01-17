"""
Data quality assessment utilities.

Detects and flags common data quality issues: missing data, saturation,
noisy baselines, low signal, and timing outliers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class QualityFlag(str, Enum):
    """Quality issue flags."""
    MISSING_DATA = "missing_data"
    SATURATION = "saturation"
    NOISY_BASELINE = "noisy_baseline"
    LOW_SIGNAL = "low_signal"
    TIMING_OUTLIER = "timing_outlier"


@dataclass
class QualityResult:
    """Quality assessment result for a single channel."""
    flags: list[QualityFlag]
    details: dict
    
    @property
    def is_clean(self) -> bool:
        """Return True if no quality issues detected."""
        return len(self.flags) == 0
    
    @property
    def flag_names(self) -> list[str]:
        """Return list of flag names as strings."""
        return [f.value for f in self.flags]


def check_missing_data(
    data: np.ndarray,
    threshold_fraction: float = 0.1,
) -> tuple[bool, dict]:
    """
    Check for excessive missing data (NaN values).
    
    Args:
        data: Signal data array
        threshold_fraction: Fraction of NaN values to trigger flag
    
    Returns:
        Tuple of (flag_triggered, details_dict)
    """
    n_total = len(data)
    n_nan = np.sum(np.isnan(data))
    nan_fraction = n_nan / n_total if n_total > 0 else 0
    
    return (
        nan_fraction > threshold_fraction,
        {
            "nan_count": int(n_nan),
            "nan_fraction": float(nan_fraction),
            "threshold": threshold_fraction,
        }
    )


def check_saturation(
    data: np.ndarray,
    stuck_samples_threshold: int = 10,
    clip_detection: bool = True,
) -> tuple[bool, dict]:
    """
    Check for sensor saturation (clipped or stuck values).
    
    Args:
        data: Signal data array
        stuck_samples_threshold: Number of consecutive identical values to flag
        clip_detection: Whether to check for clipping at min/max
    
    Returns:
        Tuple of (flag_triggered, details_dict)
    """
    valid_data = data[~np.isnan(data)]
    if len(valid_data) < stuck_samples_threshold:
        return False, {"saturated": False}
    
    details = {"saturated": False}
    
    # Check for stuck values (consecutive identical values)
    if len(valid_data) > stuck_samples_threshold:
        diff = np.diff(valid_data)
        # Find runs of zero diff
        zero_diff = np.abs(diff) < 1e-15
        
        # Count longest run
        max_run = 0
        current_run = 0
        for is_zero in zero_diff:
            if is_zero:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        if max_run >= stuck_samples_threshold:
            details["stuck_run_length"] = max_run
            details["saturated"] = True
    
    # Check for clipping at extremes
    if clip_detection and len(valid_data) > 10:
        data_max = np.max(valid_data)
        data_min = np.min(valid_data)
        
        # Count values at or very near the extremes
        at_max = np.sum(np.abs(valid_data - data_max) < 1e-10)
        at_min = np.sum(np.abs(valid_data - data_min) < 1e-10)
        
        # If many values are clipped at the extreme
        clip_threshold = len(valid_data) * 0.02  # 2% threshold
        if at_max > clip_threshold or at_min > clip_threshold:
            details["clipped_at_max"] = int(at_max)
            details["clipped_at_min"] = int(at_min)
            details["saturated"] = True
    
    return details["saturated"], details


def check_noisy_baseline(
    data: np.ndarray,
    time: np.ndarray,
    baseline_end: float = 0.0001,
    noise_threshold_factor: float = 0.1,
) -> tuple[bool, dict]:
    """
    Check for excessively noisy baseline.
    
    Compares baseline std to overall signal amplitude.
    
    Args:
        data: Signal data array
        time: Time array
        baseline_end: End time of baseline region
        noise_threshold_factor: Ratio of baseline_std to amplitude to flag
    
    Returns:
        Tuple of (flag_triggered, details_dict)
    """
    valid_mask = ~np.isnan(data)
    baseline_mask = (time < baseline_end) & valid_mask
    
    if np.sum(baseline_mask) < 5 or np.sum(valid_mask) < 10:
        return False, {"baseline_std": np.nan, "noisy": False}
    
    baseline_data = data[baseline_mask]
    baseline_std = np.std(baseline_data)
    
    # Compare to overall signal amplitude
    signal_range = np.max(data[valid_mask]) - np.min(data[valid_mask])
    
    if signal_range < 1e-15:
        ratio = 0.0
    else:
        ratio = baseline_std / signal_range
    
    is_noisy = ratio > noise_threshold_factor
    
    return is_noisy, {
        "baseline_std": float(baseline_std),
        "signal_range": float(signal_range),
        "noise_ratio": float(ratio),
        "threshold": noise_threshold_factor,
        "noisy": is_noisy,
    }


def check_low_signal(
    data: np.ndarray,
    peak_threshold: float = 0.1,
) -> tuple[bool, dict]:
    """
    Check for unusually low signal amplitude.
    
    Args:
        data: Signal data array
        peak_threshold: Minimum expected peak value
    
    Returns:
        Tuple of (flag_triggered, details_dict)
    """
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return True, {"peak": np.nan, "low_signal": True}
    
    peak = np.max(np.abs(valid_data))
    is_low = peak < peak_threshold
    
    return is_low, {
        "peak": float(peak),
        "threshold": peak_threshold,
        "low_signal": is_low,
    }


def check_timing_outlier(
    event_time: float,
    reference_times: list[float],
    threshold_std: float = 2.0,
) -> tuple[bool, dict]:
    """
    Check if event time is an outlier compared to reference shots.
    
    Args:
        event_time: Detected event time for this shot
        reference_times: Event times from other shots
        threshold_std: Number of std deviations to consider outlier
    
    Returns:
        Tuple of (flag_triggered, details_dict)
    """
    if len(reference_times) < 3 or np.isnan(event_time):
        return False, {"timing_outlier": False}
    
    ref_array = np.array([t for t in reference_times if not np.isnan(t)])
    if len(ref_array) < 3:
        return False, {"timing_outlier": False}
    
    median = np.median(ref_array)
    std = np.std(ref_array)
    
    if std < 1e-10:
        is_outlier = False
        z_score = 0.0
    else:
        z_score = abs(event_time - median) / std
        is_outlier = z_score > threshold_std
    
    return is_outlier, {
        "event_time": float(event_time),
        "median_time": float(median),
        "std_time": float(std),
        "z_score": float(z_score),
        "threshold_std": threshold_std,
        "timing_outlier": is_outlier,
    }


def assess_channel_quality(
    data: np.ndarray,
    time: np.ndarray,
    channel_name: str,
    event_times: Optional[list[float]] = None,
    detected_event_time: Optional[float] = None,
) -> QualityResult:
    """
    Perform complete quality assessment for a channel.
    
    Args:
        data: Signal data array
        time: Time array
        channel_name: Name of the channel
        event_times: Event times from other shots (for timing outlier check)
        detected_event_time: Detected event time for this shot
    
    Returns:
        QualityResult with all detected issues
    """
    flags = []
    details = {}
    
    # Check missing data
    missing_flag, missing_details = check_missing_data(data)
    details["missing"] = missing_details
    if missing_flag:
        flags.append(QualityFlag.MISSING_DATA)
    
    # Check saturation
    sat_flag, sat_details = check_saturation(data)
    details["saturation"] = sat_details
    if sat_flag:
        flags.append(QualityFlag.SATURATION)
    
    # Check noisy baseline
    noisy_flag, noisy_details = check_noisy_baseline(data, time)
    details["noise"] = noisy_details
    if noisy_flag:
        flags.append(QualityFlag.NOISY_BASELINE)
    
    # Check low signal (threshold depends on channel)
    # These thresholds are rough estimates for synthetic data
    peak_thresholds = {
        "b_dot": 0.05,
        "interferometer": 0.2,
        "photodiode": 0.1,
        "xray_proxy": 0.05,
    }
    threshold = peak_thresholds.get(channel_name, 0.1)
    low_flag, low_details = check_low_signal(data, threshold)
    details["low_signal"] = low_details
    if low_flag:
        flags.append(QualityFlag.LOW_SIGNAL)
    
    # Check timing outlier
    if event_times is not None and detected_event_time is not None:
        timing_flag, timing_details = check_timing_outlier(
            detected_event_time, event_times
        )
        details["timing"] = timing_details
        if timing_flag:
            flags.append(QualityFlag.TIMING_OUTLIER)
    
    return QualityResult(flags=flags, details=details)


def assess_shot_quality(
    shot,  # Shot object
    channels: Optional[list[str]] = None,
    all_event_times: Optional[dict[str, list[float]]] = None,
) -> dict[str, QualityResult]:
    """
    Assess quality for all channels in a shot.
    
    Args:
        shot: Shot object
        channels: List of channels to assess (None = all)
        all_event_times: Dict mapping channel -> list of event times from all shots
    
    Returns:
        Dictionary mapping channel names to QualityResult
    """
    from .preprocess import detect_event_time
    
    if channels is None:
        channels = shot.channels
    
    time = shot.time
    results = {}
    
    for channel in channels:
        if channel not in shot.channels:
            continue
        
        data = shot.get_channel(channel)
        
        # Detect event time for this channel
        event_time = detect_event_time(data, time)
        
        # Get reference event times
        ref_times = None
        if all_event_times is not None and channel in all_event_times:
            ref_times = all_event_times[channel]
        
        results[channel] = assess_channel_quality(
            data, time, channel,
            event_times=ref_times,
            detected_event_time=event_time,
        )
    
    return results


def summarize_quality_flags(
    all_quality: dict[str, dict[str, QualityResult]],
) -> dict[str, int]:
    """
    Summarize quality flags across all shots.
    
    Args:
        all_quality: Dict mapping shot_id -> channel -> QualityResult
    
    Returns:
        Dict mapping flag name -> count
    """
    counts = {flag.value: 0 for flag in QualityFlag}
    
    for shot_id, channel_results in all_quality.items():
        for channel, result in channel_results.items():
            for flag in result.flags:
                counts[flag.value] += 1
    
    return counts
