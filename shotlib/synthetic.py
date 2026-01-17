"""
Synthetic plasma shot data generator.

Generates realistic-looking multi-channel shot time series with noise,
drift, missing data, saturation, and shot-to-shot variation.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional


# Channel configuration with realistic characteristics
CHANNEL_CONFIG = {
    "b_dot": {
        "description": "Magnetic field derivative (B-dot probe)",
        "units": "T/s",
        "base_amplitude": 1.0,
    },
    "interferometer": {
        "description": "Line-integrated electron density proxy",
        "units": "1e19 m^-2",
        "base_amplitude": 5.0,
    },
    "photodiode": {
        "description": "Visible light emission",
        "units": "V",
        "base_amplitude": 2.5,
    },
    "xray_proxy": {
        "description": "Soft X-ray emission proxy",
        "units": "a.u.",
        "base_amplitude": 0.8,
    },
}


def generate_b_dot(
    t: np.ndarray,
    event_time: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate B-dot probe signal: oscillatory with transient spikes.
    
    The B-dot probe measures dB/dt, showing oscillatory behavior during
    plasma formation and transient spikes during instabilities.
    """
    signal = np.zeros_like(t)
    
    # Main oscillation during plasma phase
    plasma_mask = t > event_time
    freq = 50e3 + rng.normal(0, 5e3)  # ~50 kHz oscillation
    decay_time = 0.0003 + rng.uniform(-0.0001, 0.0001)
    
    t_rel = t[plasma_mask] - event_time
    oscillation = amplitude * np.sin(2 * np.pi * freq * t_rel)
    envelope = np.exp(-t_rel / decay_time)
    signal[plasma_mask] = oscillation * envelope
    
    # Add transient spikes (instabilities)
    n_spikes = rng.integers(2, 6)
    for _ in range(n_spikes):
        spike_time = event_time + rng.uniform(0.0001, 0.0004)
        spike_width = 5e-6
        spike_amp = amplitude * rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
        spike = spike_amp * np.exp(-((t - spike_time) ** 2) / (2 * spike_width ** 2))
        signal += spike
    
    # Baseline noise
    signal += rng.normal(0, amplitude * 0.05, len(t))
    
    return signal


def generate_interferometer(
    t: np.ndarray,
    event_time: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate interferometer signal: smooth density-like hump.
    
    Represents line-integrated electron density with smooth rise and fall.
    """
    signal = np.zeros_like(t)
    
    # Smooth density profile (asymmetric Gaussian-like)
    rise_time = 0.0001 + rng.uniform(-0.00002, 0.00002)
    fall_time = 0.0003 + rng.uniform(-0.0001, 0.0001)
    peak_time = event_time + rise_time * 3
    
    # Rising phase
    rise_mask = (t > event_time) & (t <= peak_time)
    t_rise = t[rise_mask] - event_time
    signal[rise_mask] = amplitude * (1 - np.exp(-t_rise / rise_time))
    
    # Falling phase
    fall_mask = t > peak_time
    t_fall = t[fall_mask] - peak_time
    signal[fall_mask] = amplitude * np.exp(-t_fall / fall_time)
    
    # Add slow drift and noise
    signal += rng.normal(0, amplitude * 0.02, len(t))
    
    return signal


def generate_photodiode(
    t: np.ndarray,
    event_time: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate photodiode signal: fast rising emission peak.
    
    Visible light emission with fast rise time and slower decay.
    """
    signal = np.zeros_like(t)
    
    # Fast rise, slower decay emission profile
    rise_time = 2e-5 + rng.uniform(-5e-6, 5e-6)
    decay_time = 0.0002 + rng.uniform(-0.00005, 0.00005)
    
    plasma_mask = t > event_time
    t_rel = t[plasma_mask] - event_time
    
    # Double exponential profile
    signal[plasma_mask] = amplitude * (
        np.exp(-t_rel / decay_time) - np.exp(-t_rel / rise_time)
    )
    # Normalize peak to amplitude
    if np.any(plasma_mask):
        signal[plasma_mask] *= amplitude / (signal[plasma_mask].max() + 1e-10)
    
    # Add noise
    signal += rng.normal(0, amplitude * 0.03, len(t))
    signal = np.maximum(signal, 0)  # Photodiode can't go negative
    
    return signal


def generate_xray_proxy(
    t: np.ndarray,
    event_time: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate X-ray proxy signal: spiky, high-noise bursts.
    
    Soft X-ray emission with bursty behavior and high noise floor.
    """
    signal = np.zeros_like(t)
    
    # Base emission envelope
    plasma_mask = t > event_time
    if not np.any(plasma_mask):
        return signal + rng.normal(0, amplitude * 0.1, len(t))
    
    t_rel = t[plasma_mask] - event_time
    decay_time = 0.00025 + rng.uniform(-0.0001, 0.0001)
    envelope = amplitude * 0.5 * np.exp(-t_rel / decay_time)
    
    # Add bursty spikes
    n_bursts = rng.integers(5, 15)
    for _ in range(n_bursts):
        burst_time = rng.uniform(0, t_rel.max()) if t_rel.max() > 0 else 0
        burst_width = rng.uniform(5e-6, 2e-5)
        burst_amp = amplitude * rng.uniform(0.3, 1.2)
        burst = burst_amp * np.exp(-((t_rel - burst_time) ** 2) / (2 * burst_width ** 2))
        envelope += burst
    
    signal[plasma_mask] = envelope
    
    # High noise floor
    signal += np.abs(rng.normal(0, amplitude * 0.15, len(t)))
    
    return signal


CHANNEL_GENERATORS = {
    "b_dot": generate_b_dot,
    "interferometer": generate_interferometer,
    "photodiode": generate_photodiode,
    "xray_proxy": generate_xray_proxy,
}


def inject_imperfections(
    data: dict[str, np.ndarray],
    shot_idx: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    Inject realistic imperfections into shot data.
    
    Returns modified data and list of injected imperfections.
    """
    imperfections = []
    
    # Missing channel (10% chance per channel)
    for channel in list(data.keys()):
        if channel == "t":
            continue
        if rng.random() < 0.08:
            # Inject NaNs for portion of the signal
            n_points = len(data[channel])
            start = rng.integers(0, n_points // 2)
            end = rng.integers(start + n_points // 4, n_points)
            data[channel][start:end] = np.nan
            imperfections.append(f"{channel}_missing")
    
    # Saturation clipping (15% chance for photodiode)
    if "photodiode" in data and rng.random() < 0.15:
        max_val = np.nanmax(data["photodiode"])
        clip_level = max_val * rng.uniform(0.6, 0.85)
        data["photodiode"] = np.clip(data["photodiode"], None, clip_level)
        imperfections.append("photodiode_saturated")
    
    # Baseline drift (20% chance for interferometer)
    if "interferometer" in data and rng.random() < 0.2:
        drift = np.linspace(0, rng.uniform(-0.5, 0.5), len(data["interferometer"]))
        data["interferometer"] += drift
        imperfections.append("interferometer_drift")
    
    # Noisy baseline (10% chance for b_dot)
    if "b_dot" in data and rng.random() < 0.1:
        extra_noise = rng.normal(0, np.nanstd(data["b_dot"]) * 2, len(data["b_dot"]))
        data["b_dot"] += extra_noise
        imperfections.append("b_dot_noisy")
    
    return data, imperfections


def generate_control_vars(rng: np.random.Generator) -> dict:
    """Generate realistic control variables for a shot."""
    return {
        "gas_pressure_mTorr": round(rng.uniform(1.0, 10.0), 2),
        "injector_voltage_V": round(rng.uniform(800, 1500), 0),
        "timing_offset_us": round(rng.uniform(-5.0, 5.0), 2),
        "coil_current_kA": round(rng.uniform(8.0, 15.0), 2),
    }


def generate_shot(
    shot_idx: int,
    rng: np.random.Generator,
    dt: float = 1e-6,
    duration: float = 0.001,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Generate a complete synthetic shot with all channels.
    
    Args:
        shot_idx: Shot index/number
        rng: Random number generator
        dt: Time step in seconds
        duration: Total duration in seconds
    
    Returns:
        Tuple of (data dict with time and channels, metadata dict)
    """
    t = np.arange(0, duration, dt)
    
    # Event time with shot-to-shot jitter
    base_event_time = 0.0002
    event_time = base_event_time + rng.uniform(-3e-5, 3e-5)
    
    # Control variables affect signal characteristics
    control_vars = generate_control_vars(rng)
    
    # Amplitude scaling based on control vars
    pressure_factor = control_vars["gas_pressure_mTorr"] / 5.0
    voltage_factor = control_vars["injector_voltage_V"] / 1000.0
    
    data = {"t": t}
    
    for channel, config in CHANNEL_CONFIG.items():
        base_amp = config["base_amplitude"]
        # Shot-to-shot amplitude variation + control var effects
        amplitude = base_amp * rng.uniform(0.7, 1.3) * pressure_factor * voltage_factor
        
        # Time jitter from timing_offset
        channel_event_time = event_time + control_vars["timing_offset_us"] * 1e-6
        
        data[channel] = CHANNEL_GENERATORS[channel](t, channel_event_time, amplitude, rng)
    
    # Inject imperfections
    data, imperfections = inject_imperfections(data, shot_idx, rng)
    
    # Build metadata
    metadata = {
        "shot_id": f"{shot_idx:04d}",
        "control_vars": control_vars,
        "imperfections_injected": imperfections,
        "dt_s": dt,
        "duration_s": duration,
        "n_points": len(t),
        "channels": list(CHANNEL_CONFIG.keys()),
        "channel_units": {ch: cfg["units"] for ch, cfg in CHANNEL_CONFIG.items()},
    }
    
    return data, metadata


def generate_dataset(
    n_shots: int,
    out_dir: Path,
    seed: Optional[int] = None,
) -> None:
    """
    Generate a complete synthetic dataset.
    
    Args:
        n_shots: Number of shots to generate
        out_dir: Output directory for shot files
        seed: Random seed for reproducibility
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    
    print(f"Generating {n_shots} synthetic shots in {out_dir}")
    
    for i in range(1, n_shots + 1):
        data, metadata = generate_shot(i, rng)
        
        # Save CSV
        csv_path = out_dir / f"shot_{i:04d}.csv"
        
        # Build CSV content
        columns = ["t"] + list(CHANNEL_CONFIG.keys())
        header = ",".join(columns)
        
        rows = []
        for j in range(len(data["t"])):
            row = [f"{data[col][j]:.8e}" if not np.isnan(data[col][j]) else "" 
                   for col in columns]
            rows.append(",".join(row))
        
        csv_content = header + "\n" + "\n".join(rows)
        csv_path.write_text(csv_content)
        
        # Save metadata JSON
        meta_path = out_dir / f"shot_{i:04d}.meta.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        
        if i % 10 == 0 or i == n_shots:
            print(f"  Generated {i}/{n_shots} shots")
    
    print(f"Dataset generation complete: {out_dir}")
