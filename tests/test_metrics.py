"""
Tests for metrics computation module.
"""

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from shotlib.metrics import (
    compute_peak,
    compute_min,
    compute_peak_to_peak,
    compute_time_of_peak,
    compute_auc,
    compute_rise_time_10_90,
    compute_dominant_frequency,
    compute_snr,
    ChannelMetrics,
)


class TestBasicMetrics:
    """Test basic metric computations."""
    
    def test_compute_peak(self):
        """Test peak value computation."""
        data = np.array([1, 2, 5, 3, 1])
        assert compute_peak(data) == 5
    
    def test_compute_peak_with_nan(self):
        """Test peak with NaN values."""
        data = np.array([1, np.nan, 5, np.nan, 1])
        assert compute_peak(data) == 5
    
    def test_compute_peak_all_nan(self):
        """Test peak when all values are NaN."""
        data = np.array([np.nan, np.nan, np.nan])
        assert np.isnan(compute_peak(data))
    
    def test_compute_min(self):
        """Test minimum value computation."""
        data = np.array([3, 1, 5, 2, 4])
        assert compute_min(data) == 1
    
    def test_compute_min_negative(self):
        """Test minimum with negative values."""
        data = np.array([3, -2, 5, 1])
        assert compute_min(data) == -2
    
    def test_compute_peak_to_peak(self):
        """Test peak-to-peak computation."""
        data = np.array([1, 5, 2, -1, 3])
        assert compute_peak_to_peak(data) == 6  # 5 - (-1)
    
    def test_compute_time_of_peak(self):
        """Test time of peak detection."""
        data = np.array([1, 2, 5, 3, 1])
        time = np.array([0, 1, 2, 3, 4])
        assert compute_time_of_peak(data, time) == 2


class TestAUC:
    """Test area under curve computation."""
    
    def test_auc_simple(self):
        """Test AUC for simple rectangular signal."""
        data = np.array([1, 1, 1, 1, 1])
        time = np.array([0, 1, 2, 3, 4])
        auc = compute_auc(data, time)
        assert np.isclose(auc, 4.0)  # height=1, width=4
    
    def test_auc_triangle(self):
        """Test AUC for triangular signal."""
        data = np.array([0, 1, 2, 1, 0])
        time = np.array([0, 1, 2, 3, 4])
        auc = compute_auc(data, time)
        assert np.isclose(auc, 4.0)  # Triangle area
    
    def test_auc_with_window(self):
        """Test AUC with specified window."""
        data = np.array([0, 1, 2, 3, 4])
        time = np.array([0, 1, 2, 3, 4])
        auc = compute_auc(data, time, window_start=1, window_end=3)
        assert np.isclose(auc, 4.0)  # Trapezoid from t=1 to t=3


class TestRiseTime:
    """Test rise time computation."""
    
    def test_rise_time_clean_signal(self):
        """Test rise time for clean step-like signal."""
        # Create signal that rises from 0 to 10
        t = np.linspace(0, 10, 1000)
        # Sigmoid-like rise
        data = 10 / (1 + np.exp(-(t - 5)))
        
        rise_time = compute_rise_time_10_90(data, t)
        assert rise_time is not None
        assert 0 < rise_time < 10
    
    def test_rise_time_flat_signal(self):
        """Test rise time for flat signal returns None."""
        data = np.ones(100)
        time = np.linspace(0, 1, 100)
        
        rise_time = compute_rise_time_10_90(data, time)
        assert rise_time is None


class TestDominantFrequency:
    """Test dominant frequency computation."""
    
    def test_dominant_freq_sine(self):
        """Test dominant frequency detection for sine wave."""
        dt = 1e-5  # 100 kHz sampling
        t = np.arange(0, 0.01, dt)  # 10 ms
        freq = 5000  # 5 kHz signal
        data = np.sin(2 * np.pi * freq * t)
        
        detected_freq = compute_dominant_frequency(data, dt)
        assert detected_freq is not None
        # Allow some FFT bin resolution error
        assert abs(detected_freq - freq) < 200
    
    def test_dominant_freq_with_noise(self):
        """Test frequency detection with noise."""
        dt = 1e-5
        t = np.arange(0, 0.01, dt)
        freq = 10000
        data = np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.1, len(t))
        
        detected_freq = compute_dominant_frequency(data, dt)
        assert detected_freq is not None


class TestSNR:
    """Test signal-to-noise ratio computation."""
    
    def test_snr_high_signal(self):
        """Test SNR for high signal case."""
        time = np.linspace(0, 1e-3, 1000)
        # Low baseline, high event
        data = np.zeros(1000)
        data[:100] = np.random.normal(0, 0.01, 100)  # Baseline with low noise
        data[200:] = np.random.normal(5, 1, 800)  # Signal region
        
        snr = compute_snr(data, time, event_start=2e-4, baseline_end=1e-4)
        assert snr > 1
    
    def test_snr_no_signal(self):
        """Test SNR when signal is similar to baseline."""
        time = np.linspace(0, 1e-3, 1000)
        data = np.random.normal(0, 1, 1000)  # All noise
        
        snr = compute_snr(data, time)
        assert np.isclose(snr, 1, atol=0.5)


class TestChannelMetrics:
    """Test ChannelMetrics dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ChannelMetrics(
            peak=10.0,
            min_val=-2.0,
            peak_to_peak=12.0,
            time_of_peak=0.5,
            auc=5.0,
            rise_time_10_90=0.1,
            dominant_freq_hz=1000.0,
            snr_estimate=5.0,
        )
        
        d = metrics.to_dict()
        assert d["peak"] == 10.0
        assert d["min"] == -2.0
        assert d["peak_to_peak"] == 12.0
        assert d["rise_time_10_90"] == 0.1
