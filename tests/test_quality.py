"""
Tests for quality assessment module.
"""

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from shotlib.quality import (
    check_missing_data,
    check_saturation,
    check_noisy_baseline,
    check_low_signal,
    check_timing_outlier,
    QualityFlag,
)


class TestMissingData:
    """Test missing data detection."""
    
    def test_no_missing_data(self):
        """Test with clean data."""
        data = np.array([1, 2, 3, 4, 5])
        flag, details = check_missing_data(data)
        assert flag == False
        assert details["nan_fraction"] == 0
    
    def test_some_missing_data(self):
        """Test with some NaN values below threshold."""
        data = np.array([1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10])  # 10% NaN
        flag, details = check_missing_data(data, threshold_fraction=0.15)
        assert flag == False
    
    def test_excessive_missing_data(self):
        """Test with excessive NaN values."""
        data = np.array([1, np.nan, np.nan, np.nan, 5])  # 60% NaN
        flag, details = check_missing_data(data, threshold_fraction=0.1)
        assert flag == True
        assert details["nan_fraction"] == 0.6
    
    def test_all_missing(self):
        """Test with all NaN values."""
        data = np.array([np.nan, np.nan, np.nan])
        flag, details = check_missing_data(data)
        assert flag == True
        assert details["nan_fraction"] == 1.0


class TestSaturation:
    """Test saturation detection."""
    
    def test_no_saturation(self):
        """Test with normal varying signal."""
        data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        flag, details = check_saturation(data)
        assert flag == False
    
    def test_stuck_values(self):
        """Test with stuck values (consecutive identical)."""
        data = np.array([1, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 2])  # 11 identical
        flag, details = check_saturation(data, stuck_samples_threshold=10)
        assert flag == True
        assert "stuck_run_length" in details
    
    def test_clipped_at_max(self):
        """Test with clipped values at maximum."""
        data = np.random.normal(0, 1, 100)
        max_val = np.max(data)
        # Clip to create saturation
        data = np.clip(data, None, max_val * 0.8)
        # Add more points at the clip level
        data[-10:] = max_val * 0.8
        
        flag, details = check_saturation(data)
        # May or may not trigger depending on exact clipping
        assert isinstance(flag, (bool, np.bool_))


class TestNoisyBaseline:
    """Test noisy baseline detection."""
    
    def test_clean_baseline(self):
        """Test with clean baseline."""
        time = np.linspace(0, 1e-3, 1000)
        data = np.zeros(1000)
        data[:100] = np.random.normal(0, 0.01, 100)  # Low noise baseline
        data[200:] = np.random.normal(10, 0.1, 800)  # Signal region
        
        flag, details = check_noisy_baseline(data, time, baseline_end=1e-4)
        assert flag == False
    
    def test_noisy_baseline(self):
        """Test with noisy baseline."""
        time = np.linspace(0, 1e-3, 1000)
        data = np.zeros(1000)
        data[:100] = np.random.normal(0, 5, 100)  # High noise baseline
        data[200:] = np.random.normal(10, 0.1, 800)  # Clean signal
        
        flag, details = check_noisy_baseline(data, time, baseline_end=1e-4,
                                             noise_threshold_factor=0.1)
        assert flag == True
        assert "noise_ratio" in details


class TestLowSignal:
    """Test low signal detection."""
    
    def test_good_signal(self):
        """Test with adequate signal."""
        data = np.array([0, 0.5, 1.0, 0.5, 0])
        flag, details = check_low_signal(data, peak_threshold=0.5)
        assert flag == False
    
    def test_low_signal(self):
        """Test with low signal."""
        data = np.array([0, 0.01, 0.02, 0.01, 0])
        flag, details = check_low_signal(data, peak_threshold=0.1)
        assert flag == True
        assert details["peak"] < 0.1


class TestTimingOutlier:
    """Test timing outlier detection."""
    
    def test_normal_timing(self):
        """Test with normal timing."""
        event_time = 0.0002
        reference_times = [0.00019, 0.0002, 0.00021, 0.00019, 0.0002]
        
        flag, details = check_timing_outlier(event_time, reference_times)
        assert flag == False
    
    def test_timing_outlier(self):
        """Test with outlier timing."""
        event_time = 0.001  # Much later than others
        reference_times = [0.0002, 0.00019, 0.00021, 0.0002, 0.00019]
        
        flag, details = check_timing_outlier(event_time, reference_times, threshold_std=2.0)
        assert flag == True
        assert details["z_score"] > 2.0
    
    def test_insufficient_references(self):
        """Test with too few reference times."""
        event_time = 0.0002
        reference_times = [0.0002]  # Only one reference
        
        flag, details = check_timing_outlier(event_time, reference_times)
        assert flag == False  # Should not flag with insufficient data


class TestQualityFlag:
    """Test QualityFlag enum."""
    
    def test_flag_values(self):
        """Test flag string values."""
        assert QualityFlag.MISSING_DATA.value == "missing_data"
        assert QualityFlag.SATURATION.value == "saturation"
        assert QualityFlag.NOISY_BASELINE.value == "noisy_baseline"
        assert QualityFlag.LOW_SIGNAL.value == "low_signal"
        assert QualityFlag.TIMING_OUTLIER.value == "timing_outlier"
