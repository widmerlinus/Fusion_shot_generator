# Shot Explorer ⚡

A Python mini-dashboard for viewing and analyzing pulsed plasma experiment data. Built with Streamlit for interactive exploration, this tool demonstrates plasma data analysis workflows including multi-channel time series visualization, metrics computation, quality assessment, and report generation. Vibe coded with Claude Opus 4.5.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Multi-shot overlay plots** - Compare time series across multiple shots on aligned axes
- **Per-shot metrics computation** - Peak, min, AUC, rise time, dominant frequency, SNR
- **Automatic quality flagging** - Detect missing data, saturation, noisy baselines, timing outliers
- **Trend analysis** - Scatter plots showing metric vs shot index or control variables
- **Report export** - Generate Markdown reports with embedded plots and metrics tables
- **Synthetic data generator** - Create realistic test datasets with controlled imperfections

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd shot-explorer

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
python generate_synthetic_data.py --n_shots 30 --seed 42
```

This creates 30 synthetic shot files in `data/shots/` with realistic:
- 4 diagnostic channels (B-dot, interferometer, photodiode, X-ray proxy)
- Shot-to-shot variation in amplitude and timing
- Control variables (gas pressure, injector voltage, coil current)
- Data quality imperfections (missing data, saturation, drift, noise)

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard opens in your browser at `http://localhost:8501`.

## Dashboard Guide

### Sidebar Controls

- **Data Selection**: Point to your shot data folder
- **Shot Selection**: Filter by range, exclude flagged shots, multi-select specific shots
- **Channel Selection**: Choose which diagnostic channels to display
- **Preprocessing**: Toggle baseline subtraction, smoothing, and normalization
- **Analysis Window**: Set time range for metric computation

### Main Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | Dataset summary, quality flag counts, selected shot list |
| **Overlay Plots** | Multi-shot overlays per channel with optional mean±std envelope |
| **Metrics Table** | Sortable table of computed metrics per shot |
| **Trends** | Scatter plot: metric vs shot index or control variable |
| **Report** | Export Markdown report with plots and tables |

## Data Format

### Shot CSV Files

Each shot is stored as a CSV file with columns:
```csv
t,b_dot,interferometer,photodiode,xray_proxy
0.00000000e+00,-1.23456789e-02,1.23456789e-01,...
1.00000000e-06,-1.34567890e-02,1.24567890e-01,...
```

- `t`: Time in seconds (float)
- Additional columns: One per diagnostic channel

### Metadata Sidecar (Optional)

```json
{
  "shot_id": "0001",
  "control_vars": {
    "gas_pressure_mTorr": 5.2,
    "injector_voltage_V": 1200,
    "coil_current_kA": 12.5
  }
}
```

Saved as `shot_0001.meta.json` alongside `shot_0001.csv`.

## How This Maps to Pulsed Plasma Workflows

This tool mirrors the daily workflow of experimental plasma physics groups:

| Concept | In Real Labs | In Shot Explorer |
|---------|-------------|------------------|
| **Shot** | One experimental pulse (~1-10 ms) | One CSV file with time series |
| **Channels** | B-dot probes, interferometry, spectroscopy, photodiodes | Configurable columns in CSV |
| **Overlay plots** | "Does this look like yesterday's good shots?" | Multi-shot comparison with mean±std |
| **Metrics** | "What was the peak density? When did breakdown occur?" | Automated extraction: peak, timing, AUC |
| **Quality flags** | "Probe 3 saturated again" | Automatic detection of common issues |
| **Trends** | "How does emission scale with fill pressure?" | Metric vs control variable scatter plots |
| **Reports** | Weekly summaries for PI | Auto-generated Markdown + PNG export |

## Computed Metrics

| Metric | Description | Channels |
|--------|-------------|----------|
| `peak` | Maximum value | All |
| `min` | Minimum value | All |
| `peak_to_peak` | Max - Min | All |
| `time_of_peak` | Time at maximum | All |
| `auc` | Area under curve (trapezoid integration) | All |
| `rise_time_10_90` | 10% to 90% rise time | photodiode, interferometer |
| `dominant_freq_hz` | FFT peak frequency (excl. DC) | b_dot |
| `snr_estimate` | Signal std / baseline std | All |

## Quality Flags

| Flag | Detection Method |
|------|-----------------|
| `missing_data` | >10% NaN values |
| `saturation` | ≥10 consecutive identical values, or clipping at extremes |
| `noisy_baseline` | Baseline std > 10% of signal range |
| `low_signal` | Peak below channel-specific threshold |
| `timing_outlier` | Event time >2σ from median across shots |

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
shot-explorer/
├── app.py                      # Streamlit dashboard
├── generate_synthetic_data.py  # Data generation CLI
├── shotlib/
│   ├── io.py                   # Data loading
│   ├── preprocess.py           # Signal preprocessing
│   ├── metrics.py              # Metrics computation
│   ├── quality.py              # Quality assessment
│   ├── plotting.py             # Visualization
│   ├── report.py               # Report generation
│   └── synthetic.py            # Synthetic data generation
├── data/shots/                 # Shot data files
├── reports/                    # Generated reports
├── tests/
│   ├── test_metrics.py
│   └── test_quality.py
├── requirements.txt
├── README.md
└── LICENSE
```

## If Integrated Into a Lab

This tool is designed as a demonstration, but could adapt to real lab environments:

1. **HDF5/MDSplus support** - Replace CSV loader with h5py or MDSplus client for standard fusion data formats

2. **Calibration tables** - Add per-channel calibration factors loaded from config files, with automatic unit conversion

3. **Shot database integration** - Query shots by date range, operator, or experimental campaign from a SQL/MongoDB backend

4. **Real-time mode** - Watch a folder for new shots and auto-update the dashboard during experimental runs

5. **Machine learning flagging** - Train classifiers on manually-labeled "good/bad" shots to automatically score new data

6. **Multi-device support** - Handle data from different diagnostics with varying sample rates via interpolation to a common time base

7. **Reproducible analysis** - Export preprocessing + metric parameters as YAML config files for batch reprocessing

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Built as a demonstration project for plasma data analysis workflows.*
