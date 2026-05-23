# Shot Explorer - Project Context

## Purpose
A Streamlit dashboard for analyzing pulsed plasma experiment data. Built as a 
portfolio project for a plasma data analysis co-op application at a fusion company.

## Architecture
- `app.py` - Main Streamlit dashboard with 5 tabs (Overview, Overlay Plots, 
  Metrics Table, Trends, Report)
- `generate_synthetic_data.py` - CLI for generating sample data
- `shotlib/` - Core library:
  - `io.py` - Shot loading from CSV with metadata sidecars
  - `synthetic.py` - Synthetic data generator with physics-inspired models
  - `preprocess.py` - Baseline subtraction, smoothing, normalization
  - `metrics.py` - Peak, AUC, rise time, dominant frequency, SNR
  - `quality.py` - Auto-flagging: missing data, saturation, noisy baseline, 
    timing outliers
  - `plotting.py` - Overlay plots, trend plots, quality summary
  - `report.py` - Markdown report generation with PNG plots
- `tests/` - pytest tests for metrics and quality modules

## Data Model
- One CSV per shot: `shot_NNNN.csv` with columns `t, b_dot, interferometer, 
  photodiode, xray_proxy`
- Optional metadata sidecar: `shot_NNNN.meta.json` with control_vars dict
- Channels represent: B-dot probe (dB/dt), interferometer (line-integrated 
  density), photodiode (visible emission), X-ray proxy (soft X-ray emission)

## Key Design Decisions
- Uses `matplotlib.use('Agg')` for Streamlit Cloud compatibility
- Plots displayed via `st.image()` with BytesIO buffer (not `st.pyplot`)
- Data auto-generates on first run if `data/shots/` is empty
- `seed=None` in app.py so data differs between regenerations
- Uses `np.trapezoid` (NumPy 2.0+) with fallback to `np.trapz`

## Deployment
- Hosted on Streamlit Community Cloud
- Pushed to GitHub repo
- Data is generated at runtime (not committed to repo) to keep it small

## Known Quirks
- Quality flag types use numpy bools, so tests use `==` not `is`
- Report tab parses markdown manually to display images via `st.image()` since 
  Streamlit can't resolve relative image paths in markdown