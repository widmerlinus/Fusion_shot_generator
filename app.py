#!/usr/bin/env python3
"""
Shot Explorer - Plasma Shot Analysis Dashboard

A Streamlit-based mini-dashboard for viewing and analyzing pulsed plasma
experiment data.
"""

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from io import BytesIO

from shotlib.io import load_dataset, get_dataset_info, Shot
from shotlib.preprocess import (
    preprocess_signal, NormalizationMethod, detect_event_time, align_by_event
)
from shotlib.metrics import compute_shot_metrics
from shotlib.quality import assess_shot_quality, QualityFlag
from shotlib.plotting import (
    plot_channel_overlay, plot_trend, plot_quality_summary,
    save_figure, get_shot_color
)
from shotlib.report import generate_report

st.set_page_config(
    page_title="Shot Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


def show_fig(fig):
    """Convert matplotlib figure to PNG and display with st.image()."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


@st.cache_data
def load_shots_cached(data_dir: str):
    """Load and cache shot data."""
    shots = load_dataset(Path(data_dir))
    return [
        {
            "shot_id": s.shot_id,
            "data": s.data.to_dict(),
            "metadata": s.metadata,
            "filepath": str(s.filepath) if s.filepath else None,
        }
        for s in shots
    ]


def reconstruct_shots(cached_data):
    """Reconstruct Shot objects from cached data."""
    return [
        Shot(
            shot_id=item["shot_id"],
            data=pd.DataFrame(item["data"]),
            metadata=item["metadata"],
            filepath=Path(item["filepath"]) if item["filepath"] else None,
        )
        for item in cached_data
    ]


@st.cache_data
def compute_all_metrics_cached(cached_shots, channels, window_start, window_end):
    """Compute metrics for all shots."""
    shots = reconstruct_shots(cached_shots)
    all_metrics = {}
    for shot in shots:
        metrics = compute_shot_metrics(shot, channels=list(channels),
                                       window_start=window_start, window_end=window_end)
        all_metrics[shot.shot_id] = {ch: m.to_dict() for ch, m in metrics.items()}
    return all_metrics


@st.cache_data  
def compute_all_quality_cached(cached_shots, channels):
    """Compute quality assessment for all shots."""
    shots = reconstruct_shots(cached_shots)
    
    all_event_times = {ch: [] for ch in channels}
    for shot in shots:
        for ch in channels:
            if ch in shot.channels:
                event_time = detect_event_time(shot.get_channel(ch), shot.time)
                if event_time is not None:
                    all_event_times[ch].append(event_time)
    
    all_quality = {}
    for shot in shots:
        quality = assess_shot_quality(shot, channels=list(channels),
                                      all_event_times=all_event_times)
        all_quality[shot.shot_id] = {
            ch: {"flags": r.flag_names, "details": r.details}
            for ch, r in quality.items()
        }
    
    flag_counts = {f.value: 0 for f in QualityFlag}
    for shot_id, channel_results in all_quality.items():
        for channel, result in channel_results.items():
            for flag in result["flags"]:
                flag_counts[flag] += 1
    
    return all_quality, flag_counts


def format_metric(val, precision=3):
    """Format metric value for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, float):
        if abs(val) > 1000 or (abs(val) < 0.01 and val != 0):
            return f"{val:.{precision}e}"
        return f"{val:.{precision}f}"
    return str(val)


def main():
    st.title("⚡ Shot Explorer")
    st.markdown("*Plasma Shot Analysis Dashboard*")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Selection")
        
        data_dir = st.text_input("Dataset folder", value="data/shots",
                                 help="Path to folder containing shot CSV files")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
        
        # Auto-generate sample data if directory is empty
        if not any(data_path.glob("shot_*.csv")):
            st.info("No shot data found. Generating sample dataset...")
            from shotlib.synthetic import generate_dataset
            generate_dataset(n_shots=30, out_dir=data_path, seed=42)
            st.rerun()
        
        try:
            cached_data = load_shots_cached(data_dir)
            shots = reconstruct_shots(cached_data)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return
        
        if not shots:
            st.warning("No shot files found")
            return
        
        dataset_info = get_dataset_info(shots)
        st.success(f"Loaded {len(shots)} shots")
        
        st.divider()
        st.header("🎯 Shot Selection")
        
        all_shot_ids = [s.shot_id for s in shots]
        
        # Range filter
        col1, col2 = st.columns(2)
        with col1:
            start_idx = st.number_input("From shot", min_value=1, 
                                        max_value=len(shots), value=1)
        with col2:
            end_idx = st.number_input("To shot", min_value=1,
                                      max_value=len(shots), value=min(20, len(shots)))
        
        # Quality filter
        exclude_saturated = st.checkbox("Exclude saturated shots", value=False)
        exclude_missing = st.checkbox("Exclude shots with missing data", value=False)
        
        # Compute quality for filtering
        all_channels = tuple(dataset_info["channels"])
        all_quality, flag_counts = compute_all_quality_cached(cached_data, all_channels)
        
        # Apply filters
        filtered_ids = all_shot_ids[start_idx-1:end_idx]
        
        if exclude_saturated or exclude_missing:
            new_filtered = []
            for sid in filtered_ids:
                if sid in all_quality:
                    flags = []
                    for ch_result in all_quality[sid].values():
                        flags.extend(ch_result["flags"])
                    if exclude_saturated and "saturation" in flags:
                        continue
                    if exclude_missing and "missing_data" in flags:
                        continue
                new_filtered.append(sid)
            filtered_ids = new_filtered
        
        # Multi-select
        selected_ids = st.multiselect(
            "Select shots",
            options=filtered_ids,
            default=filtered_ids[:min(10, len(filtered_ids))],
            help="Select shots to analyze"
        )
        
        if st.button("Select all filtered"):
            selected_ids = filtered_ids
            st.rerun()
        
        st.divider()
        st.header("📊 Channel Selection")
        
        selected_channels = st.multiselect(
            "Channels to plot",
            options=dataset_info["channels"],
            default=dataset_info["channels"][:4]
        )
        
        st.divider()
        st.header("⚙️ Preprocessing")
        
        baseline_subtract = st.checkbox("Baseline subtraction", value=False)
        apply_smoothing = st.checkbox("Smoothing", value=False)
        
        if apply_smoothing:
            smooth_window = st.slider("Smoothing window", 5, 51, 11, step=2)
        else:
            smooth_window = 11
        
        normalization = st.selectbox(
            "Normalization",
            options=["none", "zscore", "divide_by_peak"],
            format_func=lambda x: {"none": "None", "zscore": "Z-score", 
                                   "divide_by_peak": "Divide by peak"}[x]
        )
        norm_method = NormalizationMethod(normalization)
        
        st.divider()
        st.header("⏱️ Analysis Window")
        
        t_min = shots[0].time.min() * 1e6
        t_max = shots[0].time.max() * 1e6
        window_range = st.slider(
            "Time window (µs)",
            min_value=float(t_min),
            max_value=float(t_max),
            value=(float(t_min), float(t_max))
        )
        window_start = window_range[0] * 1e-6
        window_end = window_range[1] * 1e-6
    
    # Main content
    if not selected_ids or not selected_channels:
        st.info("Select shots and channels from the sidebar to begin analysis")
        return
    
    selected_shots = [s for s in shots if s.shot_id in selected_ids]
    
    # Preprocess data
    preprocessed = {ch: {} for ch in selected_channels}
    aligned_times = {}
    
    for shot in selected_shots:
        aligned_times[shot.shot_id] = shot.time.copy()
        for ch in selected_channels:
            if ch in shot.channels:
                data = preprocess_signal(
                    shot.time, shot.get_channel(ch),
                    baseline_subtract=baseline_subtract,
                    smooth=apply_smoothing,
                    smooth_window=smooth_window,
                    normalization=norm_method
                )
                preprocessed[ch][shot.shot_id] = data
    
    # Compute metrics
    all_metrics = compute_all_metrics_cached(
        cached_data, tuple(selected_channels), window_start, window_end
    )
    
    # Tabs
    tab_overview, tab_plots, tab_metrics, tab_trends, tab_report = st.tabs([
        "📋 Overview", "📈 Overlay Plots", "📊 Metrics Table", 
        "📉 Trends", "📄 Report"
    ])
    
    # --- Overview Tab ---
    with tab_overview:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Shots", len(shots))
        with col2:
            st.metric("Selected Shots", len(selected_shots))
        with col3:
            st.metric("Channels", len(dataset_info["channels"]))
        with col4:
            dt_range = dataset_info["dt_range"]
            if dt_range[0]:
                st.metric("Time Step", f"{dt_range[0]*1e6:.1f} µs")
        
        st.subheader("Quality Summary")
        
        fig = plot_quality_summary(flag_counts)
        show_fig(fig)
        
        st.subheader("Selected Shots")
        
        shot_info = []
        for shot in selected_shots[:20]:
            info = {"Shot ID": shot.shot_id}
            info.update(shot.control_vars)
            
            if shot.shot_id in all_quality:
                flags = []
                for ch_result in all_quality[shot.shot_id].values():
                    flags.extend(ch_result["flags"])
                info["Quality"] = "✅" if not flags else f"⚠️ {len(flags)}"
            shot_info.append(info)
        
        st.dataframe(pd.DataFrame(shot_info), use_container_width=True)
    
    # --- Overlay Plots Tab ---
    with tab_plots:
        st.header("Channel Overlay Plots")
        
        show_mean_std = st.checkbox("Show mean ± std envelope", value=True)
        
        highlight_shot = st.selectbox(
            "Highlight shot",
            options=["None"] + selected_ids,
            index=0
        )
        if highlight_shot == "None":
            highlight_shot = None
        
        for channel in selected_channels:
            st.subheader(channel)
            
            fig = plot_channel_overlay(
                selected_shots,
                channel,
                preprocessed_data=preprocessed.get(channel),
                aligned_times=aligned_times,
                highlight_shot=highlight_shot,
                show_mean_std=show_mean_std,
                title=f"{channel} - {len(selected_shots)} shots",
            )
            show_fig(fig)
    
    # --- Metrics Table Tab ---
    with tab_metrics:
        st.header("Shot Metrics")
        
        metric_channel = st.selectbox(
            "Channel for metrics",
            options=selected_channels,
            key="metric_channel"
        )
        
        rows = []
        for shot in selected_shots:
            row = {"shot_id": shot.shot_id}
            row.update(shot.control_vars)
            
            if shot.shot_id in all_metrics:
                if metric_channel in all_metrics[shot.shot_id]:
                    for metric_name, val in all_metrics[shot.shot_id][metric_channel].items():
                        row[metric_name] = format_metric(val)
            
            if shot.shot_id in all_quality:
                if metric_channel in all_quality[shot.shot_id]:
                    flags = all_quality[shot.shot_id][metric_channel]["flags"]
                    row["quality"] = "✅" if not flags else ", ".join(flags)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "metrics.csv",
            "text/csv"
        )
    
    # --- Trends Tab ---
    with tab_trends:
        st.header("Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_options = ["shot_index"] + list(shots[0].control_vars.keys())
            x_var = st.selectbox("X-axis variable", options=x_options)
        
        with col2:
            y_channel = st.selectbox("Y-axis channel", options=selected_channels,
                                     key="trend_channel")
        
        y_metrics = ["peak", "min", "peak_to_peak", "time_of_peak", "auc", "snr_estimate"]
        y_metric = st.selectbox("Y-axis metric", options=y_metrics)
        
        x_values = []
        y_values = []
        shot_ids_for_trend = []
        
        for i, shot in enumerate(selected_shots):
            shot_ids_for_trend.append(shot.shot_id)
            
            if x_var == "shot_index":
                x_values.append(i)
            else:
                x_values.append(shot.control_vars.get(x_var, np.nan))
            
            if shot.shot_id in all_metrics:
                if y_channel in all_metrics[shot.shot_id]:
                    y_values.append(all_metrics[shot.shot_id][y_channel].get(y_metric, np.nan))
                else:
                    y_values.append(np.nan)
            else:
                y_values.append(np.nan)
        
        x_label = x_var.replace("_", " ").title() if x_var != "shot_index" else "Shot Index"
        y_label = f"{y_channel} {y_metric}".replace("_", " ").title()
        
        show_trendline = st.checkbox("Show trend line", value=True)
        
        fig = plot_trend(
            x_values, y_values, x_label, y_label, shot_ids_for_trend,
            show_trendline=show_trendline,
            title=f"{y_label} vs {x_label}"
        )
        show_fig(fig)
    
    # --- Report Tab ---
    with tab_report:
        st.header("Export Report")
        
        st.subheader("Focus Shots")
        focus_shots = st.multiselect(
            "Select 1-5 shots for detailed report",
            options=selected_ids,
            default=selected_ids[:min(3, len(selected_ids))],
            max_selections=5
        )
        
        notes = st.text_area("Report Notes", placeholder="Add any notes...")
        
        # Trend config for report
        trend_config = {
            "x_var": x_var if 'x_var' in dir() else "shot_index",
            "y_metric": y_metric if 'y_metric' in dir() else "peak",
            "y_channel": y_channel if 'y_channel' in dir() else selected_channels[0],
        }
        
        if st.button("📥 Generate Report", type="primary"):
            if not focus_shots:
                st.warning("Select at least one focus shot")
            else:
                with st.spinner("Generating report..."):
                    focus_shot_objs = [s for s in shots if s.shot_id in focus_shots]
                    
                    # Reconstruct quality results for report
                    quality_for_report = {}
                    for sid in focus_shots:
                        if sid in all_quality:
                            quality_for_report[sid] = {}
                            for ch, data in all_quality[sid].items():
                                class QR:
                                    def __init__(self, flags, details):
                                        self.flags = flags
                                        self.flag_names = flags
                                        self.details = details
                                quality_for_report[sid][ch] = QR(data["flags"], data["details"])
                    
                    # Metrics for report (only selected channels)
                    metrics_for_report = {
                        sid: {ch: all_metrics[sid][ch] 
                              for ch in selected_channels if ch in all_metrics.get(sid, {})}
                        for sid in focus_shots
                    }
                    
                    try:
                        report_dir = generate_report(
                            shots=shots,
                            selected_shots=focus_shot_objs,
                            channel_metrics=metrics_for_report,
                            quality_results=quality_for_report,
                            quality_summary=flag_counts,
                            preprocessed_data=preprocessed,
                            aligned_times=aligned_times,
                            channels=selected_channels,
                            trend_config=trend_config,
                            notes=notes,
                        )
                        
                        st.success(f"Report generated: {report_dir}")
                        
                        # Show report preview with properly displayed images
                        report_path = report_dir / "report.md"
                        plots_dir = report_dir / "plots"
                        
                        if report_path.exists():
                            with st.expander("Preview Report", expanded=True):
                                # Read markdown and display sections with images inline
                                md_content = report_path.read_text()
                                
                                # Split by image references and display properly
                                import re
                                parts = re.split(r'!\[([^\]]*)\]\(([^)]+)\)', md_content)
                                
                                # parts = [text, alt1, path1, text, alt2, path2, ...]
                                i = 0
                                while i < len(parts):
                                    if i % 3 == 0:
                                        # Regular markdown text
                                        if parts[i].strip():
                                            st.markdown(parts[i])
                                    elif i % 3 == 2:
                                        # Image path
                                        img_path = report_dir / parts[i]
                                        if img_path.exists():
                                            st.image(str(img_path), use_container_width=True)
                                        else:
                                            st.warning(f"Image not found: {parts[i]}")
                                    i += 1
                        
                        # Offer download of the report folder as zip
                        st.subheader("Download Report")
                        
                        # Create zip of report
                        import shutil
                        zip_path = report_dir.parent / f"{report_dir.name}.zip"
                        shutil.make_archive(str(report_dir), 'zip', report_dir.parent, report_dir.name)
                        
                        with open(zip_path, 'rb') as f:
                            st.download_button(
                                label="📦 Download Report (ZIP)",
                                data=f.read(),
                                file_name=f"{report_dir.name}.zip",
                                mime="application/zip"
                            )
                    
                    except Exception as e:
                        st.error(f"Failed to generate report: {e}")
                        raise e


if __name__ == "__main__":
    main()
