"""
Report generation utilities.

Generates Markdown reports with embedded plots and metrics tables.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import Shot
from .plotting import plot_channel_overlay, plot_trend, plot_quality_summary, save_figure


def generate_report(
    shots: list[Shot],
    selected_shots: list[Shot],
    channel_metrics: dict[str, dict[str, dict]],  # shot_id -> channel -> metrics
    quality_results: dict[str, dict],  # shot_id -> channel -> QualityResult
    quality_summary: dict[str, int],
    preprocessed_data: dict[str, dict[str, np.ndarray]],  # channel -> shot_id -> data
    aligned_times: dict[str, np.ndarray],  # shot_id -> time
    channels: list[str],
    trend_config: Optional[dict] = None,  # {x_var, y_metric, y_channel}
    output_dir: Optional[Path] = None,
    notes: str = "",
) -> Path:
    """
    Generate a complete analysis report.
    
    Args:
        shots: All loaded shots
        selected_shots: Shots selected for detailed analysis
        channel_metrics: Computed metrics per shot/channel
        quality_results: Quality assessment results
        quality_summary: Summary flag counts
        preprocessed_data: Preprocessed signal data
        aligned_times: Time-aligned time arrays
        channels: Channels to include
        trend_config: Configuration for trend plot
        output_dir: Output directory (default: reports/report_<timestamp>)
        notes: Additional notes to include
    
    Returns:
        Path to generated report directory
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("reports") / f"report_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    # Generate plots
    plot_paths = {}
    
    # Overlay plots for each channel
    for channel in channels:
        fig = plot_channel_overlay(
            selected_shots,
            channel,
            preprocessed_data=preprocessed_data.get(channel),
            aligned_times=aligned_times,
            show_mean_std=True,
            title=f"{channel} - Selected Shots Overlay",
        )
        plot_path = plots_dir / f"overlay_{channel}.png"
        save_figure(fig, str(plot_path))
        plot_paths[f"overlay_{channel}"] = plot_path.relative_to(output_dir)
    
    # Quality summary plot
    fig = plot_quality_summary(quality_summary)
    plot_path = plots_dir / "quality_summary.png"
    save_figure(fig, str(plot_path))
    plot_paths["quality_summary"] = plot_path.relative_to(output_dir)
    
    # Trend plot if configured
    if trend_config is not None:
        x_var = trend_config.get("x_var", "shot_index")
        y_metric = trend_config.get("y_metric", "peak")
        y_channel = trend_config.get("y_channel", channels[0] if channels else "b_dot")
        
        x_values = []
        y_values = []
        shot_ids = []
        
        for i, shot in enumerate(selected_shots):
            shot_ids.append(shot.shot_id)
            
            # Get x value
            if x_var == "shot_index":
                x_values.append(i)
            else:
                x_values.append(shot.control_vars.get(x_var, np.nan))
            
            # Get y value (metric)
            if shot.shot_id in channel_metrics:
                if y_channel in channel_metrics[shot.shot_id]:
                    metrics = channel_metrics[shot.shot_id][y_channel]
                    y_values.append(metrics.get(y_metric, np.nan))
                else:
                    y_values.append(np.nan)
            else:
                y_values.append(np.nan)
        
        x_label = x_var.replace("_", " ").title() if x_var != "shot_index" else "Shot Index"
        y_label = f"{y_channel} {y_metric}".replace("_", " ").title()
        
        fig = plot_trend(
            x_values, y_values, x_label, y_label, shot_ids,
            show_trendline=True,
            title=f"Trend: {y_label} vs {x_label}",
        )
        plot_path = plots_dir / "trend.png"
        save_figure(fig, str(plot_path))
        plot_paths["trend"] = plot_path.relative_to(output_dir)
    
    # Generate metrics CSV
    metrics_rows = []
    for shot in selected_shots:
        row = {
            "shot_id": shot.shot_id,
            **shot.control_vars,
        }
        
        if shot.shot_id in channel_metrics:
            for channel, metrics in channel_metrics[shot.shot_id].items():
                for metric_name, value in metrics.items():
                    row[f"{channel}_{metric_name}"] = value
        
        # Add quality flags
        if shot.shot_id in quality_results:
            flags = []
            for ch, result in quality_results[shot.shot_id].items():
                flags.extend([f"{ch}:{f}" for f in result.flag_names])
            row["quality_flags"] = "; ".join(flags) if flags else "clean"
        
        metrics_rows.append(row)
    
    metrics_df = pd.DataFrame(metrics_rows)
    csv_path = tables_dir / "metrics_selected.csv"
    metrics_df.to_csv(csv_path, index=False)
    
    # Generate Markdown report
    md_lines = [
        f"# Shot Analysis Report",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"---",
        f"",
        f"## Dataset Summary",
        f"",
        f"- **Total Shots in Dataset:** {len(shots)}",
        f"- **Selected Shots for Analysis:** {len(selected_shots)}",
        f"- **Channels Analyzed:** {', '.join(channels)}",
        f"",
        f"### Selected Shots",
        f"",
    ]
    
    # List selected shots with control vars
    for shot in selected_shots[:10]:  # Limit to first 10
        cv_str = ", ".join(f"{k}={v}" for k, v in shot.control_vars.items())
        md_lines.append(f"- **Shot {shot.shot_id}**: {cv_str}")
    
    if len(selected_shots) > 10:
        md_lines.append(f"- ... and {len(selected_shots) - 10} more")
    
    md_lines.extend([
        f"",
        f"---",
        f"",
        f"## Quality Assessment",
        f"",
        f"![Quality Summary]({plot_paths.get('quality_summary', 'plots/quality_summary.png')})",
        f"",
    ])
    
    # Add quality flag summary
    md_lines.append("| Flag | Count |")
    md_lines.append("|------|-------|")
    for flag, count in sorted(quality_summary.items()):
        md_lines.append(f"| {flag.replace('_', ' ').title()} | {count} |")
    
    md_lines.extend([
        f"",
        f"---",
        f"",
        f"## Channel Overlay Plots",
        f"",
    ])
    
    # Add overlay plots
    for channel in channels:
        plot_key = f"overlay_{channel}"
        if plot_key in plot_paths:
            md_lines.append(f"### {channel}")
            md_lines.append(f"")
            md_lines.append(f"![{channel} Overlay]({plot_paths[plot_key]})")
            md_lines.append(f"")
    
    # Add trend plot if present
    if "trend" in plot_paths:
        md_lines.extend([
            f"---",
            f"",
            f"## Trend Analysis",
            f"",
            f"![Trend Plot]({plot_paths['trend']})",
            f"",
        ])
    
    md_lines.extend([
        f"---",
        f"",
        f"## Metrics Summary",
        f"",
        f"Full metrics available in `tables/metrics_selected.csv`",
        f"",
    ])
    
    # Add brief metrics table (first few columns)
    if not metrics_df.empty:
        display_cols = ["shot_id"] + list(metrics_df.columns[1:6])  # First 5 data columns
        display_cols = [c for c in display_cols if c in metrics_df.columns]
        
        md_lines.append("| " + " | ".join(display_cols) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
        
        for _, row in metrics_df.head(10).iterrows():
            values = []
            for col in display_cols:
                val = row[col]
                if isinstance(val, float):
                    if np.isnan(val):
                        val = "N/A"
                    elif abs(val) > 1000 or (abs(val) < 0.01 and val != 0):
                        val = f"{val:.2e}"
                    else:
                        val = f"{val:.3f}"
                values.append(str(val))
            md_lines.append("| " + " | ".join(values) + " |")
    
    # Notes section
    md_lines.extend([
        f"",
        f"---",
        f"",
        f"## Notes",
        f"",
        notes if notes else "*No additional notes.*",
        f"",
        f"---",
        f"",
        f"*Report generated by Shot Explorer*",
    ])
    
    # Write report
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(md_lines))
    
    return output_dir
