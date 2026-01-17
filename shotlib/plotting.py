"""
Plotting utilities for shot visualization.

Provides functions for overlay plots, metrics visualization, and trend analysis.
"""

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# Consistent color palette for shots
SHOT_COLORS = list(mcolors.TABLEAU_COLORS.values())


def get_shot_color(index: int) -> str:
    """Get consistent color for a shot by index."""
    return SHOT_COLORS[index % len(SHOT_COLORS)]


def plot_channel_overlay(
    shots: list,  # List of Shot objects
    channel: str,
    preprocessed_data: Optional[dict[str, np.ndarray]] = None,
    aligned_times: Optional[dict[str, np.ndarray]] = None,
    highlight_shot: Optional[str] = None,
    show_mean_std: bool = False,
    title: Optional[str] = None,
    units: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """
    Create overlay plot of multiple shots for a single channel.
    
    Args:
        shots: List of Shot objects to plot
        channel: Channel name to plot
        preprocessed_data: Optional dict mapping shot_id -> preprocessed data
        aligned_times: Optional dict mapping shot_id -> aligned time array
        highlight_shot: Shot ID to highlight (thicker line)
        show_mean_std: Whether to show mean ± std envelope
        title: Plot title (default: channel name)
        units: Y-axis units label
        ax: Existing axes to plot on
        figsize: Figure size if creating new figure
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Collect data for mean/std calculation
    all_times = []
    all_data = []
    
    for i, shot in enumerate(shots):
        if channel not in shot.channels:
            continue
        
        # Get time and data
        if aligned_times is not None and shot.shot_id in aligned_times:
            t = aligned_times[shot.shot_id]
        else:
            t = shot.time
        
        if preprocessed_data is not None and shot.shot_id in preprocessed_data:
            data = preprocessed_data[shot.shot_id]
        else:
            data = shot.get_channel(channel)
        
        # Plot styling
        color = get_shot_color(i)
        alpha = 0.7 if highlight_shot is None else (1.0 if shot.shot_id == highlight_shot else 0.3)
        linewidth = 2.0 if shot.shot_id == highlight_shot else 1.0
        
        # Limit legend entries
        label = f"Shot {shot.shot_id}" if i < 10 else None
        
        ax.plot(t * 1e6, data, color=color, alpha=alpha, linewidth=linewidth, label=label)
        
        all_times.append(t)
        all_data.append(data)
    
    # Show mean ± std if requested
    if show_mean_std and len(all_data) > 1:
        # Interpolate to common time grid
        t_min = max(t.min() for t in all_times)
        t_max = min(t.max() for t in all_times)
        t_common = np.linspace(t_min, t_max, 500)
        
        interpolated = []
        for t, d in zip(all_times, all_data):
            valid = ~np.isnan(d)
            if np.sum(valid) > 2:
                interp = np.interp(t_common, t[valid], d[valid])
                interpolated.append(interp)
        
        if len(interpolated) > 1:
            data_array = np.array(interpolated)
            mean = np.nanmean(data_array, axis=0)
            std = np.nanstd(data_array, axis=0)
            
            ax.fill_between(
                t_common * 1e6, mean - std, mean + std,
                alpha=0.2, color='black', label='Mean ± Std'
            )
            ax.plot(t_common * 1e6, mean, 'k--', linewidth=2, label='Mean')
    
    # Labels and formatting
    ax.set_xlabel("Time (µs)")
    ylabel = channel
    if units:
        ylabel += f" ({units})"
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{channel} - {len(shots)} shots")
    
    ax.grid(True, alpha=0.3)
    
    # Only show legend if reasonable number of entries
    if len(shots) <= 10 or show_mean_std:
        ax.legend(loc='upper right', fontsize=8)
    
    fig.tight_layout()
    return fig


def plot_metrics_table(
    metrics_data: list[dict],
    columns: list[str],
    title: str = "Shot Metrics",
    figsize: tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Create a table visualization of metrics.
    
    Args:
        metrics_data: List of dicts with metrics per shot
        columns: Column names to display
        title: Table title
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Build table data
    table_data = []
    for row in metrics_data:
        table_row = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                if np.isnan(val):
                    val = "N/A"
                elif abs(val) > 1000 or (abs(val) < 0.01 and val != 0):
                    val = f"{val:.2e}"
                else:
                    val = f"{val:.3f}"
            table_row.append(val)
        table_data.append(table_row)
    
    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    fig.tight_layout()
    
    return fig


def plot_trend(
    x_values: list[float],
    y_values: list[float],
    x_label: str,
    y_label: str,
    shot_ids: list[str],
    highlight_shot: Optional[str] = None,
    show_trendline: bool = True,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Create scatter plot showing metric trends.
    
    Args:
        x_values: X-axis values
        y_values: Y-axis values (metric)
        x_label: X-axis label
        y_label: Y-axis label
        shot_ids: Shot IDs for hover/annotation
        highlight_shot: Shot ID to highlight
        show_trendline: Whether to show linear trend line
        title: Plot title
        ax: Existing axes
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert to arrays and filter NaN
    x = np.array(x_values)
    y = np.array(y_values)
    valid = ~(np.isnan(x) | np.isnan(y))
    
    # Plot points
    colors = ['red' if sid == highlight_shot else 'steelblue' 
              for sid in shot_ids]
    sizes = [100 if sid == highlight_shot else 50 for sid in shot_ids]
    
    scatter = ax.scatter(x[valid], y[valid], 
                        c=[c for c, v in zip(colors, valid) if v],
                        s=[s for s, v in zip(sizes, valid) if v],
                        alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Add trendline
    if show_trendline and np.sum(valid) > 2:
        x_valid = x[valid]
        y_valid = y[valid]
        
        # Linear regression
        coeffs = np.polyfit(x_valid, y_valid, 1)
        x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        
        ax.plot(x_line, y_line, 'r--', alpha=0.5, linewidth=2, label='Linear fit')
        
        # Calculate R²
        y_pred = np.polyval(coeffs, x_valid)
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - y_valid.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{y_label} vs {x_label}")
    
    ax.grid(True, alpha=0.3)
    
    if show_trendline:
        ax.legend(loc='upper right')
    
    fig.tight_layout()
    return fig


def plot_quality_summary(
    flag_counts: dict[str, int],
    title: str = "Quality Flag Summary",
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Create bar chart of quality flag counts.
    
    Args:
        flag_counts: Dict mapping flag name -> count
        title: Chart title
        ax: Existing axes
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Sort by count
    sorted_flags = sorted(flag_counts.items(), key=lambda x: -x[1])
    names = [f[0].replace('_', ' ').title() for f in sorted_flags]
    counts = [f[1] for f in sorted_flags]
    
    # Color by severity
    colors = []
    for name, count in sorted_flags:
        if count == 0:
            colors.append('lightgreen')
        elif count < 5:
            colors.append('yellow')
        else:
            colors.append('salmon')
    
    bars = ax.barh(names, counts, color=colors, edgecolor='gray')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10)
    
    ax.set_xlabel("Count")
    ax.set_title(title)
    ax.invert_yaxis()  # Highest at top
    
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save figure to file."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
