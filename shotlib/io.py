"""
Data I/O utilities for loading and parsing shot files.

Supports CSV format with optional JSON metadata sidecars.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Shot:
    """
    Container for a single shot's data and metadata.
    
    Attributes:
        shot_id: Unique identifier for the shot
        data: DataFrame with time column 't' and channel columns
        metadata: Optional metadata dict from sidecar file
        filepath: Path to the source CSV file
    """
    shot_id: str
    data: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    filepath: Optional[Path] = None
    
    @property
    def time(self) -> np.ndarray:
        """Return time array."""
        return self.data["t"].values
    
    @property
    def dt(self) -> float:
        """Return time step."""
        t = self.time
        if len(t) > 1:
            return float(np.median(np.diff(t)))
        return 1e-6
    
    @property
    def channels(self) -> list[str]:
        """Return list of data channel names (excluding time)."""
        return [c for c in self.data.columns if c != "t"]
    
    @property
    def control_vars(self) -> dict:
        """Return control variables from metadata."""
        return self.metadata.get("control_vars", {})
    
    def get_channel(self, channel: str) -> np.ndarray:
        """Get data for a specific channel."""
        if channel not in self.data.columns:
            raise ValueError(f"Channel '{channel}' not found in shot {self.shot_id}")
        return self.data[channel].values


def load_shot(csv_path: Path) -> Shot:
    """
    Load a single shot from CSV file.
    
    Args:
        csv_path: Path to the shot CSV file
    
    Returns:
        Shot object with data and metadata
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV is malformed
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Shot file not found: {csv_path}")
    
    # Load CSV data
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV {csv_path}: {e}")
    
    if "t" not in data.columns:
        raise ValueError(f"CSV must have 't' (time) column: {csv_path}")
    
    # Extract shot_id from filename
    shot_id = csv_path.stem.replace("shot_", "")
    
    # Try to load metadata sidecar
    metadata = {}
    meta_path = csv_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass  # Ignore malformed metadata
    
    return Shot(
        shot_id=shot_id,
        data=data,
        metadata=metadata,
        filepath=csv_path,
    )


def discover_shots(data_dir: Path) -> list[Path]:
    """
    Discover all shot CSV files in a directory.
    
    Args:
        data_dir: Directory to search for shot files
    
    Returns:
        List of paths to shot CSV files, sorted by name
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        return []
    
    # Find all CSV files matching shot pattern
    shot_files = sorted(data_dir.glob("shot_*.csv"))
    
    return shot_files


def load_dataset(data_dir: Path) -> list[Shot]:
    """
    Load all shots from a directory.
    
    Args:
        data_dir: Directory containing shot files
    
    Returns:
        List of Shot objects, sorted by shot_id
    """
    shot_files = discover_shots(data_dir)
    
    shots = []
    for path in shot_files:
        try:
            shot = load_shot(path)
            shots.append(shot)
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Skipping {path}: {e}")
    
    return shots


def get_dataset_info(shots: list[Shot]) -> dict:
    """
    Get summary information about a dataset.
    
    Args:
        shots: List of Shot objects
    
    Returns:
        Dictionary with dataset statistics
    """
    if not shots:
        return {
            "n_shots": 0,
            "channels": [],
            "dt_range": (None, None),
            "duration_range": (None, None),
        }
    
    # Collect info across all shots
    all_channels = set()
    dts = []
    durations = []
    
    for shot in shots:
        all_channels.update(shot.channels)
        dts.append(shot.dt)
        t = shot.time
        if len(t) > 0:
            durations.append(t[-1] - t[0])
    
    return {
        "n_shots": len(shots),
        "channels": sorted(all_channels),
        "dt_range": (min(dts), max(dts)) if dts else (None, None),
        "duration_range": (min(durations), max(durations)) if durations else (None, None),
        "shot_ids": [s.shot_id for s in shots],
    }
