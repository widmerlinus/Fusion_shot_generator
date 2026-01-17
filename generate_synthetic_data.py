#!/usr/bin/env python3
"""
Synthetic plasma shot data generator CLI.

Generates realistic-looking multi-channel shot time series with noise,
drift, missing data, saturation, and shot-to-shot variation.

Usage:
    python generate_synthetic_data.py --n_shots 60 --out_dir data/shots --seed 123
"""

import argparse
from pathlib import Path

from shotlib.synthetic import generate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic plasma shot data for testing and demonstration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_synthetic_data.py --n_shots 30
    python generate_synthetic_data.py --n_shots 100 --out_dir my_data/shots --seed 42
    
The generator creates:
    - CSV files with time-series data for 4 channels (b_dot, interferometer, photodiode, xray_proxy)
    - JSON metadata files with control variables and generation info
    - Realistic imperfections: missing data, saturation, drift, noise
        """
    )
    
    parser.add_argument(
        "--n_shots", "-n",
        type=int,
        default=30,
        help="Number of shots to generate (default: 30)"
    )
    
    parser.add_argument(
        "--out_dir", "-o",
        type=str,
        default="data/shots",
        help="Output directory for shot files (default: data/shots)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)"
    )
    
    args = parser.parse_args()
    
    out_path = Path(args.out_dir)
    
    print(f"Generating synthetic plasma shot data...")
    print(f"  Shots: {args.n_shots}")
    print(f"  Output: {out_path.absolute()}")
    print(f"  Seed: {args.seed if args.seed is not None else 'random'}")
    print()
    
    generate_dataset(
        n_shots=args.n_shots,
        out_dir=out_path,
        seed=args.seed,
    )
    
    print()
    print("Done! You can now run the dashboard:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()
