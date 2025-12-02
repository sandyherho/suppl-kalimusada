#!/usr/bin/env python3
"""
Figure: 3D Phase Space Dynamics
================================
2x2 panel figure showing the Ma-Chen strange attractor for each test case.
Economy A and B trajectories shown with divergence visualization.

Author: Sandy H. S. Herho
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from netCDF4 import Dataset
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / 'data'
FIGS_DIR = Path(__file__).parent.parent / 'figs'
STATS_DIR = Path(__file__).parent.parent / 'stats'

FIGS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# Case files and labels
CASES = {
    'case1': ('case1_standard_chaos.nc', 'Case 1: Standard'),
    'case2': ('case2_high_sensitivity.nc', 'Case 2: High Sensitivity'),
    'case3': ('case3_modified_parameters.nc', 'Case 3: Modified Parameters'),
    'case4': ('case4_long_term_evolution.nc', 'Case 4: Long-Term'),
}

# Publication-quality colors
COLOR_A = '#0077BB'   # Blue for Economy A
COLOR_B = '#EE3377'   # Magenta for Economy B

# =============================================================================
# Matplotlib Configuration for Publication Quality
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 0.6,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'text.usetex': False,
    'mathtext.fontset': 'dejavuserif',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})


def load_case_data(case_name: str) -> dict:
    """Load NetCDF data for a specific case."""
    filename, label = CASES[case_name]
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    with Dataset(filepath, 'r') as nc:
        data = {
            'time': nc.variables['time'][:],
            'A_x': nc.variables['economy_A_x'][:],
            'A_y': nc.variables['economy_A_y'][:],
            'A_z': nc.variables['economy_A_z'][:],
            'B_x': nc.variables['economy_B_x'][:],
            'B_y': nc.variables['economy_B_y'][:],
            'B_z': nc.variables['economy_B_z'][:],
            'euclidean_distance': nc.variables['euclidean_distance'][:],
            # Global attributes
            'parameter_a': nc.parameter_a,
            'parameter_b': nc.parameter_b,
            'parameter_c': nc.parameter_c,
            'init_perturbation': abs(nc.init_B_x - nc.init_A_x),
            't_end': nc.t_end,
            'label': label,
        }
    
    return data


def compute_statistics(all_data: dict) -> dict:
    """Compute attractor statistics for all cases."""
    stats = {}
    
    for case_name, data in all_data.items():
        x_A, y_A, z_A = data['A_x'], data['A_y'], data['A_z']
        
        case_stats = {
            'case_label': data['label'],
            'parameters': f"a={data['parameter_a']}, b={data['parameter_b']}, c={data['parameter_c']}",
            'initial_perturbation': data['init_perturbation'],
            't_end': data['t_end'],
            # Attractor extent (Economy A)
            'x_min': np.min(x_A),
            'x_max': np.max(x_A),
            'x_range': np.max(x_A) - np.min(x_A),
            'y_min': np.min(y_A),
            'y_max': np.max(y_A),
            'y_range': np.max(y_A) - np.min(y_A),
            'z_min': np.min(z_A),
            'z_max': np.max(z_A),
            'z_range': np.max(z_A) - np.min(z_A),
            # Standard deviations
            'x_std': np.std(x_A),
            'y_std': np.std(y_A),
            'z_std': np.std(z_A),
            # Attractor "volume" (bounding box)
            'bounding_box_volume': (np.max(x_A) - np.min(x_A)) * 
                                   (np.max(y_A) - np.min(y_A)) * 
                                   (np.max(z_A) - np.min(z_A)),
        }
        
        stats[case_name] = case_stats
    
    return stats


def save_statistics(stats: dict, filepath: Path):
    """Save attractor statistics to text file."""
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("3D PHASE SPACE DYNAMICS - ATTRACTOR STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        for case_name, case_stats in stats.items():
            f.write(f"{case_stats['case_label']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Parameters:           {case_stats['parameters']}\n")
            f.write(f"  Initial perturbation: {case_stats['initial_perturbation']:.2e}\n")
            f.write(f"  Simulation time:      {case_stats['t_end']:.0f}\n")
            f.write("\n  Attractor Extent:\n")
            f.write(f"    x (interest rate):  [{case_stats['x_min']:.4f}, {case_stats['x_max']:.4f}], range = {case_stats['x_range']:.4f}\n")
            f.write(f"    y (investment):     [{case_stats['y_min']:.4f}, {case_stats['y_max']:.4f}], range = {case_stats['y_range']:.4f}\n")
            f.write(f"    z (price index):    [{case_stats['z_min']:.4f}, {case_stats['z_max']:.4f}], range = {case_stats['z_range']:.4f}\n")
            f.write("\n  Standard Deviations:\n")
            f.write(f"    sigma_x = {case_stats['x_std']:.4f}\n")
            f.write(f"    sigma_y = {case_stats['y_std']:.4f}\n")
            f.write(f"    sigma_z = {case_stats['z_std']:.4f}\n")
            f.write(f"\n  Bounding Box Volume: {case_stats['bounding_box_volume']:.4f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Notes:\n")
        f.write("  - Attractor extent computed from Economy A trajectory\n")
        f.write("  - Bounding box volume = delta_x * delta_y * delta_z\n")
        f.write("=" * 80 + "\n")
    
    print(f"Statistics saved to: {filepath}")


def create_figure(all_data: dict, stats: dict):
    """Create the 2x2 3D phase space figure."""
    
    fig = plt.figure(figsize=(9, 9), facecolor='white')
    
    # Panel labels as subtitles
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    case_order = ['case1', 'case2', 'case3', 'case4']
    
    # View angles for each panel
    view_angles = [
        (25, 45),   # Case 1
        (20, 135),  # Case 2
        (30, -60),  # Case 3
        (25, 60),   # Case 4
    ]
    
    # Collect legend handles
    legend_handles = []
    legend_labels = []
    
    for idx, case_name in enumerate(case_order):
        if case_name not in all_data:
            continue
        
        data = all_data[case_name]
        case_stats = stats[case_name]
        
        # Create 3D subplot
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d', facecolor='white')
        
        # Subtitle above each panel
        ax.set_title(panel_labels[idx], fontsize=14, fontweight='bold', pad=10, loc='center')
        
        # Subsample trajectory for cleaner visualization
        n = len(data['time'])
        skip = max(1, n // 5000)
        
        # Select time window (skip transient, use developed attractor)
        t_start_frac = 0.1
        t_end_frac = 0.9
        start_idx = int(n * t_start_frac)
        end_idx = int(n * t_end_frac)
        idx_range = slice(start_idx, end_idx, skip)
        
        # Plot Economy A (main trajectory)
        line_a, = ax.plot(data['A_x'][idx_range], 
                          data['A_y'][idx_range], 
                          data['A_z'][idx_range],
                          color=COLOR_A, linewidth=0.4, alpha=0.8)
        
        # Plot Economy B (perturbed trajectory)
        line_b, = ax.plot(data['B_x'][idx_range], 
                          data['B_y'][idx_range], 
                          data['B_z'][idx_range],
                          color=COLOR_B, linewidth=0.4, alpha=0.6)
        
        # Collect legend handles only once
        if idx == 0:
            legend_handles = [line_a, line_b]
            legend_labels = ['Economy A', 'Economy B (Perturbed)']
        
        # Set view angle
        ax.view_init(elev=view_angles[idx][0], azim=view_angles[idx][1])
        
        # Labels with bold
        ax.set_xlabel(r'$\mathbf{x}$', fontweight='bold', labelpad=5, fontsize=11)
        ax.set_ylabel(r'$\mathbf{y}$', fontweight='bold', labelpad=5, fontsize=11)
        ax.set_zlabel(r'$\mathbf{z}$', fontweight='bold', labelpad=5, fontsize=11)
        
        # Set axis limits with padding
        pad = 0.15
        x_lim = [case_stats['x_min'] - pad, case_stats['x_max'] + pad]
        y_lim = [case_stats['y_min'] - pad, case_stats['y_max'] + pad]
        z_lim = [case_stats['z_min'] - pad, case_stats['z_max'] + pad]
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        
        # Reduce tick density
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.zaxis.set_major_locator(plt.MaxNLocator(4))
        
        # White panes with light edges
        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_facecolor('white')
        ax.yaxis.pane.set_facecolor('white')
        ax.zaxis.pane.set_facecolor('white')
        ax.xaxis.pane.set_edgecolor((0.7, 0.7, 0.7, 1.0))
        ax.yaxis.pane.set_edgecolor((0.7, 0.7, 0.7, 1.0))
        ax.zaxis.pane.set_edgecolor((0.7, 0.7, 0.7, 1.0))
        
        # Light grid
        ax.xaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 1.0)
        ax.yaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 1.0)
        ax.zaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 1.0)
        
        # Bold tick labels
        ax.tick_params(axis='x', labelsize=9, pad=2)
        ax.tick_params(axis='y', labelsize=9, pad=2)
        ax.tick_params(axis='z', labelsize=9, pad=2)
        
        # Make tick labels bold by setting them manually
        for label in ax.xaxis.get_ticklabels():
            label.set_fontweight('bold')
        for label in ax.yaxis.get_ticklabels():
            label.set_fontweight('bold')
        for label in ax.zaxis.get_ticklabels():
            label.set_fontweight('bold')
    
    # Adjust layout to make room for legend
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.10, hspace=0.15, wspace=0.10)
    
    # Bottom Legend (shared for all panels)
    fig.legend(legend_handles, legend_labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.01),
               ncol=2, 
               fontsize=11,
               frameon=True,
               fancybox=False,
               edgecolor='0.3',
               framealpha=1.0,
               handlelength=2.5,
               handletextpad=0.5,
               columnspacing=2.0,
               prop={'weight': 'bold'})
    
    # Save figures
    pdf_path = FIGS_DIR / 'phase_space_3d.pdf'
    png_path = FIGS_DIR / 'phase_space_3d.png'
    
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300, facecolor='white')
    
    print(f"Figure saved to: {pdf_path}")
    print(f"Figure saved to: {png_path}")
    
    plt.close(fig)


def main():
    """Main execution function."""
    print("=" * 60)
    print("3D Phase Space Dynamics")
    print("=" * 60)
    
    # Load all case data
    print("\nLoading data...")
    all_data = {}
    for case_name in CASES.keys():
        try:
            all_data[case_name] = load_case_data(case_name)
            print(f"  Loaded: {case_name} (t_end = {all_data[case_name]['t_end']})")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
    
    if not all_data:
        raise RuntimeError("No data files found!")
    
    # Compute statistics
    print("\nComputing attractor statistics...")
    stats = compute_statistics(all_data)
    
    # Save statistics
    stats_path = STATS_DIR / 'phase_space_3d_stats.txt'
    save_statistics(stats, stats_path)
    
    # Create figure
    print("\nGenerating figure...")
    create_figure(all_data, stats)
    
    print("\nDone!")
    print("=" * 60)


if __name__ == '__main__':
    main()
