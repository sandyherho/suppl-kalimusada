#!/usr/bin/env python
"""
Figure: Divergence Time Series Comparison
==========================================
Overlays Euclidean distance and log-divergence evolution for all 4 test cases.
Shows exponential growth characteristic of chaotic dynamics.

Author: Sandy H. S. Herho
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
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

# Publication-quality color palette (colorblind-friendly)
COLORS = {
    'case1': '#0077BB',  # Blue
    'case2': '#EE7733',  # Orange
    'case3': '#009988',  # Teal
    'case4': '#CC3311',  # Red
}

# Line styles for distinction
LINESTYLES = {
    'case1': '-',
    'case2': '--',
    'case3': '-.',
    'case4': ':',
}

# =============================================================================
# Matplotlib Configuration for Publication Quality
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.4,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 2.5,
    'ytick.minor.size': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'legend.framealpha': 1.0,
    'legend.edgecolor': '0.3',
    'legend.fancybox': False,
    'text.usetex': False,
    'mathtext.fontset': 'dejavuserif',
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
            'euclidean_distance': nc.variables['euclidean_distance'][:],
            'log_divergence': nc.variables['log_divergence'][:],
            'relative_error': nc.variables['relative_error'][:],
            'rmse_windowed': nc.variables['rmse_windowed'][:],
            # Global attributes
            'parameter_a': nc.parameter_a,
            'parameter_b': nc.parameter_b,
            'parameter_c': nc.parameter_c,
            'init_perturbation': abs(nc.init_B_x - nc.init_A_x),
            'max_euclidean_distance': nc.max_euclidean_distance,
            'mean_euclidean_distance': nc.mean_euclidean_distance,
            'final_euclidean_distance': nc.final_euclidean_distance,
            't_end': nc.t_end,
            'label': label,
        }
    
    return data


def compute_statistics(all_data: dict) -> dict:
    """Compute comprehensive statistics for all cases."""
    stats = {}
    
    for case_name, data in all_data.items():
        time = data['time']
        euc_dist = data['euclidean_distance']
        log_div = data['log_divergence']
        
        # Basic statistics
        case_stats = {
            'case_label': data['label'],
            'parameters': f"a={data['parameter_a']}, b={data['parameter_b']}, c={data['parameter_c']}",
            'initial_perturbation': data['init_perturbation'],
            'time_span': f"[{time[0]:.1f}, {time[-1]:.1f}]",
            'n_points': len(time),
            'max_euclidean_distance': data['max_euclidean_distance'],
            'mean_euclidean_distance': data['mean_euclidean_distance'],
            'final_euclidean_distance': data['final_euclidean_distance'],
        }
        
        # Estimate Lyapunov exponent from exponential growth phase
        valid_mask = ~np.isinf(log_div) & ~np.isnan(log_div) & (log_div < 0)
        if np.sum(valid_mask) > 50:
            t_valid = time[valid_mask]
            ld_valid = log_div[valid_mask]
            n_fit = min(len(t_valid), len(t_valid) // 2)
            if n_fit > 20:
                coeffs = np.polyfit(t_valid[:n_fit], ld_valid[:n_fit], 1)
                lyap_estimate = coeffs[0] * np.log(10)
                case_stats['lyapunov_exponent_estimate'] = lyap_estimate
                case_stats['lyapunov_fit_r2'] = 1 - np.var(ld_valid[:n_fit] - np.polyval(coeffs, t_valid[:n_fit])) / np.var(ld_valid[:n_fit])
            else:
                case_stats['lyapunov_exponent_estimate'] = np.nan
                case_stats['lyapunov_fit_r2'] = np.nan
        else:
            case_stats['lyapunov_exponent_estimate'] = np.nan
            case_stats['lyapunov_fit_r2'] = np.nan
        
        # Time to reach significant divergence
        threshold_mask = euc_dist > 0.1
        if np.any(threshold_mask):
            case_stats['time_to_divergence_0.1'] = time[np.argmax(threshold_mask)]
        else:
            case_stats['time_to_divergence_0.1'] = np.nan
        
        sat_mask = euc_dist > 1.0
        if np.any(sat_mask):
            case_stats['time_to_saturation_1.0'] = time[np.argmax(sat_mask)]
        else:
            case_stats['time_to_saturation_1.0'] = np.nan
        
        case_stats['max_relative_error'] = np.max(data['relative_error'])
        case_stats['mean_relative_error'] = np.mean(data['relative_error'])
        
        stats[case_name] = case_stats
    
    return stats


def save_statistics(stats: dict, filepath: Path):
    """Save statistics to text file."""
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DIVERGENCE TIME SERIES - STATISTICAL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for case_name, case_stats in stats.items():
            f.write(f"{case_stats['case_label']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Parameters:                    {case_stats['parameters']}\n")
            f.write(f"  Initial perturbation:          {case_stats['initial_perturbation']:.2e}\n")
            f.write(f"  Time span:                     {case_stats['time_span']}\n")
            f.write(f"  Number of points:              {case_stats['n_points']:,}\n")
            f.write(f"  Max Euclidean distance:        {case_stats['max_euclidean_distance']:.6f}\n")
            f.write(f"  Mean Euclidean distance:       {case_stats['mean_euclidean_distance']:.6f}\n")
            f.write(f"  Final Euclidean distance:      {case_stats['final_euclidean_distance']:.6f}\n")
            f.write(f"  Lyapunov exponent (est.):      {case_stats['lyapunov_exponent_estimate']:.4f}\n")
            f.write(f"  Lyapunov fit R²:               {case_stats['lyapunov_fit_r2']:.4f}\n")
            f.write(f"  Time to d > 0.1:               {case_stats['time_to_divergence_0.1']:.2f}\n")
            f.write(f"  Time to d > 1.0:               {case_stats['time_to_saturation_1.0']:.2f}\n")
            f.write(f"  Max relative error:            {case_stats['max_relative_error']:.4e}\n")
            f.write(f"  Mean relative error:           {case_stats['mean_relative_error']:.4e}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Notes:\n")
        f.write("  - Lyapunov exponent estimated from linear fit to log10(d) during exponential growth\n")
        f.write("  - Positive lambda indicates chaotic dynamics with exponential divergence\n")
        f.write("=" * 80 + "\n")
    
    print(f"Statistics saved to: {filepath}")


def create_figure(all_data: dict, stats: dict):
    """Create the publication-quality divergence comparison figure (3x1 layout)."""
    
    # Find maximum time across all cases
    t_max_global = max(data['time'][-1] for data in all_data.values())
    
    # Create figure with 3x1 layout and space for bottom legend
    fig = plt.figure(figsize=(7.5, 9.5))
    
    # Create grid spec with space for legend at bottom
    gs = fig.add_gridspec(3, 1, hspace=0.25,
                          left=0.12, right=0.95, top=0.95, bottom=0.10)
    
    # Panel labels as subtitles
    panel_labels = ['(a)', '(b)', '(c)']
    
    # Collect legend handles
    legend_handles = []
    legend_labels = []
    
    # =========================================================================
    # Panel (a): Euclidean Distance - Linear Scale
    # =========================================================================
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(panel_labels[0], fontsize=13, fontweight='bold', loc='center', pad=8)
    
    for case_name, data in all_data.items():
        n = len(data['time'])
        skip = max(1, n // 2500)
        line, = ax.plot(data['time'][::skip], data['euclidean_distance'][::skip],
                        color=COLORS[case_name], linestyle=LINESTYLES[case_name],
                        linewidth=1.4, alpha=0.9)
        if case_name not in [l for l in legend_labels]:
            legend_handles.append(line)
            legend_labels.append(data['label'])
    
    ax.set_ylabel(r'Euclidean Distance $\mathbf{d(t)}$', fontweight='bold')
    ax.set_xlim(0, t_max_global)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0)
    ax.tick_params(labelbottom=False)  # Hide x-axis labels
    
    # =========================================================================
    # Panel (b): Log₁₀ Divergence (for Lyapunov estimation)
    # =========================================================================
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title(panel_labels[1], fontsize=13, fontweight='bold', loc='center', pad=8)
    
    for case_name, data in all_data.items():
        n = len(data['time'])
        skip = max(1, n // 2500)
        log_div = data['log_divergence']
        valid_mask = ~np.isinf(log_div) & ~np.isnan(log_div)
        time_valid = data['time'][valid_mask]
        log_div_valid = log_div[valid_mask]
        
        ax.plot(time_valid[::skip], log_div_valid[::skip],
                color=COLORS[case_name], linestyle=LINESTYLES[case_name],
                linewidth=1.4, alpha=0.9)
    
    ax.set_ylabel(r'$\mathbf{\log_{10}\, d(t)}$', fontweight='bold')
    ax.set_xlim(0, t_max_global)
    ax.set_ylim(-8, 1.5)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.axhline(y=0, color='0.4', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0)
    ax.tick_params(labelbottom=False)  # Hide x-axis labels
    
    # =========================================================================
    # Panel (c): Windowed RMSE
    # =========================================================================
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title(panel_labels[2], fontsize=13, fontweight='bold', loc='center', pad=8)
    
    for case_name, data in all_data.items():
        n = len(data['time'])
        skip = max(1, n // 2500)
        ax.plot(data['time'][::skip], data['rmse_windowed'][::skip],
                color=COLORS[case_name], linestyle=LINESTYLES[case_name],
                linewidth=1.4, alpha=0.9)
    
    ax.set_xlabel(r'Time $\mathbf{t}$', fontweight='bold')
    ax.set_ylabel(r'Windowed RMSE', fontweight='bold')
    ax.set_xlim(0, t_max_global)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0)
    
    # =========================================================================
    # Bottom Legend (shared for all panels)
    # =========================================================================
    fig.legend(legend_handles, legend_labels, 
               loc='lower center', 
               bbox_to_anchor=(0.53, 0.01),
               ncol=4, 
               fontsize=10,
               frameon=True,
               fancybox=False,
               edgecolor='0.3',
               framealpha=1.0,
               handlelength=2.5,
               handletextpad=0.5,
               columnspacing=1.2,
               prop={'weight': 'bold'})
    
    # Save figures
    pdf_path = FIGS_DIR / 'divergence_timeseries.pdf'
    png_path = FIGS_DIR / 'divergence_timeseries.png'
    
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    
    print(f"Figure saved to: {pdf_path}")
    print(f"Figure saved to: {png_path}")
    
    plt.close(fig)


def main():
    """Main execution function."""
    print("=" * 60)
    print("Divergence Time Series Comparison")
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
    print("\nComputing statistics...")
    stats = compute_statistics(all_data)
    
    # Save statistics
    stats_path = STATS_DIR / 'divergence_timeseries_stats.txt'
    save_statistics(stats, stats_path)
    
    # Create figure
    print("\nGenerating figure...")
    create_figure(all_data, stats)
    
    print("\nDone!")
    print("=" * 60)


if __name__ == '__main__':
    main()
