#!/usr/bin/env python3
"""
Figure: State Variable Time Series
===================================
4x3 panel figure showing state variables (x, y, z) for all 4 test cases.
Rows: Cases (1-4), Columns: Variables (x, y, z)

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

# Publication-quality colors for Economy A and B
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
    'lines.linewidth': 1.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
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
            'init_A_x': nc.init_A_x,
            'init_A_y': nc.init_A_y,
            'init_A_z': nc.init_A_z,
            'init_B_x': nc.init_B_x,
            'init_B_y': nc.init_B_y,
            'init_B_z': nc.init_B_z,
            'init_perturbation': abs(nc.init_B_x - nc.init_A_x),
            't_end': nc.t_end,
            'n_points': nc.n_points,
            'label': label,
        }
    
    return data


def compute_statistics(all_data: dict) -> dict:
    """Compute state variable statistics for all cases."""
    stats = {}
    
    for case_name, data in all_data.items():
        time = data['time']
        
        case_stats = {
            'case_label': data['label'],
            'parameters': f"a={data['parameter_a']}, b={data['parameter_b']}, c={data['parameter_c']}",
            'initial_conditions_A': f"({data['init_A_x']}, {data['init_A_y']}, {data['init_A_z']})",
            'initial_conditions_B': f"({data['init_B_x']}, {data['init_B_y']}, {data['init_B_z']})",
            'initial_perturbation': data['init_perturbation'],
            'time_span': f"[{time[0]:.1f}, {time[-1]:.1f}]",
            'n_points': data['n_points'],
        }
        
        # Statistics for each state variable
        for var in ['x', 'y', 'z']:
            A_data = data[f'A_{var}']
            B_data = data[f'B_{var}']
            
            case_stats[f'{var}_A_min'] = np.min(A_data)
            case_stats[f'{var}_A_max'] = np.max(A_data)
            case_stats[f'{var}_A_mean'] = np.mean(A_data)
            case_stats[f'{var}_A_std'] = np.std(A_data)
            
            case_stats[f'{var}_B_min'] = np.min(B_data)
            case_stats[f'{var}_B_max'] = np.max(B_data)
            case_stats[f'{var}_B_mean'] = np.mean(B_data)
            case_stats[f'{var}_B_std'] = np.std(B_data)
            
            # Correlation between A and B
            corr = np.corrcoef(A_data, B_data)[0, 1]
            case_stats[f'{var}_correlation'] = corr
        
        # Cross-correlations
        case_stats['xy_correlation_A'] = np.corrcoef(data['A_x'], data['A_y'])[0, 1]
        case_stats['xz_correlation_A'] = np.corrcoef(data['A_x'], data['A_z'])[0, 1]
        case_stats['yz_correlation_A'] = np.corrcoef(data['A_y'], data['A_z'])[0, 1]
        
        stats[case_name] = case_stats
    
    return stats


def save_statistics(stats: dict, filepath: Path):
    """Save state variable statistics to text file."""
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATE VARIABLE TIME SERIES - STATISTICAL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for case_name, case_stats in stats.items():
            f.write(f"{case_stats['case_label']}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Parameters:           {case_stats['parameters']}\n")
            f.write(f"Initial conditions A: {case_stats['initial_conditions_A']}\n")
            f.write(f"Initial conditions B: {case_stats['initial_conditions_B']}\n")
            f.write(f"Initial perturbation: {case_stats['initial_perturbation']:.2e}\n")
            f.write(f"Time span:            {case_stats['time_span']}\n")
            f.write(f"Number of points:     {case_stats['n_points']:,}\n")
            f.write("\n")
            
            # Interest rate x
            f.write("Interest Rate x(t):\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Economy A:  min={case_stats['x_A_min']:.4f}, max={case_stats['x_A_max']:.4f}, ")
            f.write(f"mean={case_stats['x_A_mean']:.4f}, std={case_stats['x_A_std']:.4f}\n")
            f.write(f"  Economy B:  min={case_stats['x_B_min']:.4f}, max={case_stats['x_B_max']:.4f}, ")
            f.write(f"mean={case_stats['x_B_mean']:.4f}, std={case_stats['x_B_std']:.4f}\n")
            f.write(f"  Correlation A-B: {case_stats['x_correlation']:.4f}\n")
            f.write("\n")
            
            # Investment y
            f.write("Investment Demand y(t):\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Economy A:  min={case_stats['y_A_min']:.4f}, max={case_stats['y_A_max']:.4f}, ")
            f.write(f"mean={case_stats['y_A_mean']:.4f}, std={case_stats['y_A_std']:.4f}\n")
            f.write(f"  Economy B:  min={case_stats['y_B_min']:.4f}, max={case_stats['y_B_max']:.4f}, ")
            f.write(f"mean={case_stats['y_B_mean']:.4f}, std={case_stats['y_B_std']:.4f}\n")
            f.write(f"  Correlation A-B: {case_stats['y_correlation']:.4f}\n")
            f.write("\n")
            
            # Price index z
            f.write("Price Index z(t):\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Economy A:  min={case_stats['z_A_min']:.4f}, max={case_stats['z_A_max']:.4f}, ")
            f.write(f"mean={case_stats['z_A_mean']:.4f}, std={case_stats['z_A_std']:.4f}\n")
            f.write(f"  Economy B:  min={case_stats['z_B_min']:.4f}, max={case_stats['z_B_max']:.4f}, ")
            f.write(f"mean={case_stats['z_B_mean']:.4f}, std={case_stats['z_B_std']:.4f}\n")
            f.write(f"  Correlation A-B: {case_stats['z_correlation']:.4f}\n")
            f.write("\n")
            
            # Cross-correlations
            f.write("Cross-Correlations (Economy A):\n")
            f.write("-" * 40 + "\n")
            f.write(f"  corr(x, y): {case_stats['xy_correlation_A']:.4f}\n")
            f.write(f"  corr(x, z): {case_stats['xz_correlation_A']:.4f}\n")
            f.write(f"  corr(y, z): {case_stats['yz_correlation_A']:.4f}\n")
            f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Notes:\n")
        f.write("  - x: Interest rate (cost of borrowing)\n")
        f.write("  - y: Investment demand (aggregate investment)\n")
        f.write("  - z: Price index (general price level)\n")
        f.write("=" * 80 + "\n")
    
    print(f"Statistics saved to: {filepath}")


def create_figure(all_data: dict, stats: dict):
    """Create the 4x3 state variable time series figure."""
    
    # Find maximum time across all cases
    t_max_global = max(data['time'][-1] for data in all_data.values())
    
    # Create figure: 4 rows (cases) x 3 columns (variables)
    fig = plt.figure(figsize=(10, 11))
    
    # Create grid spec with space for legend at bottom
    gs = fig.add_gridspec(4, 3, hspace=0.15, wspace=0.25,
                          left=0.08, right=0.97, top=0.94, bottom=0.08)
    
    # Case order
    case_order = ['case1', 'case2', 'case3', 'case4']
    
    # Panel labels (a) through (l) for 4x3 = 12 panels
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', 
                    '(g)', '(h)', '(i)', '(j)', '(k)', '(l)']
    
    # Column labels (variables) - short names only
    col_labels = [r'$\mathbf{x(t)}$', 
                  r'$\mathbf{y(t)}$', 
                  r'$\mathbf{z(t)}$']
    var_keys = ['x', 'y', 'z']
    
    # Compute global y-limits for each variable (consistent across cases)
    y_limits = {}
    for var in var_keys:
        all_min = min(min(all_data[c][f'A_{var}'].min(), all_data[c][f'B_{var}'].min()) 
                      for c in case_order if c in all_data)
        all_max = max(max(all_data[c][f'A_{var}'].max(), all_data[c][f'B_{var}'].max()) 
                      for c in case_order if c in all_data)
        pad = (all_max - all_min) * 0.08
        y_limits[var] = (all_min - pad, all_max + pad)
    
    # Collect legend handles (only once)
    legend_handles = []
    legend_labels = []
    
    for row_idx, case_name in enumerate(case_order):
        if case_name not in all_data:
            continue
        
        data = all_data[case_name]
        time = data['time']
        n = len(time)
        skip = max(1, n // 2000)
        t_sub = time[::skip]
        
        for col_idx, var in enumerate(var_keys):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Panel label index (row * 3 + col)
            panel_idx = row_idx * 3 + col_idx
            
            # Subtitle for each panel
            ax.set_title(panel_labels[panel_idx], fontsize=12, fontweight='bold', pad=6, loc='center')
            
            A_data = data[f'A_{var}'][::skip]
            B_data = data[f'B_{var}'][::skip]
            
            # Plot both trajectories
            line_a, = ax.plot(t_sub, A_data, color=COLOR_A, linewidth=0.9, alpha=0.9)
            line_b, = ax.plot(t_sub, B_data, color=COLOR_B, linewidth=0.9, alpha=0.9)
            
            # Collect legend handles only once (from first subplot)
            if row_idx == 0 and col_idx == 0:
                legend_handles = [line_a, line_b]
                legend_labels = ['Economy A', 'Economy B (Perturbed)']
            
            # Set limits
            ax.set_xlim(0, t_max_global)
            ax.set_ylim(y_limits[var])
            
            # Ticks
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis='both', which='major', labelsize=9, width=0.8)
            
            # X-axis labels only on bottom row
            if row_idx == 3:
                ax.set_xlabel(r'Time $\mathbf{t}$', fontweight='bold', fontsize=10)
            else:
                ax.tick_params(labelbottom=False)
            
            # Y-axis labels only on left column
            if col_idx == 0:
                ax.set_ylabel(r'$\mathbf{x}$', fontweight='bold', fontsize=10)
            elif col_idx == 1:
                ax.set_ylabel(r'$\mathbf{y}$', fontweight='bold', fontsize=10)
            elif col_idx == 2:
                ax.set_ylabel(r'$\mathbf{z}$', fontweight='bold', fontsize=10)
    
    # =========================================================================
    # Bottom Legend (shared for all panels)
    # =========================================================================
    fig.legend(legend_handles, legend_labels, 
               loc='lower center', 
               bbox_to_anchor=(0.52, 0.01),
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
    pdf_path = FIGS_DIR / 'state_variables.pdf'
    png_path = FIGS_DIR / 'state_variables.png'
    
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    
    print(f"Figure saved to: {pdf_path}")
    print(f"Figure saved to: {png_path}")
    
    plt.close(fig)


def main():
    """Main execution function."""
    print("=" * 60)
    print("State Variable Time Series")
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
    stats_path = STATS_DIR / 'state_variables_stats.txt'
    save_statistics(stats, stats_path)
    
    # Create figure
    print("\nGenerating figure...")
    create_figure(all_data, stats)
    
    print("\nDone!")
    print("=" * 60)


if __name__ == '__main__':
    main()
