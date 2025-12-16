"""
Generate Nature-style publication figures for CASP paper
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

# Nature-style settings
plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Nature-inspired color palette
COLORS = {
    'euclidean': '#8C8C8C',      # Gray
    'volnorm': '#B0B0B0',        # Light gray
    'casp_basic': '#3B7EA1',     # Steel blue
    'casp_retsel': '#6BAED6',    # Light blue
    'ra_casp': '#D55E00',        # Vermillion/orange
    'minvar': '#CC79A7',         # Pink
    'sharpe': '#4D4D4D',         # Dark gray
}

# Paths (resolve relative to this file, not the current working directory)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(_BASE_DIR, '..', 'paper', 'figures'))
RESULTS_DIR = os.path.abspath(os.path.join(_BASE_DIR, '..', 'results'))


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing results file: {path}.\n"
            "Run `python src/analysis.py` first to generate results under `results/`."
        )
    return pd.read_csv(path)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float('nan')


def create_fig1_mechanism():
    """Conceptual diagram showing Euclidean vs Ω-metric (tracking-error) projection"""
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.8))
    
    for ax, title, proj_type in zip(axes, ['Euclidean Projection', 'Ω-metric Projection (CASP)'],
                                     ['euclidean', 'mahalanobis']):
        # Draw feasible region (simplex-like triangle)
        triangle = plt.Polygon([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]], 
                               fill=True, facecolor='#E8E8E8', edgecolor='#666666', linewidth=1)
        ax.add_patch(triangle)
        
        # Infeasible point
        z = [0.7, 0.6]
        ax.plot(z[0], z[1], 'o', color='#333333', markersize=8, zorder=5)
        ax.annotate('z (infeasible)', xy=(z[0], z[1]), xytext=(z[0]+0.08, z[1]+0.05),
                   fontsize=7, color='#333333')
        
        if proj_type == 'euclidean':
            w_euc = [0.55, 0.45]
            ax.plot(w_euc[0], w_euc[1], 's', color=COLORS['euclidean'], markersize=8, zorder=5)
            ax.annotate('$w^*_{Euc}$', xy=(w_euc[0], w_euc[1]), xytext=(w_euc[0]-0.12, w_euc[1]-0.08),
                       fontsize=8, color=COLORS['euclidean'])
            ax.annotate('', xy=(w_euc[0], w_euc[1]), xytext=(z[0], z[1]),
                       arrowprops=dict(arrowstyle='->', color=COLORS['euclidean'], lw=1.5))
            
            for r in [0.1, 0.2, 0.3]:
                circle = plt.Circle(z, r, fill=False, color=COLORS['euclidean'], 
                                   linestyle='--', linewidth=0.5, alpha=0.5)
                ax.add_patch(circle)
        else:
            w_casp = [0.45, 0.35]
            ax.plot(w_casp[0], w_casp[1], 's', color=COLORS['casp_basic'], markersize=8, zorder=5)
            ax.annotate('$w^*_{CASP}$', xy=(w_casp[0], w_casp[1]), xytext=(w_casp[0]-0.15, w_casp[1]-0.08),
                       fontsize=8, color=COLORS['casp_basic'])
            ax.annotate('', xy=(w_casp[0], w_casp[1]), xytext=(z[0], z[1]),
                       arrowprops=dict(arrowstyle='->', color=COLORS['casp_basic'], lw=1.5))
            
            for r in [0.12, 0.24, 0.36]:
                ellipse = Ellipse(z, r*1.5, r*0.8, angle=30, fill=False, 
                                 color=COLORS['casp_basic'], linestyle='--', linewidth=0.5, alpha=0.5)
                ax.add_patch(ellipse)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('Weight $w_1$', fontsize=8)
        ax.set_ylabel('Weight $w_2$', fontsize=8)
        ax.set_title(title, fontsize=9, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    
    axes[0].text(-0.1, 1.05, 'a', transform=axes[0].transAxes, fontsize=12, fontweight='bold')
    axes[1].text(-0.1, 1.05, 'b', transform=axes[1].transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_mechanism.pdf'), bbox_inches='tight', format='pdf')
    plt.close()
    print("  Created fig1_mechanism.pdf")


def create_fig2_ablation():
    """Panel A: Variance reduction, Panel B: Sharpe comparison"""
    
    fig = plt.figure(figsize=(7, 3))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)
    
    ab = _safe_read_csv(os.path.join(RESULTS_DIR, 'ablation_summary.csv'))
    # Prefer a paper-friendly ordering if all methods exist
    preferred = ['Euclidean', 'VolNorm+Euc', 'MinVar+Euc', 'CASP-Basic', 'CASP-RetSel', 'RA-CASP', 'Sharpe+Euc']
    if set(preferred).issubset(set(ab['method'])):
        ab = ab.set_index('method').loc[preferred].reset_index()
    
    methods = ab['method'].tolist()
    variance = ab['mean_variance'].to_numpy()
    sharpe = ab['mean_sharpe'].to_numpy()
    var_reduction = ab['var_reduction_pct_vs_Euclidean'].to_numpy()
    
    def method_color(m: str) -> str:
        if m == 'Euclidean':
            return COLORS['euclidean']
        if m == 'VolNorm+Euc':
            return COLORS['volnorm']
        if m == 'MinVar+Euc':
            return COLORS['minvar']
        if m == 'Sharpe+Euc':
            return COLORS['sharpe']
        if m == 'CASP-Basic':
            return COLORS['casp_basic']
        if m == 'CASP-RetSel':
            return COLORS['casp_retsel']
        if m == 'RA-CASP':
            return COLORS['ra_casp']
        return '#777777'
    
    colors = [method_color(m) for m in methods]
    
    # Panel A: Variance
    ax1 = fig.add_subplot(gs[0])
    bars1 = ax1.bar(range(len(methods)), variance, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=35, ha='right')
    ax1.set_ylabel('Portfolio Variance')
    ax1.set_ylim(0, float(max(variance) * 1.25))
    ax1.axhline(y=variance[0], color=COLORS['euclidean'], linestyle='--', linewidth=0.8, alpha=0.7)
    
    for i, (bar, vr) in enumerate(zip(bars1, var_reduction)):
        if i == 0:
            continue
        if np.isfinite(vr):
            sign = '−' if vr >= 0 else '+'
            ax1.annotate(f'{sign}{abs(vr):.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7,
                        color=colors[i])
    
    ax1.text(-0.15, 1.05, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax1.set_title('Variance Reduction', fontsize=9, pad=10)
    
    # Panel B: Sharpe Ratio
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(range(len(methods)), sharpe, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=35, ha='right')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_ylim(0, 0.7)
    ax2.axhline(y=sharpe[0], color=COLORS['euclidean'], linestyle='--', linewidth=0.8, alpha=0.7)
    
    ax2.text(-0.15, 1.05, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold')
    ax2.set_title('Risk-Adjusted Return', fontsize=9, pad=10)
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_ablation.pdf'), bbox_inches='tight', format='pdf')
    plt.close()
    print("  Created fig2_ablation.pdf")


def create_fig3_oos():
    """Out-of-sample validation: scatter plot with trend lines"""
    
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    raw_path = os.path.join(RESULTS_DIR, 'oos_raw.csv')
    raw_df = _safe_read_csv(raw_path)

    methods = ['Euclidean', 'CASP-Basic', 'RA-CASP']
    colors = [COLORS['euclidean'], COLORS['casp_basic'], COLORS['ra_casp']]

    for ax, method, color in zip(axes, methods, colors):
        subset = raw_df[raw_df['method'] == method]
        in_sample = subset['in_sample_sharpe'].to_numpy()
        out_sample = subset['out_sample_sharpe'].to_numpy()
        corr = _spearman_corr(in_sample, out_sample)

        ax.scatter(in_sample, out_sample, c=color, alpha=0.5, s=20, edgecolors='none')

        z = np.polyfit(in_sample, out_sample, 1)
        p = np.poly1d(z)
        x_line = np.linspace(float(np.min(in_sample)), float(np.max(in_sample)), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=1.5, linestyle='-')

        ax.set_xlabel('In-Sample Sharpe')
        ax.set_ylabel('Out-of-Sample Sharpe')
        ax.set_title(f'{method}\n(ρ = {corr:.2f})', fontsize=9)
    
    axes[0].text(-0.2, 1.1, 'a', transform=axes[0].transAxes, fontsize=12, fontweight='bold')
    axes[1].text(-0.2, 1.1, 'b', transform=axes[1].transAxes, fontsize=12, fontweight='bold')
    axes[2].text(-0.2, 1.1, 'c', transform=axes[2].transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_oos.pdf'), bbox_inches='tight', format='pdf')
    plt.close()
    print("  Created fig3_oos.pdf")


if __name__ == '__main__':
    print("Generating figures for paper...")
    print("=" * 40)
    
    create_fig1_mechanism()
    create_fig2_ablation()
    create_fig3_oos()
    
    print("=" * 40)
    print("Done! Figures saved to ../paper/figures/")
