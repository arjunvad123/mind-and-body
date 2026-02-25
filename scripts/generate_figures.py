"""
Generate publication-quality figures from experiment results.

Usage:
    python scripts/generate_figures.py

Reads results JSONs from experiment directories and produces PNGs in figures/.
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Paths
ROOT = Path(__file__).parent.parent
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

COLORS = {
    'positive': '#2ecc71',
    'negative': '#e74c3c',
    'inconclusive': '#f39c12',
    'na': '#bdc3c7',
    'primary': '#3498db',
    'secondary': '#9b59b6',
    'accent': '#1abc9c',
    'dark': '#2c3e50',
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Results Heatmap ──────────────────────────────────────────

def generate_results_heatmap():
    """Cross-experiment probe results heatmap."""

    probes = [
        'Self-Model',
        'Surprise',
        'Temporal Integration',
        'First Thought',
        'Cross-Observer Preferences',
        'Layer/Neuron Preference',
        'Time Constants',
        'Phase Portrait',
        'Lyapunov Exponents',
        'Synchronization',
        'Integrated Information',
    ]

    experiments = ['Exp 1\n(RL)', 'Exp 5\n(GPT-2)', 'Exp 6\n(Liquid NN)']

    # 1 = positive, 0 = negative, 0.5 = inconclusive, -1 = N/A
    data = np.array([
        [1,    1,    1   ],  # Self-Model
        [0.5,  0,    1   ],  # Surprise
        [0,    1,    1   ],  # Temporal
        [-1,   1,    1   ],  # First Thought
        [0.5,  1,    0   ],  # Preferences
        [-1,   0,    0   ],  # Layer/Neuron
        [-1,   -1,   1   ],  # Time Constants
        [-1,   -1,   0   ],  # Phase Portrait
        [-1,   -1,   0   ],  # Lyapunov
        [-1,   -1,   1   ],  # Synchronization
        [-1,   -1,   0   ],  # Phi
    ])

    labels = np.array([
        ['+ (RSA 0.53)',    '+ (p<1e-91)',      '+ (own=0.002)'],
        ['inconclusive',    'FAILED',           'FIXED (5.14x)'],
        ['-',               '+ (21x ratio)',    '+ (24x ratio)'],
        ['N/A',             '+ (ratio 1.01)',   '+ (ratio 1.42)'],
        ['confounded',      '+ (RSA 0.89)',     '- (RSA 0.001)'],
        ['N/A',             '- (middle)',       '- (motor)'],
        ['N/A',             'N/A',              '+ (adaptive)'],
        ['N/A',             'N/A',              '- (no sep.)'],
        ['N/A',             'N/A',              '- (stable)'],
        ['N/A',             'N/A',              '+ (0.98 coh.)'],
        ['N/A',             'N/A',              '- (negative)'],
    ])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('results', [
        (0.0, COLORS['na']),
        (0.25, COLORS['negative']),
        (0.5, COLORS['inconclusive']),
        (0.75, COLORS['inconclusive']),
        (1.0, COLORS['positive']),
    ])

    # Normalize data for colormap: -1 -> 0, 0 -> 0.25, 0.5 -> 0.5, 1 -> 1
    norm_data = (data + 1) / 2

    ax.imshow(norm_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add text labels
    for i in range(len(probes)):
        for j in range(len(experiments)):
            color = 'white' if data[i, j] in [0, 1] else 'black'
            if data[i, j] == -1:
                color = '#666666'
            ax.text(j, i, labels[i, j], ha='center', va='center',
                    fontsize=8.5, color=color, fontweight='bold' if data[i, j] == 1 else 'normal')

    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(probes)))
    ax.set_yticklabels(probes, fontsize=10)
    ax.set_title('Consciousness Probe Results Across Experiments', fontsize=16, pad=15)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS['positive'], label='Positive'),
        mpatches.Patch(color=COLORS['negative'], label='Negative'),
        mpatches.Patch(color=COLORS['inconclusive'], label='Inconclusive'),
        mpatches.Patch(color=COLORS['na'], label='N/A'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'results_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated: results_heatmap.png")


# ── Figure 2: Experiment 6 Scaling ─────────────────────────────────────

def generate_scaling_figure():
    """Observer size vs key metrics from Exp 6 scaling experiment."""

    scaling_path = ROOT / "experiment6_liquid_observer" / "results" / "scaling_results.json"
    if not scaling_path.exists():
        print("  Skipped: exp6_scaling.png (no data)")
        return

    scaling = load_json(scaling_path)

    sizes = []
    own_errors = []
    surprise_ratios = []
    tau_corrs = []
    positive_counts = []

    for size_str in sorted(scaling.keys(), key=int):
        s = scaling[size_str]
        sizes.append(int(size_str))
        own_errors.append(s['self_model']['own_error_mean'])
        surprise_ratios.append(s['surprise']['surprise_ratio'])
        tau_corrs.append(abs(s['time_constants']['tau_vs_executor_change_corr']))
        positive_counts.append(s['summary']['positive_count'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Positive indicators
    ax = axes[0, 0]
    ax.plot(sizes, positive_counts, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
    ax.set_xlabel('Observer Hidden Size')
    ax.set_ylabel('Positive Indicators (of 5)')
    ax.set_title('Consciousness Indicators vs Size')
    ax.set_ylim(0, 5.5)
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)

    # Self-model error
    ax = axes[0, 1]
    ax.plot(sizes, own_errors, 's-', color=COLORS['accent'], linewidth=2, markersize=8)
    ax.set_xlabel('Observer Hidden Size')
    ax.set_ylabel('Own Executor Error (MSE)')
    ax.set_title('Self-Model Precision vs Size')
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)

    # Surprise ratio
    ax = axes[1, 0]
    ax.plot(sizes, surprise_ratios, 'D-', color=COLORS['secondary'], linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No surprise')
    ax.set_xlabel('Observer Hidden Size')
    ax.set_ylabel('Surprise Ratio (interesting/boring)')
    ax.set_title('Surprise Detection vs Size')
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.legend()

    # Tau-executor correlation
    ax = axes[1, 1]
    ax.plot(sizes, tau_corrs, '^-', color=COLORS['dark'], linewidth=2, markersize=8)
    ax.set_xlabel('Observer Hidden Size')
    ax.set_ylabel('|Tau-Executor Correlation|')
    ax.set_title('Executor Coupling Strength vs Size')
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)

    fig.suptitle('Experiment 6: Scaling Analysis (Observer Sizes 10-200)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp6_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated: exp6_scaling.png")


# ── Figure 3: Experiment 6 Controls ────────────────────────────────────

def generate_controls_figure():
    """Control comparison bar chart for Exp 6."""

    control_path = ROOT / "experiment6_liquid_observer" / "results" / "control_results.json"
    probe_path = ROOT / "experiment6_liquid_observer" / "results" / "probe_results.json"
    if not control_path.exists():
        print("  Skipped: exp6_controls.png (no data)")
        return

    controls = load_json(control_path)
    probes = load_json(probe_path)

    names = ['Trained\nObserver', 'Untrained', 'Linear\nBaseline', 'Shuffled', 'Wrong\nExecutor']
    scores = [
        probes['summary']['positive_count'],
        controls['untrained']['summary']['positive_count'],
        controls['linear']['summary']['positive_count'],
        controls['shuffled']['summary']['positive_count'],
        controls['wrong_executor']['summary']['positive_count'],
    ]
    totals = [
        probes['summary']['total_count'],
        controls['untrained']['summary']['total_count'],
        controls['linear']['summary']['total_count'],
        controls['shuffled']['summary']['total_count'],
        controls['wrong_executor']['summary']['total_count'],
    ]

    colors_list = [COLORS['primary'], COLORS['na'], COLORS['na'], COLORS['negative'], COLORS['na']]
    # Highlight shuffled as concerning
    colors_list[3] = '#e67e22'

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, scores, color=colors_list, edgecolor='white', linewidth=2)

    for bar, score, total in zip(bars, scores, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f'{score}/{total}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel('Positive Consciousness Indicators')
    ax.set_title('Experiment 6: Trained Observer vs Controls', fontsize=16)
    ax.set_ylim(0, max(totals) + 1.5)

    # Annotate the shuffled bar
    ax.annotate('Matches trained!\n(temporal order\ndoesn\'t matter)',
                xy=(3, scores[3]), xytext=(3.7, scores[3] + 2),
                fontsize=9, ha='center', color=COLORS['negative'],
                arrowprops=dict(arrowstyle='->', color=COLORS['negative']))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp6_controls.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated: exp6_controls.png")


# ── Figure 4: Surprise Comparison ──────────────────────────────────────

def generate_surprise_figure():
    """Exp 5 vs Exp 6 surprise probe comparison."""

    exp5_path = ROOT / "experiment5_transformer_observer" / "data" / "cluster_results" / "probe_results.json"
    exp6_path = ROOT / "experiment6_liquid_observer" / "results" / "probe_results.json"

    if not exp5_path.exists() or not exp6_path.exists():
        print("  Skipped: surprise_comparison.png (no data)")
        return

    exp5 = load_json(exp5_path)
    exp6 = load_json(exp6_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Exp 5 (failed)
    ax = axes[0]
    categories = ['Garden-Path\nElevated', 'Reasoning\nElevated']
    values = [
        1.0 if exp5['surprise'].get('garden_path_elevated', False) else 0.0,
        1.0 if exp5['surprise'].get('reasoning_elevated', False) else 0.0,
    ]
    bars = ax.bar(categories, values, color=[COLORS['negative'], COLORS['negative']],
                  edgecolor='white', linewidth=2, width=0.5)
    ax.set_ylim(0, 1.5)
    ax.set_title('Exp 5: Surprise Probe (FAILED)', fontsize=14, color=COLORS['negative'])
    ax.set_ylabel('Detected (1=Yes, 0=No)')
    ax.text(0.5, 0.7, 'Neither garden-path nor\nreasoning prompts elevated\nobserver surprise',
            ha='center', va='center', transform=ax.transAxes, fontsize=10,
            style='italic', color='#666666')

    # Right: Exp 6 (fixed)
    ax = axes[1]
    categories = ['Interesting\nMoments', 'Boring\nMoments']
    values = [
        exp6['surprise']['mean_surprise_interesting'],
        exp6['surprise']['mean_surprise_boring'],
    ]
    bars = ax.bar(categories, values, color=[COLORS['positive'], COLORS['na']],
                  edgecolor='white', linewidth=2, width=0.5)
    ratio = exp6['surprise']['surprise_ratio']
    ax.set_title(f'Exp 6: Surprise Probe (FIXED, {ratio:.1f}x ratio)', fontsize=14,
                 color=COLORS['positive'])
    ax.set_ylabel('Mean Prediction Error')

    # Add ratio annotation
    max_val = max(values)
    ax.annotate(f'{ratio:.1f}x', xy=(0.5, max_val * 0.7),
                fontsize=20, fontweight='bold', ha='center', color=COLORS['positive'])

    fig.suptitle('The Surprise Story: From Failure to Fix', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'surprise_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated: surprise_comparison.png")


# ── Figure 5: Architecture Diagram ─────────────────────────────────────

def generate_architecture_figure():
    """Executor-Observer architecture diagram."""

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Executor box
    exec_box = mpatches.FancyBboxPatch((0.5, 1), 4, 4, boxstyle="round,pad=0.3",
                                        facecolor='#3498db', edgecolor='#2c3e50',
                                        linewidth=2, alpha=0.9)
    ax.add_patch(exec_box)
    ax.text(2.5, 4.3, 'EXECUTOR', ha='center', va='center', fontsize=16,
            fontweight='bold', color='white')
    ax.text(2.5, 3.5, 'Perception', ha='center', va='center', fontsize=10, color='white')
    ax.text(2.5, 3.0, 'Policy / Forward Pass', ha='center', va='center', fontsize=10, color='white')
    ax.text(2.5, 2.5, 'Action / Prediction', ha='center', va='center', fontsize=10, color='white')
    ax.text(2.5, 1.8, 'Environment Interaction', ha='center', va='center', fontsize=10, color='white')

    # Observer box
    obs_box = mpatches.FancyBboxPatch((7, 1), 4.5, 4, boxstyle="round,pad=0.3",
                                       facecolor='#2ecc71', edgecolor='#2c3e50',
                                       linewidth=2, alpha=0.9)
    ax.add_patch(obs_box)
    ax.text(9.25, 4.3, 'OBSERVER', ha='center', va='center', fontsize=16,
            fontweight='bold', color='white')
    ax.text(9.25, 3.5, 'State Decoder', ha='center', va='center', fontsize=10, color='white')
    ax.text(9.25, 3.0, 'Predictor (next state)', ha='center', va='center', fontsize=10, color='white')
    ax.text(9.25, 2.5, 'Self-Model', ha='center', va='center', fontsize=10, color='white')
    ax.text(9.25, 1.8, 'Emergent Properties?', ha='center', va='center', fontsize=10,
            color='white', style='italic')

    # Arrows (one-way)
    arrow_style = dict(arrowstyle='->', color=COLORS['dark'], linewidth=2.5)
    ax.annotate('', xy=(7, 3.8), xytext=(4.5, 3.8), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 3.0), xytext=(4.5, 3.0), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 2.2), xytext=(4.5, 2.2), arrowprops=arrow_style)

    # Arrow labels
    ax.text(5.75, 4.1, 'Hidden States', ha='center', va='center', fontsize=9,
            color=COLORS['dark'], fontweight='bold')
    ax.text(5.75, 3.3, 'Actions/Outputs', ha='center', va='center', fontsize=9,
            color=COLORS['dark'], fontweight='bold')
    ax.text(5.75, 2.5, 'Env. Feedback', ha='center', va='center', fontsize=9,
            color=COLORS['dark'], fontweight='bold')

    # NO FEEDBACK label
    no_feedback = mpatches.FancyBboxPatch((4.8, 0.2), 2.4, 0.6, boxstyle="round,pad=0.1",
                                           facecolor=COLORS['negative'], edgecolor='none', alpha=0.9)
    ax.add_patch(no_feedback)
    ax.text(6.0, 0.5, 'NO FEEDBACK', ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')

    # Title
    ax.text(6, 5.7, 'The Observer Hypothesis: One-Way Information Flow',
            ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['dark'])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated: architecture.png")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("Generating figures...")
    generate_architecture_figure()
    generate_results_heatmap()
    generate_scaling_figure()
    generate_controls_figure()
    generate_surprise_figure()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
