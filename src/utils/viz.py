# Visualization utilities for convergence curves and algorithm benchmark comparisons
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Plot optimization cost convergence trajectory across iterations or generations
def plot_cost_history(history, save_path=None, title="Optimization Cost Convergence"):
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = [h.get("step", idx) for idx, h in enumerate(history)]
    best_costs = [h.get("best_cost", h.get("total_cost", 0.0)) for h in history]
    # Plot best objective value curve
    ax.plot(steps, best_costs, label="Best Cost", color="#1f77b4", linewidth=2.0)
    # Include current acceptance cost if tracked
    if any("current_cost" in h for h in history):
        curr_costs = [h.get("current_cost", np.nan) for h in history]
        ax.plot(steps, curr_costs, label="Current Cost", color="#aec7e8", alpha=0.5, linewidth=1.0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteration / Evaluation Step", fontsize=11)
    ax.set_ylabel("Cost Value", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)

# Plot comparison bar chart of total costs across multiple algorithms
def plot_comparison_bar(results_dict, save_path=None, title="Algorithm Performance Comparison"):
    fig, ax = plt.subplots(figsize=(8, 5))
    algorithms = list(results_dict.keys())
    costs = [results_dict[alg].get("total_cost", 0.0) for alg in algorithms]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"][:len(algorithms)]
    bars = ax.bar(algorithms, costs, color=colors, width=0.55, edgecolor="#333333", linewidth=1.0)
    # Annotate numeric values atop bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Objective Cost", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)
