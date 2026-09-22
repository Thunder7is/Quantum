# Visualization utilities for convergence curves, algorithm comparisons, and failure heatmaps
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Plot total cost vs iteration or generation for an algorithm
def plot_cost_history(history: list, algo_name: str = "Algorithm", out_path: str = None, **kwargs):
    # Support both positional out_path and keyword save_path
    target_path = out_path if out_path is not None else kwargs.get("save_path")
    if target_path:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    steps = [h.get("step", h.get("generation", idx)) for idx, h in enumerate(history)]
    # Retrieve cost tracking trajectory values
    best_costs = [h.get("total_cost", h.get("best_cost", h.get("total", 0.0))) for h in history]

    ax.plot(steps, best_costs, label="Best Total Cost", color="#1f77b4", linewidth=2.0)
    # Include current iteration cost if recorded
    if any("current_cost" in h for h in history):
        curr_costs = [h.get("current_cost", np.nan) for h in history]
        ax.plot(steps, curr_costs, label="Current Cost", color="#aec7e8", alpha=0.5, linewidth=1.0)

    title_text = kwargs.get("title", f"{algo_name} - Cost Convergence History")
    ax.set_title(title_text, fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteration / Generation", fontsize=11)
    ax.set_ylabel("Total Cost", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()

    if target_path:
        plt.savefig(target_path, dpi=300)
    plt.close(fig)

# Bar chart comparing all algorithms on a specified metric
def plot_comparison_bar(results: dict, metric: str = "total", out_path: str = None, **kwargs):
    # Support both positional out_path and keyword save_path
    target_path = out_path if out_path is not None else kwargs.get("save_path")
    if target_path:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    algorithms = list(results.keys())
    values = []

    # Extract requested metric value for each algorithm
    for alg in algorithms:
        res = results[alg]
        val = res.get(metric)
        if val is None and metric in ("total", "total_cost"):
            val = res.get("total_cost", res.get("total"))
        if val is None and "breakdown" in res:
            val = res["breakdown"].get(metric)
        if val is None and metric == "deadhead_km" and "breakdown" in res:
            val = res["breakdown"].get("total_deadhead_km")
        if val is None and metric == "uncovered_demand" and "breakdown" in res:
            val = res["breakdown"].get("uncovered_demand_passengers")
        values.append(float(val) if val is not None else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974"][:len(algorithms)]
    bars = ax.bar(algorithms, values, color=colors, width=0.55, edgecolor="#333333", linewidth=1.0)

    # Annotate numeric values above bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    title_text = kwargs.get("title", f"Algorithm Comparison: {metric.replace('_', ' ').title()}")
    ax.set_title(title_text, fontsize=13, fontweight="bold")
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if target_path:
        plt.savefig(target_path, dpi=300)
    plt.close(fig)

# Heatmap of where classical algorithms encounter constraint violations across trips
def plot_failure_heatmap(failure_data: dict, out_path: str):
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Extract unique trips and constraint categories
    trips = list(failure_data.keys())
    all_constraints = set()
    for trip_val in failure_data.values():
        if isinstance(trip_val, dict):
            all_constraints.update(trip_val.keys())
        elif isinstance(trip_val, list):
            all_constraints.update(trip_val)

    constraints = sorted(list(all_constraints)) if all_constraints else ["Timing", "Capacity", "Maintenance"]

    # Construct 2D violation matrix
    matrix = np.zeros((len(trips), len(constraints)))
    for r_idx, tid in enumerate(trips):
        val = failure_data[tid]
        for c_idx, cname in enumerate(constraints):
            if isinstance(val, dict):
                matrix[r_idx, c_idx] = float(val.get(cname, 0))
            elif isinstance(val, list):
                matrix[r_idx, c_idx] = 1.0 if cname in val else 0.0

    # Render heatmap figure
    fig_height = max(5, min(14, len(trips) * 0.25))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    cax = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(constraints)))
    ax.set_xticklabels(constraints, fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(trips)))
    ax.set_yticklabels(trips, fontsize=8)

    ax.set_xlabel("Constraint Type", fontsize=11, fontweight="bold")
    ax.set_ylabel("Trip ID", fontsize=11, fontweight="bold")
    ax.set_title("Classical Heuristic Failure & Constraint Violation Heatmap", fontsize=12, fontweight="bold")

    fig.colorbar(cax, ax=ax, label="Violation Severity / Count")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300)
    plt.close(fig)
