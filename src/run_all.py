# Master execution benchmark running all five rolling stock scheduling algorithms
import os
import sys
import time
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Support execution as direct script or package module
try:
    from .algorithms.greedy import main as run_greedy
    from .algorithms.simulated_annealing import main as run_sa
    from .algorithms.genetic_algorithm import main as run_ga
    from .algorithms.nsga2 import main as run_nsga2
except (ImportError, ValueError):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.algorithms.greedy import main as run_greedy
    from src.algorithms.simulated_annealing import main as run_sa
    from src.algorithms.genetic_algorithm import main as run_ga
    from src.algorithms.nsga2 import main as run_nsga2

# Base directories for inputs and comparison outputs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
COMP_DIR = os.path.join(OUTPUTS_DIR, "comparison")

# Color palette: red=Greedy, orange=SA, green=GA, blue=NSGA-II, purple=QA
ALGO_COLORS = {
    "Greedy":              "#d62728",
    "Simulated Annealing": "#ff7f0e",
    "Genetic Algorithm":   "#2ca02c",
    "NSGA-II":             "#1f77b4",
    "Quantum Annealing":   "#9b59b6",
}


def _load_breakdowns(outputs_dir: str) -> dict:
    """
    Load run_summary.json from each algorithm folder and extract
    the standardised cost breakdown dict. Handles differing key names
    across algorithm outputs.
    """
    def _read(path):
        with open(path) as f:
            return json.load(f)

    greedy_sum = _read(os.path.join(outputs_dir, "greedy",  "run_summary.json"))
    sa_sum     = _read(os.path.join(outputs_dir, "sa",      "run_summary.json"))
    ga_sum     = _read(os.path.join(outputs_dir, "ga",      "run_summary.json"))
    nsga2_sum  = _read(os.path.join(outputs_dir, "nsga2",   "run_summary.json"))

    # Quantum annealing is optional — skip gracefully if not run yet
    qa_bd = None
    qa_summary_path = os.path.join(outputs_dir, "quantum", "run_summary.json")
    if os.path.exists(qa_summary_path):
        qa_sum = _read(qa_summary_path)
        qa_bd  = qa_sum.get("breakdown", {})

    return {
        "Greedy":              greedy_sum["breakdown"],
        "Simulated Annealing": sa_sum.get("final_breakdown", sa_sum.get("breakdown", {})),
        "Genetic Algorithm":   ga_sum["breakdown"],
        "NSGA-II":             nsga2_sum.get("best_compromise_breakdown",
                                             nsga2_sum.get("breakdown", {})),
        "Quantum Annealing":   qa_bd,   # None if not available
    }


def _normalise_bd(bd: dict) -> dict:
    """
    Normalise breakdown keys to a consistent set regardless of which
    algorithm produced the summary (handles legacy key names).
    """
    return {
        "total":                   float(bd.get("total",
                                         bd.get("total_cost", 0.0))),
        "deadhead_km":             float(bd.get("deadhead_km",
                                         bd.get("total_deadhead_km", 0.0))),
        "units_used":              int(bd.get("units_used", 0)),
        "maintenance_violation_km":float(bd.get("maintenance_violation_km", 0.0)),
        "uncovered_demand":        int(bd.get("uncovered_demand",
                                       bd.get("uncovered_demand_passengers", 0))),
        "timing_violations":       int(bd.get("timing_violations", 0)),
    }


def _build_comparison_rows(breakdowns: dict, runtimes: dict) -> list:
    """
    Build list of dicts for master_comparison.csv / ranked table.
    Skips algorithms whose breakdown is None (QA not run yet).
    """
    rows = []
    for algo, bd in breakdowns.items():
        if bd is None:
            continue
        nb = _normalise_bd(bd)
        rows.append({
            "algorithm":                algo,
            "total_cost":               round(nb["total"], 1),
            "deadhead_km":              round(nb["deadhead_km"], 1),
            "units_used":               nb["units_used"],
            "maintenance_violation_km": round(nb["maintenance_violation_km"], 1),
            "uncovered_demand":         nb["uncovered_demand"],
            "timing_violations":        nb["timing_violations"],
            "runtime_sec":              runtimes.get(algo, 0.0),
        })
    return rows


def _plot_comparison(df: pd.DataFrame, out_path: str):
    """
    4-panel bar chart comparing all available algorithms across key metrics.
    Uses per-algorithm colour coding. Adds a 5th purple bar for QA if present.
    """
    metrics_info = [
        ("total_cost",    "Total Objective Cost (lower = better)"),
        ("deadhead_km",   "Deadhead Transit (km)"),
        ("units_used",    "Activated Fleet Units (count)"),
        ("uncovered_demand", "Uncovered Passenger Demand"),
    ]

    algos  = df["algorithm"].tolist()
    colors = [ALGO_COLORS.get(a, "#7f7f7f") for a in algos]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for (col, title), ax in zip(metrics_info, ax_list):
        vals = df[col].tolist()
        bars = ax.bar(algos, vals, color=colors, width=0.55,
                      edgecolor="#222222", linewidth=1.0)
        for bar in bars:
            h = bar.get_height()
            label = f"{h:.1f}" if isinstance(h, float) and h != int(h) else f"{int(h) if h == int(h) else h}"
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=15, labelsize=8.5)

    n_algos = len(algos)
    algo_label = f"{n_algos} Algorithms" if n_algos == 5 else "4 Classical Algorithms"
    plt.suptitle(
        f"Master Benchmark: Rolling Stock Scheduling — {algo_label}",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _print_ranked_table(df: pd.DataFrame):
    """Print ranked summary table to stdout, best cost first."""
    df_r = df.sort_values("total_cost").reset_index(drop=True)
    df_r.insert(0, "rank", range(1, len(df_r) + 1))
    n = len(df_r)
    algo_label = "5 Algorithms" if n == 5 else "4 Classical Algorithms"
    print("\n=================================================================")
    print(f"  Ranked Performance Summary — {algo_label} (by Total Cost)")
    print("=================================================================")
    print(df_r.to_string(index=False))
    print("=================================================================\n")


def main():
    os.makedirs(COMP_DIR, exist_ok=True)
    runtimes = {}

    # ── Run classical algorithms ──────────────────────────────────────────────
    n_algos = 4
    # Check if quantum results already exist to decide header
    qa_exists = os.path.exists(os.path.join(OUTPUTS_DIR, "quantum", "run_summary.json"))
    total_label = "5" if qa_exists else "4"

    print("=================================================================")
    print(f"  Master Rolling Stock Benchmark: Running All {total_label} Solvers")
    print("=================================================================\n")

    print("[1/4] Executing Pure Greedy Baseline...")
    t0 = time.time(); run_greedy()
    runtimes["Greedy"] = round(time.time() - t0, 2)

    print("\n[2/4] Executing Simulated Annealing Solver...")
    t0 = time.time(); run_sa()
    runtimes["Simulated Annealing"] = round(time.time() - t0, 2)

    print("\n[3/4] Executing Genetic Algorithm Solver...")
    t0 = time.time(); run_ga()
    runtimes["Genetic Algorithm"] = round(time.time() - t0, 2)

    print("\n[4/4] Executing NSGA-II Multi-Objective Solver...")
    t0 = time.time(); run_nsga2()
    runtimes["NSGA-II"] = round(time.time() - t0, 2)

    # ── Optionally run Quantum Annealing ─────────────────────────────────────
    # QA is run only when explicitly requested via environment variable
    # Set RUN_QA=1 to include quantum annealing in the benchmark
    # e.g.: RUN_QA=1 python src/run_all.py
    run_qa = os.environ.get("RUN_QA", "0") == "1"

    if run_qa:
        print("\n[5/5] Executing Quantum Annealing (QUBO + D-Wave Simulation)...")
        try:
            from src.algorithms.quantum_annealing import solve_quantum_annealing
            from src.utils.data_loader import load_instance, get_lookups

            inst = load_instance()
            fleet        = inst["fleet"]
            trips        = inst["trips"]
            dist_lookup  = inst["dist_lookup"]
            maint_lookup = inst["maint_lookup"]

            t0 = time.time()
            _, qa_cost, qa_bd, _, qa_time = solve_quantum_annealing(
                fleet, trips, dist_lookup, maint_lookup
            )
            runtimes["Quantum Annealing"] = round(time.time() - t0, 2)
            print(f"\n  QA best cost: {qa_cost:.1f} | "
                  f"Sampling time: {qa_time:.2f}s | "
                  f"Total runtime: {runtimes['Quantum Annealing']}s")
        except Exception as exc:
            print(f"  Quantum Annealing skipped — {exc}")
    elif qa_exists:
        # QA was run previously; load its runtime from saved summary
        with open(os.path.join(OUTPUTS_DIR, "quantum", "run_summary.json")) as f:
            _qa = json.load(f)
        runtimes["Quantum Annealing"] = _qa.get("sampling_time_sec", 0.0)

    # ── Load all breakdowns and build comparison ──────────────────────────────
    breakdowns = _load_breakdowns(OUTPUTS_DIR)
    rows       = _build_comparison_rows(breakdowns, runtimes)
    df         = pd.DataFrame(rows)

    # Save master_comparison.csv
    csv_path = os.path.join(COMP_DIR, "master_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Save master_comparison.png
    img_path = os.path.join(COMP_DIR, "master_comparison.png")
    _plot_comparison(df, img_path)
    print(f"Saved figure: {img_path}")

    # Print ranked table
    _print_ranked_table(df)


if __name__ == "__main__":
    main()