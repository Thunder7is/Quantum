# Benchmark execution script running and evaluating all four rolling stock scheduling algorithms
import os
import time
import json
import pandas as pd
from utils.data_loader import load_instance
from utils.greedy_init import chains_to_schedule_df
from utils.viz import plot_comparison_bar, plot_cost_history
from algorithms.greedy import solve_greedy
from algorithms.simulated_annealing import solve_simulated_annealing
from algorithms.genetic_algorithm import solve_genetic_algorithm
from algorithms.nsga2 import solve_nsga2

# Root output path for benchmark comparisons
COMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "comparison"))
BASE_OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))

# Run all algorithms, benchmark performance, and persist comparative statistics
def run_benchmark():
    os.makedirs(COMP_DIR, exist_ok=True)
    inst = load_instance()
    fleet, trips, dist_lookup, maint_lookup = inst["fleet"], inst["trips"], inst["dist_lookup"], inst["maint_lookup"]
    stations = inst["stations"]

    print("=================================================================")
    print("  Rolling Stock Scheduling Benchmark (4 Algorithms)")
    print(f"  Fleet: {len(fleet)} units | Trips: {len(trips)} | Stations: {len(stations)}")
    print("=================================================================\n")

    results = {}
    comparison_rows = []

    # 1. Pure Greedy Heuristic
    print("[1/4] Running Pure Greedy...")
    t0 = time.time()
    g_chains, g_cost, g_breakdown = solve_greedy(fleet, trips, dist_lookup, maint_lookup)
    g_time = time.time() - t0
    results["Greedy"] = {"total_cost": g_cost, "breakdown": g_breakdown, "runtime_sec": round(g_time, 2)}
    comparison_rows.append({
        "Algorithm": "Greedy",
        "Total Cost": g_cost,
        "Deadhead (km)": g_breakdown["total_deadhead_km"],
        "Units Used": g_breakdown["units_used"],
        "Maint Violations (km)": g_breakdown["maintenance_violation_km"],
        "Uncovered Demand": g_breakdown["uncovered_demand_passengers"],
        "Timing Violations": g_breakdown["timing_violations"],
        "Runtime (s)": round(g_time, 2),
    })
    # Persist greedy schedule
    g_dir = os.path.join(BASE_OUT, "greedy")
    os.makedirs(g_dir, exist_ok=True)
    chains_to_schedule_df(g_chains, trips).to_csv(os.path.join(g_dir, "final_schedule.csv"), index=False)
    with open(os.path.join(g_dir, "run_summary.json"), "w") as f:
        json.dump(results["Greedy"], f, indent=2)

    # 2. Simulated Annealing
    print("[2/4] Running Simulated Annealing...")
    t0 = time.time()
    sa_chains, sa_cost, sa_breakdown, sa_history = solve_simulated_annealing(
        fleet, trips, dist_lookup, maint_lookup, init_chains=g_chains
    )
    sa_time = time.time() - t0
    results["Simulated Annealing"] = {"total_cost": sa_cost, "breakdown": sa_breakdown, "runtime_sec": round(sa_time, 2)}
    comparison_rows.append({
        "Algorithm": "Simulated Annealing",
        "Total Cost": sa_cost,
        "Deadhead (km)": sa_breakdown["total_deadhead_km"],
        "Units Used": sa_breakdown["units_used"],
        "Maint Violations (km)": sa_breakdown["maintenance_violation_km"],
        "Uncovered Demand": sa_breakdown["uncovered_demand_passengers"],
        "Timing Violations": sa_breakdown["timing_violations"],
        "Runtime (s)": round(sa_time, 2),
    })
    # Persist SA results
    sa_dir = os.path.join(BASE_OUT, "sa")
    os.makedirs(sa_dir, exist_ok=True)
    chains_to_schedule_df(sa_chains, trips).to_csv(os.path.join(sa_dir, "final_schedule.csv"), index=False)
    pd.DataFrame(sa_history).to_csv(os.path.join(sa_dir, "sa_cost_history.csv"), index=False)
    plot_cost_history(sa_history, save_path=os.path.join(sa_dir, "sa_convergence.png"), title="Simulated Annealing Convergence")
    with open(os.path.join(sa_dir, "run_summary.json"), "w") as f:
        json.dump(results["Simulated Annealing"], f, indent=2)

    # 3. Genetic Algorithm
    print("[3/4] Running Genetic Algorithm...")
    t0 = time.time()
    ga_chains, ga_cost, ga_breakdown, ga_history = solve_genetic_algorithm(
        fleet, trips, dist_lookup, maint_lookup
    )
    ga_time = time.time() - t0
    results["Genetic Algorithm"] = {"total_cost": ga_cost, "breakdown": ga_breakdown, "runtime_sec": round(ga_time, 2)}
    comparison_rows.append({
        "Algorithm": "Genetic Algorithm",
        "Total Cost": ga_cost,
        "Deadhead (km)": ga_breakdown["total_deadhead_km"],
        "Units Used": ga_breakdown["units_used"],
        "Maint Violations (km)": ga_breakdown["maintenance_violation_km"],
        "Uncovered Demand": ga_breakdown["uncovered_demand_passengers"],
        "Timing Violations": ga_breakdown["timing_violations"],
        "Runtime (s)": round(ga_time, 2),
    })
    # Persist GA results
    ga_dir = os.path.join(BASE_OUT, "ga")
    os.makedirs(ga_dir, exist_ok=True)
    chains_to_schedule_df(ga_chains, trips).to_csv(os.path.join(ga_dir, "final_schedule.csv"), index=False)
    pd.DataFrame(ga_history).to_csv(os.path.join(ga_dir, "ga_cost_history.csv"), index=False)
    plot_cost_history(ga_history, save_path=os.path.join(ga_dir, "ga_convergence.png"), title="Genetic Algorithm Convergence")
    with open(os.path.join(ga_dir, "run_summary.json"), "w") as f:
        json.dump(results["Genetic Algorithm"], f, indent=2)

    # 4. NSGA-II
    print("[4/4] Running NSGA-II...")
    t0 = time.time()
    ns_chains, ns_cost, ns_breakdown, ns_pareto, ns_history = solve_nsga2(
        fleet, trips, dist_lookup, maint_lookup
    )
    ns_time = time.time() - t0
    results["NSGA-II"] = {"total_cost": ns_cost, "breakdown": ns_breakdown, "runtime_sec": round(ns_time, 2)}
    comparison_rows.append({
        "Algorithm": "NSGA-II",
        "Total Cost": ns_cost,
        "Deadhead (km)": ns_breakdown["total_deadhead_km"],
        "Units Used": ns_breakdown["units_used"],
        "Maint Violations (km)": ns_breakdown["maintenance_violation_km"],
        "Uncovered Demand": ns_breakdown["uncovered_demand_passengers"],
        "Timing Violations": ns_breakdown["timing_violations"],
        "Runtime (s)": round(ns_time, 2),
    })
    # Persist NSGA-II results
    ns_dir = os.path.join(BASE_OUT, "nsga2")
    os.makedirs(ns_dir, exist_ok=True)
    chains_to_schedule_df(ns_chains, trips).to_csv(os.path.join(ns_dir, "final_schedule.csv"), index=False)
    pd.DataFrame(ns_pareto).to_csv(os.path.join(ns_dir, "pareto_front.csv"), index=False)
    pd.DataFrame(ns_history).to_csv(os.path.join(ns_dir, "nsga2_cost_history.csv"), index=False)
    plot_cost_history(ns_history, save_path=os.path.join(ns_dir, "nsga2_convergence.png"), title="NSGA-II Convergence")
    with open(os.path.join(ns_dir, "run_summary.json"), "w") as f:
        json.dump(results["NSGA-II"], f, indent=2)

    # Create comparison summary table
    comp_df = pd.DataFrame(comparison_rows)
    csv_path = os.path.join(COMP_DIR, "algorithm_comparison.csv")
    comp_df.to_csv(csv_path, index=False)

    json_path = os.path.join(COMP_DIR, "algorithm_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    bar_path = os.path.join(COMP_DIR, "comparison_bar.png")
    plot_comparison_bar(results, save_path=bar_path, title="Rolling Stock Solvers: Total Cost Comparison")

    print("\n=================================================================")
    print("  Benchmark Summary Results:")
    print("=================================================================")
    print(comp_df.to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved Chart: {bar_path}\n")

# Main execution entrypoint
if __name__ == "__main__":
    run_benchmark()
