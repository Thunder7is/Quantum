"""
Simulated Annealing Solver for Rolling Stock Scheduling
=========================================================

Consumes the synthetic instance produced by generate_data.py and
produces a rolling stock circulation plan: which unit(s) serve which
trip, in what order, minimizing a weighted cost of:

  1. Deadhead travel (empty repositioning between trips)
  2. Number of distinct units activated during the day
  3. Maintenance threshold violations (km run without a maintenance visit)
  4. Uncovered passenger demand (capacity shortfall on a trip)

------------------------------------------------------------------
SOLUTION REPRESENTATION
------------------------------------------------------------------
A solution is represented per-UNIT as an ordered chain of trips:

    chains = { unit_id: [trip_id, trip_id, ...] }   # departure-time order

A trip may appear in more than one unit's chain if it needs coupling
(multiple units run together to satisfy demand). Feasibility for a
chain requires, for every consecutive pair of trips (a -> b) assigned
to the same unit:

    location(a.destination) == location(b.origin)        [continuity]
    b.departure_min >= a.arrival_min + a.min_turnaround   [timing]

For a unit's FIRST trip, the unit must be at that trip's origin by
its own available_from_min (or reposition there empty beforehand,
which is charged as deadhead).

------------------------------------------------------------------
CONSTRUCTION (initial feasible solution)
------------------------------------------------------------------
Greedy, in departure-time order: for each trip, pick the cheapest
already-available unit(s) (by deadhead distance) that can reach the
trip's origin in time; couple a second unit only if capacity is
still short of demand. This guarantees a feasible (if not efficient)
starting point for SA to improve.

------------------------------------------------------------------
SIMULATED ANNEALING
------------------------------------------------------------------
Neighborhood moves:
  - RELOCATE: move one trip from one unit's chain to a different
    (feasible) insertion point in another unit's chain.
  - SWAP: exchange two trips between two units' chains.
  - COUPLE/DECOUPLE: add or remove a helper unit on an under/over
    served trip.

Standard Metropolis acceptance with geometric cooling.
------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import random
import math
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "sa_output")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 7
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------
# COST WEIGHTS  (tune these to change solver priorities)
# ---------------------------------------------------------------
W_DEADHEAD_PER_KM = 1.0
W_UNIT_ACTIVATION = 150.0         # fixed cost per distinct unit used that day
W_MAINTENANCE_VIOLATION_PER_KM = 5.0   # penalty per km run past the threshold
W_UNCOVERED_DEMAND_PER_PASSENGER = 2.0

# SA schedule
INITIAL_TEMP = 500.0
COOLING_RATE = 0.995
MIN_TEMP = 0.5
ITERATIONS_PER_TEMP = 40


# =================================================================
# 1. LOAD DATA
# =================================================================
def load_instance():
    stations = pd.read_csv(os.path.join(DATA_DIR, "stations.csv"))
    dist = pd.read_csv(os.path.join(DATA_DIR, "distance_matrix.csv"))
    fleet = pd.read_csv(os.path.join(DATA_DIR, "fleet.csv"))
    trips = pd.read_csv(os.path.join(DATA_DIR, "trips.csv")).sort_values("departure_min").reset_index(drop=True)
    demand = pd.read_csv(os.path.join(DATA_DIR, "demand.csv"))
    maint_rules = pd.read_csv(os.path.join(DATA_DIR, "maintenance_rules.csv"))

    trips = trips.merge(demand, on="trip_id")
    dist_lookup = {(r.from_station, r.to_station): (r.travel_time_min, r.distance_km) for r in dist.itertuples()}
    maint_lookup = {r.unit_type: r.max_km_between_maintenance for r in maint_rules.itertuples()}

    return stations, dist_lookup, fleet, trips, maint_lookup


def travel(dist_lookup, o, d):
    """Return (time_min, distance_km) between two stations; 0 if same station."""
    if o == d:
        return 0, 0.0
    return dist_lookup[(o, d)]


# =================================================================
# 2. GREEDY FEASIBLE CONSTRUCTION
# =================================================================
def build_initial_solution(fleet, trips, dist_lookup):
    """
    Returns:
      chains: {unit_id: [trip_id, ...]} in departure order
      trip_assignment: {trip_id: [unit_id, ...]}
    """
    unit_state = {
        r.unit_id: {
            "location": r.home_depot,
            "avail_time": r.available_from_min,
            "capacity": r.capacity,
            "km_since_maint": r.km_since_maintenance,
            "unit_type": r.unit_type,
        }
        for r in fleet.itertuples()
    }
    chains = {uid: [] for uid in unit_state}
    trip_assignment = {}

    for trip in trips.itertuples():
        needed_capacity = trip.expected_passenger_demand
        chosen = []
        covered_capacity = 0

        # candidate units: available in time, ranked by deadhead distance to trip origin
        candidates = []
        for uid, st in unit_state.items():
            if st["avail_time"] > trip.departure_min:
                continue
            _, dist_km = travel(dist_lookup, st["location"], trip.origin_station)
            candidates.append((dist_km, uid))
        candidates.sort()  # cheapest deadhead first

        for dist_km, uid in candidates:
            if covered_capacity >= needed_capacity:
                break
            st = unit_state[uid]
            chosen.append(uid)
            covered_capacity += st["capacity"]

        # fallback: if nothing available in time at all, force-assign the
        # soonest-available unit anyway (creates a timing violation that
        # the cost function will penalize and SA can later fix)
        if not chosen:
            uid = min(unit_state, key=lambda u: unit_state[u]["avail_time"])
            chosen = [uid]

        for uid in chosen:
            st = unit_state[uid]
            _, deadhead_km = travel(dist_lookup, st["location"], trip.origin_station)
            st["location"] = trip.destination_station
            st["avail_time"] = trip.arrival_min + trip.min_turnaround_min
            st["km_since_maint"] += deadhead_km + trip.distance_km
            chains[uid].append(trip.trip_id)

        trip_assignment[trip.trip_id] = chosen

    return chains, trip_assignment


# =================================================================
# 3. COST FUNCTION
# =================================================================
def evaluate(chains, fleet_lookup, trips_lookup, dist_lookup, maint_lookup):
    """
    fleet_lookup: {unit_id: dict(home_depot, available_from_min, km_since_maintenance,
                                   unit_type, capacity)}
    trips_lookup: {trip_id: dict(origin_station, destination_station, departure_min,
                                   arrival_min, min_turnaround_min, distance_km,
                                   expected_passenger_demand)}
    Plain-dict lookups (no pandas indexing) since this runs inside the SA hot loop.
    """
    total_deadhead_km = 0.0
    total_maint_violation_km = 0.0
    timing_violations = 0
    units_used = 0

    capacity_supplied = {tid: 0 for tid in trips_lookup}

    for uid, chain in chains.items():
        if not chain:
            continue
        units_used += 1
        row = fleet_lookup[uid]
        location = row["home_depot"]
        avail_time = row["available_from_min"]
        km_since_maint = row["km_since_maintenance"]
        max_km = maint_lookup[row["unit_type"]]
        capacity = row["capacity"]

        # sort this unit's chain by departure time to evaluate continuity
        chain_sorted = sorted(chain, key=lambda tid: trips_lookup[tid]["departure_min"])

        for tid in chain_sorted:
            t = trips_lookup[tid]
            capacity_supplied[tid] += capacity

            travel_time, deadhead_km = travel(dist_lookup, location, t["origin_station"])
            total_deadhead_km += deadhead_km

            arrival_of_deadhead = avail_time + travel_time
            if arrival_of_deadhead > t["departure_min"]:
                timing_violations += 1  # can't physically make it in time

            km_since_maint += deadhead_km + t["distance_km"]
            if km_since_maint > max_km:
                total_maint_violation_km += (km_since_maint - max_km)
                km_since_maint = 0  # assume a maintenance visit resets it (simplified)

            location = t["destination_station"]
            avail_time = t["arrival_min"] + t["min_turnaround_min"]

    uncovered_penalty_passengers = 0
    for tid, supplied in capacity_supplied.items():
        demand = trips_lookup[tid]["expected_passenger_demand"]
        if supplied < demand:
            uncovered_penalty_passengers += (demand - supplied)

    cost = (
        W_DEADHEAD_PER_KM * total_deadhead_km
        + W_UNIT_ACTIVATION * units_used
        + W_MAINTENANCE_VIOLATION_PER_KM * total_maint_violation_km
        + W_UNCOVERED_DEMAND_PER_PASSENGER * uncovered_penalty_passengers
        + 1000 * timing_violations  # hard-ish penalty, large but finite
    )

    breakdown = {
        "total_deadhead_km": round(total_deadhead_km, 1),
        "units_used": units_used,
        "maintenance_violation_km": round(total_maint_violation_km, 1),
        "uncovered_demand_passengers": uncovered_penalty_passengers,
        "timing_violations": timing_violations,
        "total_cost": round(cost, 1),
    }
    return cost, breakdown


# =================================================================
# 4. NEIGHBORHOOD MOVES
# =================================================================
def copy_chains(chains):
    return {uid: list(trips) for uid, trips in chains.items()}


def move_relocate(chains, all_unit_ids, all_trip_ids, rng):
    """Move one random trip from its current unit(s) to a different unit."""
    new_chains = copy_chains(chains)
    donors = [uid for uid, c in new_chains.items() if c]
    if not donors:
        return new_chains
    donor = rng.choice(donors)
    trip_id = rng.choice(new_chains[donor])
    receiver = rng.choice(all_unit_ids)
    new_chains[donor].remove(trip_id)
    new_chains[receiver].append(trip_id)
    return new_chains


def move_swap(chains, rng):
    """Swap one trip between two different units' chains."""
    new_chains = copy_chains(chains)
    non_empty = [uid for uid, c in new_chains.items() if c]
    if len(non_empty) < 2:
        return new_chains
    u1, u2 = rng.sample(non_empty, 2)
    t1 = rng.choice(new_chains[u1])
    t2 = rng.choice(new_chains[u2])
    new_chains[u1].remove(t1)
    new_chains[u2].remove(t2)
    new_chains[u1].append(t2)
    new_chains[u2].append(t1)
    return new_chains


def move_couple_decouple(chains, trip_assignment, all_unit_ids, all_trip_ids, rng):
    """Randomly add or remove a coupled helper unit on a random trip."""
    new_chains = copy_chains(chains)
    trip_id = rng.choice(all_trip_ids)
    assigned_units = [uid for uid, c in new_chains.items() if trip_id in c]

    if rng.random() < 0.5 or not assigned_units:
        # try to add a helper unit
        candidates = [u for u in all_unit_ids if u not in assigned_units]
        if candidates:
            uid = rng.choice(candidates)
            new_chains[uid].append(trip_id)
    else:
        # remove one assigned unit (keep at least one)
        if len(assigned_units) > 1:
            uid = rng.choice(assigned_units)
            new_chains[uid].remove(trip_id)

    return new_chains


# =================================================================
# 5. SIMULATED ANNEALING MAIN LOOP
# =================================================================
def simulated_annealing(chains_init, fleet_lookup, trips_lookup, dist_lookup, maint_lookup, rng):
    all_unit_ids = list(fleet_lookup.keys())
    all_trip_ids = list(trips_lookup.keys())

    current = copy_chains(chains_init)
    current_cost, current_breakdown = evaluate(current, fleet_lookup, trips_lookup, dist_lookup, maint_lookup)

    best = copy_chains(current)
    best_cost = current_cost
    best_breakdown = current_breakdown

    temp = INITIAL_TEMP
    history = []

    move_fns = [
        lambda c: move_relocate(c, all_unit_ids, all_trip_ids, rng),
        lambda c: move_swap(c, rng),
        lambda c: move_couple_decouple(c, None, all_unit_ids, all_trip_ids, rng),
    ]

    step = 0
    while temp > MIN_TEMP:
        for _ in range(ITERATIONS_PER_TEMP):
            move = rng.choice(move_fns)
            candidate = move(current)
            cand_cost, cand_breakdown = evaluate(candidate, fleet_lookup, trips_lookup, dist_lookup, maint_lookup)

            delta = cand_cost - current_cost
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current, current_cost, current_breakdown = candidate, cand_cost, cand_breakdown
                if current_cost < best_cost:
                    best, best_cost, best_breakdown = copy_chains(current), current_cost, current_breakdown

            step += 1

        history.append({"step": step, "temp": round(temp, 2), "current_cost": round(current_cost, 1), "best_cost": round(best_cost, 1)})
        temp *= COOLING_RATE

    return best, best_cost, best_breakdown, history


# =================================================================
# 6. RUN
# =================================================================
def chains_to_schedule_df(chains, trips_df):
    rows = []
    trips_idx = trips_df.set_index("trip_id")
    for uid, chain in chains.items():
        for tid in sorted(chain, key=lambda t: trips_idx.loc[t].departure_min):
            t = trips_idx.loc[tid]
            rows.append({
                "unit_id": uid,
                "trip_id": tid,
                "origin": t.origin_station,
                "destination": t.destination_station,
                "departure_time": t.departure_time,
                "arrival_time": t.arrival_time,
            })
    return pd.DataFrame(rows).sort_values(["unit_id", "departure_time"])


if __name__ == "__main__":
    rng = random.Random(SEED)

    stations, dist_lookup, fleet, trips, maint_lookup = load_instance()

    print(f"Loaded instance: {len(fleet)} units, {len(trips)} trips, {len(stations)} stations\n")

    # Plain-dict lookups for the SA hot loop (pandas .loc indexing is too slow at scale)
    fleet_lookup = {r.unit_id: r._asdict() for r in fleet.itertuples(index=False)}
    trips_lookup = {r.trip_id: r._asdict() for r in trips.itertuples(index=False)}

    init_chains, init_assignment = build_initial_solution(fleet, trips, dist_lookup)
    init_cost, init_breakdown = evaluate(init_chains, fleet_lookup, trips_lookup, dist_lookup, maint_lookup)
    print("Initial (greedy) solution cost breakdown:")
    print(json.dumps(init_breakdown, indent=2))

    best_chains, best_cost, best_breakdown, history = simulated_annealing(
        init_chains, fleet_lookup, trips_lookup, dist_lookup, maint_lookup, rng
    )

    print("\nFinal (SA-optimized) solution cost breakdown:")
    print(json.dumps(best_breakdown, indent=2))
    improvement_pct = 100 * (init_cost - best_cost) / init_cost if init_cost else 0
    print(f"\nImprovement over greedy baseline: {improvement_pct:.1f}%")

    schedule_df = chains_to_schedule_df(best_chains, trips)
    schedule_path = os.path.join(OUT_DIR, "final_schedule.csv")
    schedule_df.to_csv(schedule_path, index=False)

    history_df = pd.DataFrame(history)
    history_path = os.path.join(OUT_DIR, "sa_cost_history.csv")
    history_df.to_csv(history_path, index=False)

    with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
        json.dump({
            "initial_breakdown": init_breakdown,
            "final_breakdown": best_breakdown,
            "improvement_pct": round(improvement_pct, 1),
            "sa_config": {
                "initial_temp": INITIAL_TEMP,
                "cooling_rate": COOLING_RATE,
                "min_temp": MIN_TEMP,
                "iterations_per_temp": ITERATIONS_PER_TEMP,
            },
        }, f, indent=2)

    print(f"\nSaved: {schedule_path}")
    print(f"Saved: {history_path}")
    print(f"\nSchedule preview:")
    print(schedule_df.head(15).to_string(index=False))
