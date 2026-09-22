"""
Synthetic Data Generator for Rolling Stock Scheduling
======================================================

Generates a self-contained problem instance for the Rolling Stock
Scheduling / Circulation Problem, suitable as input to:
  - Simulated Annealing / other metaheuristics (classical baseline)
  - Later: QUBO formulation for Quantum Annealing comparison

Output files (all in ./data/):
  stations.csv        - network nodes (stations), which are depots
  distance_matrix.csv - travel time (minutes) and distance (km) between stations
  fleet.csv            - rolling stock units (trainsets) and their state
  trips.csv            - timetable: trips that must be covered
  demand.csv           - passenger demand per trip (drives composition/coupling)
  maintenance_rules.csv- maintenance thresholds per unit type
  instance_meta.json   - summary/config used to generate the instance

Design notes
------------
- Time is represented in minutes-from-midnight over a single operating day
  (0 to 1440), which keeps the model small enough for SA experiments while
  still exercising all the real constraints (coupling, depot return,
  maintenance windows, turnaround feasibility).
- Every trip has an origin depot/station and destination station; units
  assigned to a trip must physically be at the origin at trip start, and
  after completing it, become available at the destination station
  (turnaround time later) — this is what makes it a *scheduling* problem
  and not just an assignment problem.
- Two rolling stock TYPES are modeled (like EMU-4 and EMU-8 equivalents),
  each with different capacity and coupling behavior, so demand can force
  the model to couple multiple units together on a single trip.
- Maintenance is modeled as a simple mileage-based constraint: each unit
  has `km_since_maintenance` and a `max_km_between_maintenance` threshold
  from `maintenance_rules.csv`. Any schedule that pushes a unit over this
  threshold without visiting a maintenance-capable depot is infeasible.
"""

import numpy as np
import pandas as pd
import json
import os

# ---------------------------------------------------------------
# CONFIG - tweak these to scale the instance up or down
# ---------------------------------------------------------------
SEED = 42
N_STATIONS = 8          # network size
N_DEPOTS = 3             # subset of stations that are depots (can store/maintain units)
N_MAINT_DEPOTS = 2       # subset of depots that can perform maintenance
N_UNITS = 24             # fleet size (rolling stock units)
N_TRIPS = 60             # trips to be covered over the day
OPERATING_START = 4 * 60     # 04:00 in minutes
OPERATING_END = 24 * 60      # 24:00 in minutes
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def minutes_to_hhmm(m):
    m = int(m) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------
# 1. STATIONS & DEPOTS
# ---------------------------------------------------------------
station_ids = [f"S{i+1}" for i in range(N_STATIONS)]
depot_ids = list(rng.choice(station_ids, size=N_DEPOTS, replace=False))
maint_depot_ids = list(rng.choice(depot_ids, size=N_MAINT_DEPOTS, replace=False))

stations_df = pd.DataFrame({
    "station_id": station_ids,
    "is_depot": [s in depot_ids for s in station_ids],
    "has_maintenance_facility": [s in maint_depot_ids for s in station_ids],
    "platform_capacity": rng.integers(2, 6, size=N_STATIONS),  # simultaneous units storable
})
stations_df.to_csv(os.path.join(OUT_DIR, "stations.csv"), index=False)

# ---------------------------------------------------------------
# 2. DISTANCE / TRAVEL TIME MATRIX
#    Symmetric random network; travel_time scales roughly with distance.
# ---------------------------------------------------------------
dist_rows = []
coords = {s: rng.uniform(0, 100, size=2) for s in station_ids}  # synthetic 2D layout
for i, s1 in enumerate(station_ids):
    for j, s2 in enumerate(station_ids):
        if s1 == s2:
            continue
        dist_km = round(float(np.linalg.norm(coords[s1] - coords[s2])), 1)
        travel_min = max(5, round(dist_km * rng.uniform(0.8, 1.3)))  # ~avg speed factor
        dist_rows.append({
            "from_station": s1,
            "to_station": s2,
            "distance_km": dist_km,
            "travel_time_min": int(travel_min),
        })
distance_df = pd.DataFrame(dist_rows)
distance_df.to_csv(os.path.join(OUT_DIR, "distance_matrix.csv"), index=False)


def lookup_travel(o, d):
    row = distance_df[(distance_df.from_station == o) & (distance_df.to_station == d)]
    return int(row.travel_time_min.values[0]), float(row.distance_km.values[0])


# ---------------------------------------------------------------
# 3. FLEET (rolling stock units)
# ---------------------------------------------------------------
UNIT_TYPES = {
    "EMU4": {"capacity": 320, "can_couple_max": 2},   # 4-car unit
    "EMU8": {"capacity": 680, "can_couple_max": 1},   # fixed 8-car formation, no coupling
}
type_choices = rng.choice(list(UNIT_TYPES.keys()), size=N_UNITS, p=[0.65, 0.35])

fleet_rows = []
for idx in range(N_UNITS):
    utype = type_choices[idx]
    home_depot = rng.choice(depot_ids)
    fleet_rows.append({
        "unit_id": f"U{idx+1:03d}",
        "unit_type": utype,
        "capacity": UNIT_TYPES[utype]["capacity"],
        "max_couple": UNIT_TYPES[utype]["can_couple_max"],
        "home_depot": home_depot,
        "current_location": home_depot,           # at start of day, all at home depot
        "km_since_maintenance": int(rng.integers(0, 8000)),  # random current wear
        "available_from_min": int(rng.choice([0, 0, 0, 60, 120])),  # some units start day late
    })
fleet_df = pd.DataFrame(fleet_rows)
fleet_df.to_csv(os.path.join(OUT_DIR, "fleet.csv"), index=False)

# ---------------------------------------------------------------
# 4. MAINTENANCE RULES
# ---------------------------------------------------------------
maint_rules_df = pd.DataFrame([
    {"unit_type": "EMU4", "max_km_between_maintenance": 12000, "maintenance_duration_min": 240},
    {"unit_type": "EMU8", "max_km_between_maintenance": 15000, "maintenance_duration_min": 360},
])
maint_rules_df.to_csv(os.path.join(OUT_DIR, "maintenance_rules.csv"), index=False)

# ---------------------------------------------------------------
# 5. TRIPS (timetable)
#    Each trip: origin, destination, scheduled departure/arrival,
#    and minimum turnaround time needed before the same unit can
#    be reused for another trip at the destination.
# ---------------------------------------------------------------
trip_rows = []
t = 0
while len(trip_rows) < N_TRIPS:
    origin, dest = rng.choice(station_ids, size=2, replace=False)
    dep_time = int(rng.integers(OPERATING_START, OPERATING_END - 30))
    travel_min, dist_km = lookup_travel(origin, dest)
    arr_time = dep_time + travel_min
    if arr_time > OPERATING_END:
        continue
    trip_rows.append({
        "trip_id": f"T{len(trip_rows)+1:03d}",
        "origin_station": origin,
        "destination_station": dest,
        "departure_min": dep_time,
        "arrival_min": arr_time,
        "departure_time": minutes_to_hhmm(dep_time),
        "arrival_time": minutes_to_hhmm(arr_time),
        "distance_km": dist_km,
        "min_turnaround_min": int(rng.choice([10, 15, 20, 30])),
    })

trips_df = pd.DataFrame(trip_rows).sort_values("departure_min").reset_index(drop=True)
trips_df.to_csv(os.path.join(OUT_DIR, "trips.csv"), index=False)

# ---------------------------------------------------------------
# 6. DEMAND per trip (drives how many units must be coupled together)
#    Peak hours (7-10, 17-20) get higher demand multipliers.
# ---------------------------------------------------------------
def peak_multiplier(dep_min):
    hour = (dep_min // 60) % 24
    if hour in (7, 8, 9, 17, 18, 19):
        return rng.uniform(1.4, 1.9)
    elif hour in (10, 16, 20):
        return rng.uniform(1.0, 1.3)
    else:
        return rng.uniform(0.5, 0.9)

demand_rows = []
base_demand = rng.integers(200, 500, size=len(trips_df))
for i, row in trips_df.iterrows():
    mult = peak_multiplier(row.departure_min)
    demand = int(base_demand[i] * mult)
    demand_rows.append({
        "trip_id": row.trip_id,
        "expected_passenger_demand": demand,
        "min_units_required": int(np.ceil(demand / UNIT_TYPES["EMU8"]["capacity"])) if demand > UNIT_TYPES["EMU4"]["capacity"] else 1,
    })
demand_df = pd.DataFrame(demand_rows)
demand_df.to_csv(os.path.join(OUT_DIR, "demand.csv"), index=False)

# ---------------------------------------------------------------
# 7. META / CONFIG SUMMARY
# ---------------------------------------------------------------
meta = {
    "seed": SEED,
    "n_stations": N_STATIONS,
    "depots": depot_ids,
    "maintenance_depots": maint_depot_ids,
    "n_units": N_UNITS,
    "unit_types": UNIT_TYPES,
    "n_trips": N_TRIPS,
    "operating_window": [minutes_to_hhmm(OPERATING_START), minutes_to_hhmm(OPERATING_END)],
    "files": {
        "stations": "stations.csv",
        "distance_matrix": "distance_matrix.csv",
        "fleet": "fleet.csv",
        "trips": "trips.csv",
        "demand": "demand.csv",
        "maintenance_rules": "maintenance_rules.csv",
    },
    "objective_hint": (
        "Minimize total empty (deadhead) travel + number of units used + "
        "maintenance violations, subject to: every trip covered with enough "
        "capacity, unit continuity (a unit can only start a trip where it "
        "ended the previous one, after turnaround time), and no unit exceeding "
        "max_km_between_maintenance without a maintenance visit at a "
        "maintenance-capable depot."
    ),
}
with open(os.path.join(OUT_DIR, "instance_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("Generated files in", OUT_DIR)
for fname in os.listdir(OUT_DIR):
    print(" -", fname)

print("\nSample trips:")
print(trips_df.head(8).to_string(index=False))
print("\nSample fleet:")
print(fleet_df.head(6).to_string(index=False))
print("\nSample demand:")
print(demand_df.head(6).to_string(index=False))
