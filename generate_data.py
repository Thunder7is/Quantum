# Synthetic data generator for rolling stock scheduling instances
import argparse
import json
import os
import numpy as np
import pandas as pd

# Convert minutes from midnight to HH:MM format string
def minutes_to_hhmm(m: int) -> str:
    m = int(m) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"

# Generate complete problem instance files
def generate_instance(n_trips: int = 60, n_units: int = 24, n_stations: int = 8,
                      n_depots: int = None, n_maint_depots: int = None,
                      seed: int = 42, out_dir: str = "data") -> None:
    # Resolve default depot and maintenance facility counts if not specified
    if n_depots is None:
        n_depots = max(2, int(round(n_stations * 0.375)))
    if n_maint_depots is None:
        n_maint_depots = max(1, int(round(n_depots * 0.6)))

    # Ensure depot counts do not exceed station count
    n_depots = min(n_depots, n_stations)
    n_maint_depots = min(n_maint_depots, n_depots)

    # Operating parameters (04:00 to 24:00)
    operating_start = 4 * 60
    operating_end = 24 * 60

    # Initialize random generator with deterministic seed
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Stations and Depots
    station_ids = [f"S{i+1}" for i in range(n_stations)]
    depot_ids = list(rng.choice(station_ids, size=n_depots, replace=False))
    maint_depot_ids = list(rng.choice(depot_ids, size=n_maint_depots, replace=False))

    stations_df = pd.DataFrame({
        "station_id": station_ids,
        "is_depot": [s in depot_ids for s in station_ids],
        "has_maintenance_facility": [s in maint_depot_ids for s in station_ids],
        "platform_capacity": rng.integers(2, 6, size=n_stations),
    })
    stations_df.to_csv(os.path.join(out_dir, "stations.csv"), index=False)

    # 2. Distance and Travel Time Matrix
    dist_rows = []
    coords = {s: rng.uniform(0, 100, size=2) for s in station_ids}
    for i, s1 in enumerate(station_ids):
        for j, s2 in enumerate(station_ids):
            if s1 == s2:
                continue
            dist_km = round(float(np.linalg.norm(coords[s1] - coords[s2])), 1)
            travel_min = max(5, round(dist_km * rng.uniform(0.8, 1.3)))
            dist_rows.append({
                "from_station": s1,
                "to_station": s2,
                "distance_km": dist_km,
                "travel_time_min": int(travel_min),
            })
    distance_df = pd.DataFrame(dist_rows)
    distance_df.to_csv(os.path.join(out_dir, "distance_matrix.csv"), index=False)

    # Fast in-memory lookup for distance and travel time
    dist_map = {
        (r.from_station, r.to_station): (int(r.travel_time_min), float(r.distance_km))
        for r in distance_df.itertuples()
    }

    # 3. Fleet Composition
    unit_types = {
        "EMU4": {"capacity": 320, "can_couple_max": 2},
        "EMU8": {"capacity": 680, "can_couple_max": 1},
    }
    type_choices = rng.choice(list(unit_types.keys()), size=n_units, p=[0.65, 0.35])

    fleet_rows = []
    for idx in range(n_units):
        utype = type_choices[idx]
        home_depot = rng.choice(depot_ids)
        fleet_rows.append({
            "unit_id": f"U{idx+1:03d}",
            "unit_type": utype,
            "capacity": unit_types[utype]["capacity"],
            "max_couple": unit_types[utype]["can_couple_max"],
            "home_depot": home_depot,
            "current_location": home_depot,
            "km_since_maintenance": int(rng.integers(0, 8000)),
            "available_from_min": int(rng.choice([0, 0, 0, 60, 120])),
        })
    fleet_df = pd.DataFrame(fleet_rows)
    fleet_df.to_csv(os.path.join(out_dir, "fleet.csv"), index=False)

    # 4. Maintenance Rules
    maint_rules_df = pd.DataFrame([
        {"unit_type": "EMU4", "max_km_between_maintenance": 12000, "maintenance_duration_min": 240},
        {"unit_type": "EMU8", "max_km_between_maintenance": 15000, "maintenance_duration_min": 360},
    ])
    maint_rules_df.to_csv(os.path.join(out_dir, "maintenance_rules.csv"), index=False)

    # 5. Scheduled Timetable Trips
    trip_rows = []
    while len(trip_rows) < n_trips:
        origin, dest = rng.choice(station_ids, size=2, replace=False)
        dep_time = int(rng.integers(operating_start, operating_end - 30))
        travel_min, dist_km = dist_map[(origin, dest)]
        arr_time = dep_time + travel_min
        if arr_time > operating_end:
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

    trips_df = pd.DataFrame(trip_rows).sort_values(["departure_min", "trip_id"]).reset_index(drop=True)
    trips_df.to_csv(os.path.join(out_dir, "trips.csv"), index=False)

    # 6. Passenger Demand Profiles
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
            "min_units_required": int(np.ceil(demand / unit_types["EMU8"]["capacity"])) if demand > unit_types["EMU4"]["capacity"] else 1,
        })
    demand_df = pd.DataFrame(demand_rows)
    demand_df.to_csv(os.path.join(out_dir, "demand.csv"), index=False)

    # 7. Metadata Summary Configuration
    meta = {
        "seed": seed,
        "n_stations": n_stations,
        "depots": depot_ids,
        "maintenance_depots": maint_depot_ids,
        "n_units": n_units,
        "unit_types": unit_types,
        "n_trips": n_trips,
        "operating_window": [minutes_to_hhmm(operating_start), minutes_to_hhmm(operating_end)],
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
    with open(os.path.join(out_dir, "instance_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Generated instance in {out_dir}: {n_trips} trips, {n_units} units, {n_stations} stations (seed={seed})")

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Data Generator for Rolling Stock Scheduling")
    parser.add_argument("--n_trips", type=int, default=60, help="Number of timetable trips to generate")
    parser.add_argument("--n_units", type=int, default=24, help="Number of rolling stock trainset units")
    parser.add_argument("--n_stations", type=int, default=8, help="Number of network stations")
    parser.add_argument("--n_depots", type=int, default=None, help="Number of depot stations")
    parser.add_argument("--n_maint_depots", type=int, default=None, help="Number of maintenance depots")
    parser.add_argument("--seed", type=int, default=42, help="Random number generator seed")
    parser.add_argument("--out_dir", type=str, default="data/", help="Output directory path")
    return parser.parse_args()

# Standalone execution entrypoint
if __name__ == "__main__":
    args = parse_args()
    generate_instance(
        n_trips=args.n_trips,
        n_units=args.n_units,
        n_stations=args.n_stations,
        n_depots=args.n_depots,
        n_maint_depots=args.n_maint_depots,
        seed=args.seed,
        out_dir=args.out_dir,
    )
