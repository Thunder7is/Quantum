# Synthetic data generator for rolling stock scheduling instances
# Mirrors the IBM rolling stock paper (Guth Jarkovsky et al., 2026) adapted to Indian Railway Network
import argparse
import json
import os
import numpy as np
import pandas as pd

# Named station constants adapted to Indian Railway Network
STATIONS = ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai"]
MAINTENANCE_STATION = "Kolkata"
DEPOT_STATIONS = ["Mumbai", "Delhi", "Kolkata"]

# Minimum departure gap between consecutive trips on the same route (minutes)
MIN_DEPARTURE_GAP_MIN = 800

# Approximate real Indian inter-city rail distances (km) and travel times (min at ~130 km/h)
CITY_DISTANCES = {
    ("Mumbai", "Delhi"): (1400.0, 646),
    ("Mumbai", "Bangalore"): (980.0, 452),
    ("Mumbai", "Kolkata"): (1970.0, 909),
    ("Mumbai", "Chennai"): (1330.0, 614),
    ("Delhi", "Bangalore"): (2150.0, 992),
    ("Delhi", "Kolkata"): (1470.0, 678),
    ("Delhi", "Chennai"): (2180.0, 1006),
    ("Bangalore", "Kolkata"): (1870.0, 863),
    ("Bangalore", "Chennai"): (360.0, 166),
    ("Kolkata", "Chennai"): (1660.0, 766),
}

# Convert minutes from midnight to HH:MM format string
def minutes_to_hhmm(m: int) -> str:
    m = int(m) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"

# Generate complete problem instance files
def generate_instance(n_trips: int = 30, n_units: int = 30, n_stations: int = 5,
                      n_days: int = 2, n_depots: int = None, n_maint_depots: int = None,
                      seed: int = 42, out_dir: str = "data") -> None:
    # Operating parameters (04:00 to 04:00 next day = 28h window, 52h window for 2 days)
    operating_start = 4 * 60
    operating_end = 52 * 60

    # Initialize random generator with deterministic seed
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Stations and Depots
    station_ids = STATIONS[:n_stations]
    depot_ids = [s for s in DEPOT_STATIONS if s in station_ids]
    maint_depot_ids = [MAINTENANCE_STATION] if MAINTENANCE_STATION in station_ids else [station_ids[0]]

    # Allow CLI overrides if explicitly passed
    if n_depots is not None and n_depots < len(depot_ids):
        depot_ids = depot_ids[:n_depots]
    if n_maint_depots is not None and n_maint_depots < len(maint_depot_ids):
        maint_depot_ids = maint_depot_ids[:n_maint_depots]

    stations_df = pd.DataFrame({
        "station_id": station_ids,
        "is_depot": [s in depot_ids for s in station_ids],
        "has_maintenance_facility": [s in maint_depot_ids for s in station_ids],
        "platform_capacity": rng.integers(2, 6, size=len(station_ids)),
    })
    stations_df.to_csv(os.path.join(out_dir, "stations.csv"), index=False)

    # 2. Distance and Travel Time Matrix
    dist_rows = []
    for (c1, c2), (dist_km, travel_min) in CITY_DISTANCES.items():
        if c1 in station_ids and c2 in station_ids:
            dist_rows.append({
                "from_station": c1,
                "to_station": c2,
                "distance_km": float(dist_km),
                "travel_time_min": int(travel_min),
            })
            dist_rows.append({
                "from_station": c2,
                "to_station": c1,
                "distance_km": float(dist_km),
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

        # Enforce minimum departure gap between consecutive trips on the same route
        same_route = [r for r in trip_rows if r["origin_station"] == origin and r["destination_station"] == dest]
        for r in same_route:
            if abs(dep_time - r["departure_min"]) < MIN_DEPARTURE_GAP_MIN:
                dep_time = max(dep_time, r["departure_min"]) + MIN_DEPARTURE_GAP_MIN

        # Verify no remaining departure conflict exists on this route
        if any(abs(dep_time - r["departure_min"]) < MIN_DEPARTURE_GAP_MIN for r in same_route):
            continue

        arr_time = dep_time + travel_min
        if arr_time >= 3000 or arr_time > operating_end:
            continue

        day = 1 if dep_time < 1440 else 2
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
            "day": day,
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
        "n_stations": n_stations,
        "stations": station_ids,
        "maintenance_station": MAINTENANCE_STATION,
        "depots": depot_ids,
        "scheduling_days": n_days,
        "n_units": n_units,
        "n_trips": n_trips,
        "paper_reference": "Guth Jarkovsky et al., 2026 - adapted to Indian Railway Network",
    }
    with open(os.path.join(out_dir, "instance_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Generated instance in {out_dir}: {n_trips} trips, {n_units} units, {n_stations} stations (seed={seed})")

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Data Generator for Rolling Stock Scheduling")
    parser.add_argument("--n_trips", type=int, default=30, help="Number of timetable trips to generate")
    parser.add_argument("--n_units", type=int, default=30, help="Number of rolling stock trainset units")
    parser.add_argument("--n_stations", type=int, default=5, help="Number of network stations")
    parser.add_argument("--n_days", type=int, default=2, help="Number of scheduling days")
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
        n_days=args.n_days,
        n_depots=args.n_depots,
        n_maint_depots=args.n_maint_depots,
        seed=args.seed,
        out_dir=args.out_dir,
    )
