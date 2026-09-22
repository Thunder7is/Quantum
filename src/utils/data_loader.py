# Shared data loading utility for rolling stock scheduling instances
import os
import pandas as pd

# Default data directory location relative to repository root
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

# Compute station-to-station travel time and distance
def travel(dist_lookup, origin, destination):
    # Zero time and distance if staying at same station
    if origin == destination:
        return 0, 0.0
    # Lookup precalculated route properties
    return dist_lookup[(origin, destination)]

# Load problem instance CSV files and build required lookup structures
def load_instance(data_dir: str = DEFAULT_DATA_DIR) -> dict:
    # Resolve relative data directory path if necessary
    if not os.path.isabs(data_dir) and not os.path.exists(data_dir):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", data_dir))

    # Load stations topology and depot definitions
    stations = pd.read_csv(os.path.join(data_dir, "stations.csv"))
    # Load station-to-station distance and runtime matrix
    dist = pd.read_csv(os.path.join(data_dir, "distance_matrix.csv"))
    # Load rolling stock fleet properties
    fleet = pd.read_csv(os.path.join(data_dir, "fleet.csv"))
    # Load scheduled timetable trips
    trips = pd.read_csv(os.path.join(data_dir, "trips.csv"))
    # Load passenger demand profiles
    demand = pd.read_csv(os.path.join(data_dir, "demand.csv"))
    # Load type-specific maintenance thresholds
    maint_rules = pd.read_csv(os.path.join(data_dir, "maintenance_rules.csv"))

    # Merge trips with demand on trip_id and sort chronologically by departure minute
    trips = trips.merge(demand, on="trip_id").sort_values("departure_min").reset_index(drop=True)

    # Build distance and travel time lookup keyed by (from_station, to_station)
    dist_lookup = {
        (r.from_station, r.to_station): (r.travel_time_min, r.distance_km)
        for r in dist.itertuples()
    }

    # Build maintenance interval lookup keyed by unit_type
    maint_lookup = {
        r.unit_type: r.max_km_between_maintenance
        for r in maint_rules.itertuples()
    }

    # Extract set of station identifiers that operate as depots
    depot_set = set(stations[stations["is_depot"].astype(bool)]["station_id"])

    # Extract set of station identifiers equipped with maintenance facilities
    maint_depot_set = set(stations[stations["has_maintenance_facility"].astype(bool)]["station_id"])

    # Return structured instance dictionary
    return {
        "stations": stations,
        "fleet": fleet,
        "trips": trips,
        "dist_lookup": dist_lookup,
        "maint_lookup": maint_lookup,
        "depot_set": depot_set,
        "maint_depot_set": maint_depot_set,
    }

# Build dictionary lookups for fast evaluations in optimization loops
def get_lookups(fleet, trips):
    # Fast row lookup dictionary for fleet units
    fleet_lookup = {r.unit_id: r._asdict() for r in fleet.itertuples(index=False)}
    # Fast row lookup dictionary for trips
    trips_lookup = {r.trip_id: r._asdict() for r in trips.itertuples(index=False)}
    return fleet_lookup, trips_lookup
