# Utility for loading instance data and lookup dictionaries
import os
import json
import pandas as pd

# Default data directory pointing to repository data path
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

# Compute station-to-station travel time and distance
def travel(dist_lookup, origin, destination):
    # Zero time and distance if already at station
    if origin == destination:
        return 0, 0.0
    # Lookup precalculated route properties
    return dist_lookup[(origin, destination)]

# Load all CSV and JSON instance artifacts from data directory
def load_instance(data_dir=DATA_DIR):
    # Load stations topology
    stations = pd.read_csv(os.path.join(data_dir, "stations.csv"))
    # Load station pair distance and transit time matrix
    dist = pd.read_csv(os.path.join(data_dir, "distance_matrix.csv"))
    # Load rolling stock fleet initial properties
    fleet = pd.read_csv(os.path.join(data_dir, "fleet.csv"))
    # Load timetable trips and sort in departure time order
    trips = pd.read_csv(os.path.join(data_dir, "trips.csv")).sort_values("departure_min").reset_index(drop=True)
    # Load trip passenger demand profiles
    demand = pd.read_csv(os.path.join(data_dir, "demand.csv"))
    # Load maintenance rules per rolling stock type
    maint_rules = pd.read_csv(os.path.join(data_dir, "maintenance_rules.csv"))
    # Load instance metadata configuration
    meta_path = os.path.join(data_dir, "instance_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    # Merge trips timetable with passenger demand
    trips = trips.merge(demand, on="trip_id")
    # Pre-index distance matrix into key-value pairs
    dist_lookup = {(r.from_station, r.to_station): (r.travel_time_min, r.distance_km) for r in dist.itertuples()}
    # Pre-index maximum allowed kilometers between maintenance
    maint_lookup = {r.unit_type: r.max_km_between_maintenance for r in maint_rules.itertuples()}
    return stations, dist_lookup, fleet, trips, maint_lookup, meta

# Build dictionary lookups for fast evaluations
def get_lookups(fleet, trips):
    # Fast row lookup dictionary for fleet units
    fleet_lookup = {r.unit_id: r._asdict() for r in fleet.itertuples(index=False)}
    # Fast row lookup dictionary for trips
    trips_lookup = {r.trip_id: r._asdict() for r in trips.itertuples(index=False)}
    return fleet_lookup, trips_lookup
