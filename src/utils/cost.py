# Shared multi-criteria cost calculation utility for rolling stock scheduling
import pandas as pd
from .data_loader import travel

# Default objective weight parameters
DEFAULT_WEIGHTS = {
    "deadhead": 1.0,
    "activation": 150.0,
    "maintenance": 5.0,
    "demand": 2.0,
    "timing_violation": 1000.0,
}

# Calculate comprehensive operational cost and breakdown metrics for schedule chains
def compute_cost(chains: dict, trips_df, fleet_df, dist_lookup, maint_lookup, weights: dict = None) -> dict:
    # Resolve weights with fallback to defaults
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # Normalize inputs to lookup dicts and auto-detect swapped arguments
    if isinstance(trips_df, pd.DataFrame) and "unit_id" in trips_df.columns:
        trips_df, fleet_df = fleet_df, trips_df
    trips_lookup = {r.trip_id: r._asdict() for r in trips_df.itertuples(index=False)} if isinstance(trips_df, pd.DataFrame) else trips_df
    fleet_lookup = {r.unit_id: r._asdict() for r in fleet_df.itertuples(index=False)} if isinstance(fleet_df, pd.DataFrame) else fleet_df
    # Detect dictionary key swap if applicable
    sample_trip_key = next(iter(trips_lookup)) if trips_lookup else None
    if sample_trip_key and "home_depot" in (trips_lookup[sample_trip_key] if isinstance(trips_lookup[sample_trip_key], dict) else dir(trips_lookup[sample_trip_key])):
        trips_lookup, fleet_lookup = fleet_lookup, trips_lookup

    total_deadhead_km = 0.0
    total_maint_violation_km = 0.0
    timing_violations = 0
    units_used = 0

    # Track seat capacity supplied for each scheduled timetable trip
    capacity_supplied = {tid: 0 for tid in trips_lookup}

    # Evaluate rolling stock unit chains independently
    for uid, chain in chains.items():
        if not chain:
            continue
        units_used += 1
        unit = fleet_lookup[uid]
        location = unit["home_depot"]
        avail_time = unit["available_from_min"]
        km_since_maint = unit["km_since_maintenance"]
        max_km = maint_lookup[unit["unit_type"]]
        unit_cap = unit["capacity"]

        # Sort trips chronologically by departure minute
        chain_sorted = sorted(chain, key=lambda tid: trips_lookup[tid]["departure_min"])

        # Track movements along chain
        for tid in chain_sorted:
            trip = trips_lookup[tid]
            capacity_supplied[tid] += unit_cap
            # Repositioning travel from current location to trip origin
            travel_time, deadhead_km = travel(dist_lookup, location, trip["origin_station"])
            total_deadhead_km += deadhead_km
            # Check timing feasibility of arrival before scheduled departure
            arrival_at_origin = avail_time + travel_time
            if arrival_at_origin > trip["departure_min"]:
                timing_violations += 1
            # Accumulate mileage and record maintenance overruns
            km_since_maint += deadhead_km + trip["distance_km"]
            if km_since_maint > max_km:
                total_maint_violation_km += (km_since_maint - max_km)
                km_since_maint = 0.0
            # Update location and availability timestamp after turnaround
            location = trip["destination_station"]
            avail_time = trip["arrival_min"] + trip["min_turnaround_min"]

    # Calculate aggregate unserved passenger demand
    uncovered_demand = 0
    for tid, supplied in capacity_supplied.items():
        demand = trips_lookup[tid]["expected_passenger_demand"]
        if supplied < demand:
            uncovered_demand += (demand - supplied)

    # Compute overall scalarized objective cost
    total = (
        w.get("deadhead", 1.0) * total_deadhead_km
        + w.get("activation", 150.0) * units_used
        + w.get("maintenance", 5.0) * total_maint_violation_km
        + w.get("demand", 2.0) * uncovered_demand
        + w.get("timing_violation", 1000.0) * timing_violations
    )

    # Return structured cost breakdown dictionary
    return {
        "total": round(total, 1),
        "deadhead_km": round(total_deadhead_km, 1),
        "units_used": units_used,
        "maintenance_violation_km": round(total_maint_violation_km, 1),
        "uncovered_demand": uncovered_demand,
        "timing_violations": timing_violations,
    }
