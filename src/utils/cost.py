# Shared cost calculation and breakdown for circulation schedules
from .data_loader import travel

# Default objective weights for scheduling multi-criteria evaluation
DEFAULT_WEIGHTS = {
    "deadhead_per_km": 1.0,
    "unit_activation": 150.0,
    "maint_violation_per_km": 5.0,
    "uncovered_demand_per_pax": 2.0,
    "timing_violation": 1000.0,
}

# Calculate multi-objective cost and detailed metrics breakdown
def compute_cost(chains, fleet_lookup, trips_lookup, dist_lookup, maint_lookup, weights=None):
    # Use default weights if custom weights are not provided
    w = DEFAULT_WEIGHTS if weights is None else weights
    total_deadhead_km = 0.0
    total_maint_violation_km = 0.0
    timing_violations = 0
    units_used = 0
    # Track seat capacity provided for each scheduled trip
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

        # Sort unit trips chronologically by departure time
        chain_sorted = sorted(chain, key=lambda tid: trips_lookup[tid]["departure_min"])

        # Trace spatial, temporal, and maintenance progression
        for tid in chain_sorted:
            trip = trips_lookup[tid]
            capacity_supplied[tid] += unit_cap
            # Repositioning deadhead travel to trip origin
            travel_time, deadhead_km = travel(dist_lookup, location, trip["origin_station"])
            total_deadhead_km += deadhead_km
            # Check physical feasibility of arrival before scheduled departure
            arrival_at_origin = avail_time + travel_time
            if arrival_at_origin > trip["departure_min"]:
                timing_violations += 1
            # Accumulate mileage and penalize maintenance overruns
            km_since_maint += deadhead_km + trip["distance_km"]
            if km_since_maint > max_km:
                total_maint_violation_km += (km_since_maint - max_km)
                # Reset maintenance counter upon visiting maintenance threshold
                km_since_maint = 0.0
            # Update unit position and earliest availability after turnaround
            location = trip["destination_station"]
            avail_time = trip["arrival_min"] + trip["min_turnaround_min"]

    # Calculate aggregate unserved passenger demand penalty
    uncovered_pax = 0
    for tid, supplied in capacity_supplied.items():
        demand = trips_lookup[tid]["expected_passenger_demand"]
        if supplied < demand:
            uncovered_pax += (demand - supplied)

    # Compute overall scalarized objective score
    total_cost = (
        w["deadhead_per_km"] * total_deadhead_km
        + w["unit_activation"] * units_used
        + w["maint_violation_per_km"] * total_maint_violation_km
        + w["uncovered_demand_per_pax"] * uncovered_pax
        + w["timing_violation"] * timing_violations
    )

    # Produce metric breakdown dictionary
    breakdown = {
        "total_deadhead_km": round(total_deadhead_km, 1),
        "units_used": units_used,
        "maintenance_violation_km": round(total_maint_violation_km, 1),
        "uncovered_demand_passengers": uncovered_pax,
        "timing_violations": timing_violations,
        "total_cost": round(total_cost, 1),
    }
    return total_cost, breakdown
