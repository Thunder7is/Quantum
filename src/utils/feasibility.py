# Feasibility validation routines for transitions and trip chains
from .data_loader import travel

# Check if trip B can feasibly follow trip A on the same rolling stock unit
def is_feasible_arc(trip_a, trip_b, dist_lookup):
    # Retrieve travel time required to reposition from destination of A to origin of B
    travel_time, _ = travel(dist_lookup, trip_a["destination_station"], trip_b["origin_station"])
    # Earliest possible arrival at origin of trip B
    earliest_arrival = trip_a["arrival_min"] + trip_a["min_turnaround_min"] + travel_time
    # Arc is feasible if arrival occurs before or at departure time of B
    return earliest_arrival <= trip_b["departure_min"]

# Check if an entire sequence of trips assigned to a unit is physically feasible
def is_feasible_chain(chain, unit_info, trips_lookup, dist_lookup, maint_lookup=None, check_maint=False):
    # Empty chain is always valid
    if not chain:
        return True, "valid_empty"
    # Sort trips chronologically by departure minute
    chain_sorted = sorted(chain, key=lambda tid: trips_lookup[tid]["departure_min"])
    location = unit_info["home_depot"]
    avail_time = unit_info["available_from_min"]
    km_since_maint = unit_info.get("km_since_maintenance", 0.0)
    max_km = maint_lookup[unit_info["unit_type"]] if maint_lookup else float("inf")

    # Verify initial repositioning and consecutive trip transitions
    for tid in chain_sorted:
        trip = trips_lookup[tid]
        travel_time, deadhead_km = travel(dist_lookup, location, trip["origin_station"])
        # Check if unit can reach trip origin in time
        if avail_time + travel_time > trip["departure_min"]:
            return False, f"timing_violation_at_{tid}"
        # Track cumulative mileage if maintenance check requested
        km_since_maint += deadhead_km + trip["distance_km"]
        if check_maint and km_since_maint > max_km:
            return False, f"maintenance_exceeded_at_{tid}"
        # Update current location and next available timestamp
        location = trip["destination_station"]
        avail_time = trip["arrival_min"] + trip["min_turnaround_min"]

    return True, "feasible"
