# Spatial and temporal feasibility evaluation routines
import pandas as pd
from .data_loader import travel

# Extract field attribute or dict key uniformly
def _get(row, key, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)

# Check if unit can travel from trip_a destination to trip_b origin in time including turnaround
def is_feasible_arc(unit_row, trip_a_row, trip_b_row, dist_lookup) -> bool:
    dest_a = _get(trip_a_row, "destination_station")
    orig_b = _get(trip_b_row, "origin_station")
    arr_a = _get(trip_a_row, "arrival_min")
    turnaround_a = _get(trip_a_row, "min_turnaround_min")
    dep_b = _get(trip_b_row, "departure_min")

    # Repositioning travel runtime
    travel_time, _ = travel(dist_lookup, dest_a, orig_b)
    earliest_arrival = arr_a + turnaround_a + travel_time

    # Valid if repositioning finishes prior to scheduled departure
    return bool(earliest_arrival <= dep_b)

# Check if an ordered or unordered list of trips assigned to a unit is physically feasible
def is_feasible_chain(unit_row, trip_ids: list, trips_df, dist_lookup) -> bool:
    # Empty trip assignment is trivially feasible
    if not trip_ids:
        return True

    # Convert trips DataFrame into lookup map if needed
    if isinstance(trips_df, pd.DataFrame):
        trips_lookup = {r.trip_id: r._asdict() for r in trips_df.itertuples(index=False)}
    else:
        trips_lookup = trips_df

    # Sort trips chronologically by departure minute
    sorted_trips = sorted([trips_lookup[tid] for tid in trip_ids], key=lambda t: t["departure_min"])

    # Initial unit depot positioning check
    location = _get(unit_row, "home_depot")
    avail_time = _get(unit_row, "available_from_min")

    first_trip = sorted_trips[0]
    tt, _ = travel(dist_lookup, location, first_trip["origin_station"])
    if avail_time + tt > first_trip["departure_min"]:
        return False

    # Check each consecutive transition using is_feasible_arc
    for idx in range(len(sorted_trips) - 1):
        trip_a = sorted_trips[idx]
        trip_b = sorted_trips[idx + 1]
        if not is_feasible_arc(unit_row, trip_a, trip_b, dist_lookup):
            return False

    return True
