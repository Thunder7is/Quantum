# Greedy construction heuristic for initial feasible circulation solution
import pandas as pd
from .data_loader import travel

# Construct initial rolling stock circulation schedule greedily
def greedy_construction(fleet, trips, dist_lookup):
    # Initialize working state tracking for each rolling stock unit
    unit_state = {}
    for r in fleet.itertuples():
        unit_state[r.unit_id] = {
            "location": r.home_depot,
            "avail_time": r.available_from_min,
            "capacity": r.capacity,
            "km_since_maint": r.km_since_maintenance,
            "unit_type": r.unit_type,
        }

    # Initialize assignment data structures
    chains = {uid: [] for uid in unit_state}
    trip_assignment = {}

    # Sort trips chronologically by departure minute
    trips_sorted = trips.sort_values("departure_min").reset_index(drop=True)

    # Process each scheduled trip sequentially
    for trip in trips_sorted.itertuples():
        needed_capacity = trip.expected_passenger_demand
        chosen = []
        covered_capacity = 0

        # Rank all units ready prior to departure by deadhead distance
        candidates = []
        for uid, state in unit_state.items():
            if state["avail_time"] <= trip.departure_min:
                _, dist_km = travel(dist_lookup, state["location"], trip.origin_station)
                candidates.append((dist_km, uid))
        candidates.sort()

        # Allocate minimum units needed to satisfy passenger demand
        for dist_km, uid in candidates:
            if covered_capacity >= needed_capacity:
                break
            chosen.append(uid)
            covered_capacity += unit_state[uid]["capacity"]

        # Fallback assignment to earliest available unit if no candidate ready
        if not chosen:
            earliest_uid = min(unit_state, key=lambda u: unit_state[u]["avail_time"])
            chosen = [earliest_uid]

        # Update spatial and temporal state of selected units
        for uid in chosen:
            state = unit_state[uid]
            _, deadhead_km = travel(dist_lookup, state["location"], trip.origin_station)
            state["location"] = trip.destination_station
            state["avail_time"] = trip.arrival_min + trip.min_turnaround_min
            state["km_since_maint"] += deadhead_km + trip.distance_km
            chains[uid].append(trip.trip_id)

        trip_assignment[trip.trip_id] = chosen

    return chains, trip_assignment

# Convert dictionary of unit chains into structured pandas DataFrame
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
