# Greedy construction heuristic using arc feasibility and capacity-driven coupling
import pandas as pd
from .data_loader import travel
from .feasibility import is_feasible_arc

# Construct greedy rolling stock circulation chains in departure time order
def greedy_construction(instance: dict, *args, **kwargs) -> dict:
    # Support both dictionary instance and positional invocation
    if isinstance(instance, dict) and "fleet" in instance:
        fleet = instance["fleet"]
        trips = instance["trips"]
        dist_lookup = instance["dist_lookup"]
    elif len(args) >= 2:
        fleet = instance
        trips = args[0]
        dist_lookup = args[1]
    else:
        fleet = instance.get("fleet") if isinstance(instance, dict) else instance
        trips = kwargs.get("trips")
        dist_lookup = kwargs.get("dist_lookup")

    # Fast row dictionary lookups for fleet units
    fleet_rows = {r.unit_id: r._asdict() for r in fleet.itertuples(index=False)}

    # Track operational unit state during greedy rollout
    unit_state = {}
    for uid, r in fleet_rows.items():
        unit_state[uid] = {
            "location": r["home_depot"],
            "avail_time": r["available_from_min"],
            "capacity": r["capacity"],
            "last_trip": None,
            "unit_row": r,
        }

    # Initialize empty circulation chains for all rolling stock units
    chains = {uid: [] for uid in fleet_rows}

    # Process timetable trips in chronological departure order
    trips_sorted = trips.sort_values("departure_min").reset_index(drop=True)

    for trip_tuple in trips_sorted.itertuples(index=False):
        trip = trip_tuple._asdict()
        needed_demand = trip.get("expected_passenger_demand", 0)

        # Identify all candidate units that can feasibly reach trip origin in time
        candidates = []
        for uid, state in unit_state.items():
            if state["last_trip"] is not None:
                # Validate transition using shared is_feasible_arc function
                feasible = is_feasible_arc(state["unit_row"], state["last_trip"], trip, dist_lookup)
                if feasible:
                    _, deadhead_km = travel(dist_lookup, state["location"], trip["origin_station"])
                    candidates.append((deadhead_km, uid))
            else:
                # Initial depot departure feasibility check
                tt, deadhead_km = travel(dist_lookup, state["location"], trip["origin_station"])
                if state["avail_time"] + tt <= trip["departure_min"]:
                    candidates.append((deadhead_km, uid))

        # Sort candidate units by deadhead positioning distance
        candidates.sort(key=lambda item: item[0])

        chosen = []
        if candidates:
            # Assign primary unit
            first_uid = candidates[0][1]
            chosen.append(first_uid)
            assigned_capacity = unit_state[first_uid]["capacity"]

            # Add second unit only if single-unit capacity is less than demand
            if assigned_capacity < needed_demand and len(candidates) > 1:
                second_uid = candidates[1][1]
                chosen.append(second_uid)
        else:
            # Fallback assignment to unit with earliest availability
            earliest_uid = min(unit_state, key=lambda u: unit_state[u]["avail_time"])
            chosen.append(earliest_uid)

        # Update spatial and temporal state of selected units
        for uid in chosen:
            st = unit_state[uid]
            st["location"] = trip["destination_station"]
            st["avail_time"] = trip["arrival_min"] + trip["min_turnaround_min"]
            st["last_trip"] = trip
            chains[uid].append(trip["trip_id"])

    return chains

# Convert dictionary of unit chains into structured pandas DataFrame
def chains_to_schedule_df(chains: dict, trips_df: pd.DataFrame) -> pd.DataFrame:
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
