# Rolling Stock Scheduling — Synthetic Data + Simulated Annealing

## Contents

```
rolling_stock_scheduling/
├── generate_data.py        Synthetic instance generator (stations, fleet, trips, demand)
├── simulated_annealing.py  Greedy construction + SA solver
├── data/                   Generated problem instance (input)
│   ├── stations.csv
│   ├── distance_matrix.csv
│   ├── fleet.csv
│   ├── trips.csv
│   ├── demand.csv
│   ├── maintenance_rules.csv
│   └── instance_meta.json
└── sa_output/               Solver results (output)
    ├── final_schedule.csv    Final unit-by-unit trip chains
    ├── sa_cost_history.csv   Cost vs. temperature over the anneal
    └── run_summary.json      Before/after cost breakdown + SA hyperparameters
```

## How to run

Requires Python 3 with `pandas` and `numpy`.

```bash
cd rolling_stock_scheduling
python3 generate_data.py          # regenerate data/ (optional — already included)
python3 simulated_annealing.py    # runs greedy + SA, writes sa_output/
```

## What it models

A single operating day (04:00–24:00) with:
- A fleet of two rolling-stock types (EMU4, EMU8) with different capacities and coupling rules
- A timetable of trips with origin/destination, timing, and turnaround requirements
- Passenger demand per trip (peak-hour weighted)
- Mileage-based maintenance thresholds per unit type

The solver minimizes deadhead travel, number of units activated, maintenance
violations, and uncovered demand, while eliminating physically infeasible
(timing/location) assignments.


## Mathematical Formulation

Sets

𝑈
U = set of rolling stock units, 
𝑇
T = set of trips, 
𝑆
S = set of stations

Parameters (for trip 
𝑖
i, unit 
𝑢
u)

𝑜
𝑖
,
𝑑
𝑖
o
i
	​

,d
i
	​

 — origin/destination station of trip 
𝑖
i
𝑑
𝑒
𝑝
𝑖
,
𝑎
𝑟
𝑟
𝑖
dep
i
	​

,arr
i
	​

 — departure/arrival time
𝜏
𝑖
τ
i
	​

 — minimum turnaround time at 
𝑑
𝑖
d
i
	​

ℓ
𝑖
ℓ
i
	​

 — distance (km) of trip 
𝑖
i
𝜌
𝑖
ρ
i
	​

 — passenger demand
𝑐
𝑢
c
u
	​

 — capacity of unit 
𝑢
u; 
𝜅
𝑢
κ
u
	​

 — max units unit 
𝑢
u can couple with
ℎ
𝑢
h
u
	​

 — home depot; 
𝑎
𝑢
a
u
	​

 — earliest available time
𝜇
𝑢
μ
u
	​

 — km since last maintenance; 
𝑀
𝑡
𝑦
𝑝
𝑒
(
𝑢
)
M
type(u)
	​

 — max km between maintenance for that type
𝑡
(
𝑠
1
,
𝑠
2
)
,
𝛿
(
𝑠
1
,
𝑠
2
)
t(s
1
	​

,s
2
	​

),δ(s
1
	​

,s
2
	​

) — deadhead travel time/distance between stations

Decision variables

𝑥
𝑢
,
𝑖
∈
{
0
,
1
}
x
u,i
	​

∈{0,1} — unit 
𝑢
u serves trip 
𝑖
i
𝑦
𝑢
,
𝑖
,
𝑗
∈
{
0
,
1
}
y
u,i,j
	​

∈{0,1} — unit 
𝑢
u travels directly from trip 
𝑖
i to trip 
𝑗
j (consecutive in its chain)
𝑧
𝑢
∈
{
0
,
1
}
z
u
	​

∈{0,1} — unit 
𝑢
u is activated (
𝑧
𝑢
≥
𝑥
𝑢
,
𝑖
 
∀
𝑖
z
u
	​

≥x
u,i
	​

 ∀i)
𝑚
𝑢
,
𝑖
∈
{
0
,
1
}
m
u,i
	​

∈{0,1} — unit 
𝑢
u undergoes maintenance immediately after trip 
𝑖
i

Objective


min
⁡
∑
𝑢
,
𝑖
𝛿
deadhead
(
𝑢
,
𝑖
)
 
𝑥
𝑢
,
𝑖
⋅
𝑤
1
+
𝑤
2
∑
𝑢
𝑧
𝑢
+
𝑤
3
∑
𝑢
,
𝑖
(
excess km
)
+
𝑤
4
∑
𝑖
max
⁡
(
0
,
 
𝜌
𝑖
−
∑
𝑢
𝑐
𝑢
𝑥
𝑢
,
𝑖
)
min
u,i
∑
	​

δ
deadhead
	​

(u,i)x
u,i
	​

⋅w
1
	​

+w
2
	​

u
∑
	​

z
u
	​

+w
3
	​

u,i
∑
	​

(excess km)+w
4
	​

i
∑
	​

max(0, ρ
i
	​

−
u
∑
	​

c
u
	​

x
u,i
	​

)

Constraints

Demand coverage

∑
𝑢
∈
𝑈
𝑐
𝑢
 
𝑥
𝑢
,
𝑖
  
≥
  
𝜌
𝑖
∀
𝑖
∈
𝑇
(or relaxed with a slack/shortfall term, as above)
u∈U
∑
	​

c
u
	​

x
u,i
	​

≥ρ
i
	​

∀i∈T(or relaxed with a slack/shortfall term, as above)
Coupling limit (per trip, per type)

∑
𝑢
𝑥
𝑢
,
𝑖
  
≤
  
𝜅
𝑖
∀
𝑖
and specifically 
∑
𝑢
:
 
𝑡
𝑦
𝑝
𝑒
(
𝑢
)
=
𝐸
𝑀
𝑈
8
𝑥
𝑢
,
𝑖
≤
1
 and it excludes any other unit on 
𝑖
u
∑
	​

x
u,i
	​

≤κ
i
	​

∀iand specifically 
u:type(u)=EMU8
∑
	​

x
u,i
	​

≤1 and it excludes any other unit on i
Chain flow conservation (each unit's assigned trips form a single path)

∑
𝑗
𝑦
𝑢
,
𝑖
,
𝑗
=
𝑥
𝑢
,
𝑖
,
∑
𝑖
𝑦
𝑢
,
𝑖
,
𝑗
=
𝑥
𝑢
,
𝑗
∀
𝑢
,
 
∀
𝑖
,
𝑗
∈
𝑇
j
∑
	​

y
u,i,j
	​

=x
u,i
	​

,
i
∑
	​

y
u,i,j
	​

=x
u,j
	​

∀u, ∀i,j∈T
(plus a virtual depot-source arc into each unit's first trip, and a depot-sink arc out of its last)
Temporal/spatial feasibility of an arc — 
𝑦
𝑢
,
𝑖
,
𝑗
=
1
y
u,i,j
	​

=1 is only permitted when

𝑎
𝑟
𝑟
𝑖
+
𝜏
𝑖
+
𝑡
(
𝑑
𝑖
,
𝑜
𝑗
)
  
≤
  
𝑑
𝑒
𝑝
𝑗
arr
i
	​

+τ
i
	​

+t(d
i
	​

,o
j
	​

)≤dep
j
	​

i.e., 
𝑦
𝑢
,
𝑖
,
𝑗
y
u,i,j
	​

 is forced to 0 whenever this inequality fails — this is the constraint that encodes "you can't be two places at once."
Depot start condition — for unit 
𝑢
u's first trip 
𝑖
i in its chain:

𝑎
𝑢
+
𝑡
(
ℎ
𝑢
,
𝑜
𝑖
)
  
≤
  
𝑑
𝑒
𝑝
𝑖
a
u
	​

+t(h
u
	​

,o
i
	​

)≤dep
i
	​

Maintenance threshold — letting 
𝐾
𝑢
,
𝑖
K
u,i
	​

 be cumulative km for unit 
𝑢
u up to trip 
𝑖
i:

𝐾
𝑢
,
𝑖
=
(
1
−
𝑚
𝑢
,
𝑖
−
1
)
𝐾
𝑢
,
𝑖
−
1
+
𝛿
(
prev location
,
𝑜
𝑖
)
+
ℓ
𝑖
,
𝐾
𝑢
,
𝑖
≤
𝑀
𝑡
𝑦
𝑝
𝑒
(
𝑢
)
K
u,i
	​

=(1−m
u,i−1
	​

)K
u,i−1
	​

+δ(prev location,o
i
	​

)+ℓ
i
	​

,K
u,i
	​

≤M
type(u)
	​

and 
𝑚
𝑢
,
𝑖
=
1
m
u,i
	​

=1 is only allowed if 
𝑑
𝑖
d
i
	​

 has a maintenance facility.
Activation linking

𝑧
𝑢
≥
𝑥
𝑢
,
𝑖
∀
𝑖
z
u
	​

≥x
u,i
	​

∀i

## Next step (not included yet)

Formulating the same problem as a QUBO (Quadratic Unconstrained Binary
Optimization) for comparison against quantum annealing.



Step 1 — Load the instance
Read in stations, distances, fleet, trips, demand, and maintenance rules. Convert everything to plain dictionaries (not pandas) for fast lookups, since this data gets accessed thousands of times in the loop ahead.

Step 2 — Build a feasible starting point (greedy construction)
Go through the 60 trips in departure-time order. For each trip:

Look at every unit that's already free by that trip's departure time
Rank candidates by deadhead distance (cheapest reposition first)
Assign the cheapest unit(s), adding a second only if capacity is still short of demand
Update that unit's new location, availability time, and accumulated mileage

This produces one initial solution — a set of trip-chains, one per unit — that's usable but not good. In your run it had 15 timing violations and a total cost of ~20,042.

Step 3 — Score it (the cost function)
Walk every unit's chain in order and tally:

Deadhead km between consecutive trips
Number of distinct units activated
Excess maintenance km beyond the threshold
Uncovered passenger demand
Timing violations (a unit assigned somewhere it can't physically reach in time)

Combine these into one weighted number — the current cost.

Step 4 — Set the "temperature" high
Start at temp = 500. High temperature means the algorithm will be very willing to accept moves that make things temporarily worse, so it can escape bad structural decisions made during the greedy phase rather than getting stuck near them.

Step 5 — Repeat: propose a random change, decide whether to keep it
At each temperature, run 40 random moves. Each move is one of:

Relocate — yank one trip from a unit's chain and hand it to a different unit
Swap — trade one trip between two units
Couple/decouple — add or remove a helper unit on a random trip

After each proposed move, re-score the whole schedule and compute the change in cost, 
Δ
=
cost
new
−
cost
current
Δ=cost
new
	​

−cost
current
	​

:

If 
Δ
<
0
Δ<0 (it's better) → always accept
If 
Δ
≥
0
Δ≥0 (it's worse) → accept anyway with probability 
𝑒
−
Δ
/
𝑇
e
−Δ/T

That probability shrinks as 
𝑇
T drops and as 
Δ
Δ grows — so early on (high 
𝑇
T) it'll tolerate big steps backward, but late in the run (low 
𝑇
T) it only accepts near-improvements. This is the core trick that lets SA avoid getting trapped in a mediocre local arrangement the way pure greedy/hill-climbing would.

Step 6 — Track the best solution seen so far
Even if the current solution wanders somewhere worse (which it's allowed to, per step 5), a separate "best so far" copy is only updated when something genuinely better than any prior state is found. This is what actually gets returned at the end — not whatever the search happens to be sitting on.

Step 7 — Cool down
After each batch of 40 moves, multiply temperature by 0.995. Repeat steps 5–6 at the new, slightly lower temperature. This continues until 
𝑇
T drops below 0.5 — roughly 1,400 temperature levels × 40 moves ≈ 56,000 candidate schedules evaluated total.

Step 8 — Return the best schedule found
Convert the winning unit→trip chains into a readable schedule table, along with the cost breakdown and the full temperature/cost history (useful for plotting convergence).







File	Rows	What each row represents
stations.csv	8	one station in the network
distance_matrix.csv	56	one directed station pair (8 × 7, all pairs excluding self)
fleet.csv	24	one rolling stock unit
trips.csv	60	one scheduled trip
demand.csv	60	one trip's demand figure (1:1 with trips)
maintenance_rules.csv	2	one rule per unit type (EMU4, EMU8)