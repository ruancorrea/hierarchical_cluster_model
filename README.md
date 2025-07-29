# Hierarchical Cluster Model Application (HCM)

## Problem Statement: Last-mile Incremental Capacitated Vehicle Routing Problem (ICVRP)

The Incremental Capacitated Vehicle Routing Problem (ICVRP) is a variant of the classic Capacitated Vehicle Routing Problem (CVRP) that focuses on optimizing last-mile delivery operations. In the last-mile ICVRP, the objective is to design efficient delivery routes for a fleet of vehicles with limited capacity, serving a set of geographically distributed customers from a central depot (origin hub).

What makes the ICVRP incremental is the need to dynamically allocate deliveries to vehicles as new demands arrive or as operational constraints change, reflecting real-world logistics scenarios where delivery requests can be received incrementally throughout the day. The problem is particularly relevant for urban logistics, e-commerce, and distribution networks, where minimizing total travel distance, balancing vehicle loads, and ensuring timely deliveries are critical.

Key aspects of the last-mile ICVRP include:
- **Capacity Constraints:** Each vehicle has a maximum load it can carry.
- **Incremental Assignment:** Deliveries may be assigned to vehicles in batches or as they arrive, rather than all at once.
- **Last-mile Focus:** The problem emphasizes the final leg of the delivery process, from the depot to the customers.
- **Optimization Goals:** Minimize total distance traveled, optimize vehicle utilization, and improve service efficiency.

This project addresses the last-mile ICVRP by leveraging hierarchical clustering to group deliveries and incremental algorithms to allocate and route deliveries efficiently.

## Description

This project implements a hierarchical clustering-based approach to solve the Capacitated Vehicle Routing Problem (CVRP). The proposed solution is based on the paper:

The goal is to partition delivery demands into hierarchical clusters, efficiently allocate unit loads, and then solve routing subproblems for each cluster, optimizing vehicle usage and minimizing the total distance traveled.

## Project Structure

- `app/application.py`: Main script to run the complete pipeline (offline and online phases).
- `app/hcm_offline.py`: Implements the offline phase for clustering and unit load allocation.
- `app/hcm_online.py`: Implements the online phase for routing and dynamic delivery allocation.
- `app/types.py`: Defines data structures (instances, deliveries, solutions).
- `app/solutions/`: Example solution files generated (JSON format).
- `shared/ortools.py`: Integration with OR-Tools for solving routing subproblems.
- `eval.py`, `distances.py`, `utils.py`: Utilities and helper functions.

## Installation

```bash
pip install -r requirements.txt
```

## Dependences

- `OSRM Server`: To be able to compute distances over streets, you should download and run an OSRM Server based on OpenStreetMaps.
- `Dataset`: containing the instances to be analyzed. Available for [download here](https://drive.google.com/file/d/1CEL_bCHERTV_dw2eBH0A8TrF8dD980kT/view?usp=sharing).

## How to Run

Example execution:

```bash
python3 -m app.application \
    --train_instances data/cvrp-instances-1.0/train/pa-0 \
    --eval_instances batches/instances/pa-0 \
    --output app/solutions
```

- `--train_instances`: Path to training instances (used for clustering).
- `--eval_instances`: Path to evaluation/test instances.
- `--output`: Folder where solutions will be saved (JSON format).

## Workflow

1. **Offline Phase (Planning - Clustering):**
   - Removes outliers from deliveries using Isolation Forest.
   - Applies KMeans to create delivery clusters.
   - Allocates unit loads proportionally to the size of each cluster.
   - Splits the set of deliveries into training and test partitions.

2. **Online Phase (Allocation - Routing):**
   - For each test instance, dynamically allocates deliveries to unit loads.
   - Solves the routing subproblem for each load using OR-Tools.
   - Generates solution files in JSON format.

## Data Format

### Input Instance (`CVRPInstance`)

```json
{
  "name": "pa-0",
  "region": "pa-0",
  "origin": {"lng": -47.933, "lat": -1.292},
  "vehicle_capacity": 180,
  "deliveries": [
    {"id": "abc123", "point": {"lng": -47.934, "lat": -1.279}, "size": 6},
    ...
  ]
}
```

### Generated Solution (`CVRPSolution`)

```json
{
  "name": "batch_0",
  "vehicles": [
    {
      "origin": {"lng": -47.933, "lat": -1.292},
      "deliveries": [
        {"id": "abc123", "point": {"lng": -47.934, "lat": -1.279}, "size": 6},
        ...
      ]
    },
    ...
  ]
}
```

