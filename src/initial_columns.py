"""
Cheapest Insertion Heuristic for generating initial columns (Algorithm 2, §4.2).
Generates a set of feasible routes to initialize the RMP.
"""

import math
from route import Route


def compute_route_duration(data, vehicle_nodes, drone_assignments):
    """
    Compute the total duration of a route.
    Duration = Σ travel_time(v_i, v_{i+1}) + Σ waiting_time(v_i) for each vehicle node.
    Waiting time = max(vehicle_service_time, max drone makespan at that node).
    """
    total = 0.0
    for idx in range(len(vehicle_nodes) - 1):
        i = vehicle_nodes[idx]
        j = vehicle_nodes[idx + 1]
        total += data.t_v[i, j]
        if j != 0:
            wait = data.ser_v[j]
            if j in drone_assignments:
                for dr in drone_assignments[j]:
                    drone_time = 0.0
                    for dn in dr:
                        drone_time += (data.t_0 + data.t_d[j, dn]
                                       + data.ser_d[dn] + data.t_d[dn, j])
                    wait = max(wait, drone_time)
            total += wait
    return total


def check_deadline_feasibility(data, vehicle_nodes, drone_assignments):
    """
    Check if all deadline constraints are satisfied.
    Returns True if feasible, False otherwise.
    """
    arrival = 0.0  # start at depot at time 0
    for idx in range(len(vehicle_nodes) - 1):
        i = vehicle_nodes[idx]
        j = vehicle_nodes[idx + 1]

        if idx == 0:
            # Travel from depot
            arrival = data.t_v[i, j]
        else:
            # arrival at i was computed, now add wait time at i + travel to j
            wait_i = data.ser_v[i]
            if i in drone_assignments:
                for dr in drone_assignments[i]:
                    drone_time = 0.0
                    for dn in dr:
                        drone_time += (data.t_0 + data.t_d[i, dn]
                                       + data.ser_d[dn] + data.t_d[dn, i])
                    wait_i = max(wait_i, drone_time)
            arrival += wait_i + data.t_v[i, j]

        if j != 0 and arrival > data.l[j]:
            return False

        # Check drone deadlines at node j
        if j != 0 and j in drone_assignments:
            for dr in drone_assignments[j]:
                drone_departure_time = arrival  # vehicle arrives at j
                for dn_idx, dn in enumerate(dr):
                    # drone departs from j to dn
                    # For each trip, drone needs: t_0 (loading) + fly out + serve + fly back
                    drone_arrival_at_dn = drone_departure_time + data.t_0 + data.t_d[j, dn]
                    if drone_arrival_at_dn > data.l[dn]:
                        return False
                    # drone returns to j
                    drone_departure_time = (drone_arrival_at_dn
                                            + data.ser_d[dn] + data.t_d[dn, j])

    # Check depot deadline
    if arrival > data.l[0]:
        return False
    return True


def check_capacity_feasibility(data, vehicle_nodes, drone_assignments, num_drones):
    """Check vehicle capacity constraint."""
    total_load = data.q1_d * num_drones
    for node in vehicle_nodes:
        if node != 0:
            total_load += data.demand[node]
    for vn, drone_routes in drone_assignments.items():
        for dr in drone_routes:
            for dn in dr:
                total_load += data.demand[dn]
    return total_load <= data.Q


def compute_arrival_times(data, vehicle_nodes, drone_assignments):
    """
    Compute arrival time at each vehicle node.
    Returns dict: {node: arrival_time}
    """
    arrivals = {0: 0.0}
    for idx in range(len(vehicle_nodes) - 1):
        i = vehicle_nodes[idx]
        j = vehicle_nodes[idx + 1]

        if idx == 0:
            arrivals[j] = data.t_v[i, j]
        else:
            wait_i = data.ser_v[i]
            if i in drone_assignments:
                for dr in drone_assignments[i]:
                    drone_time = 0.0
                    for dn in dr:
                        drone_time += (data.t_0 + data.t_d[i, dn]
                                       + data.ser_d[dn] + data.t_d[dn, i])
                    wait_i = max(wait_i, drone_time)
            arrivals[j] = arrivals[i] + wait_i + data.t_v[i, j]
    return arrivals


def generate_initial_columns(data):
    """
    Implement Algorithm 2: Cheapest Insertion Heuristic.

    Args:
        data: CardiffDataLoader instance

    Returns:
        list[Route]: initial set of feasible routes
    """
    n = data.n
    k_v = data.k_v
    k_d = data.k_d

    # Step 3: Initialize k_v empty routes, distribute drones evenly
    routes_vn = [[0, 0] for _ in range(k_v)]  # each route: [0, 0] (depot-depot)
    routes_da = [{} for _ in range(k_v)]       # drone assignments per route
    routes_nd = [0] * k_v                       # num_drones per route

    # Distribute drones evenly
    base_drones = k_d // k_v
    extra = k_d % k_v
    for k in range(k_v):
        routes_nd[k] = base_drones + (1 if k < extra else 0)

    # Φ: set of unserved customers
    phi = set(data.Z)

    while phi:
        # Step 5-6: For each i in Φ, compute ϕ(i) = {j: E[i,j]=1, j in Φ, j in Z_d, j≠i}
        phi_reachable = {}
        for i in phi:
            phi_reachable[i] = set()
            for j in phi:
                if j != i and j in data.Z_d and data.E[i, j] == 1:
                    phi_reachable[i].add(j)

        # Step 7: Sort Φ in non-ascending order by |ϕ(i)|
        sorted_phi = sorted(phi, key=lambda i: len(phi_reachable.get(i, set())),
                            reverse=True)

        # Step 8: i' = first node in sorted Φ
        i_prime = sorted_phi[0]

        # Step 9-14: Find best insertion position for i'
        delta_min = float('inf')
        best_route = None
        best_pos = None

        for k in range(k_v):
            vn = routes_vn[k]
            da = routes_da[k]
            nd = routes_nd[k]

            # Try each insertion position (between existing vehicle nodes)
            for pos in range(1, len(vn)):
                # Create candidate route with i' inserted at position pos
                new_vn = vn[:pos] + [i_prime] + vn[pos:]

                # Check capacity
                if not check_capacity_feasibility(data, new_vn, da, nd):
                    continue

                # Check deadline feasibility
                if not check_deadline_feasibility(data, new_vn, da):
                    continue

                # Compute incremental cost
                new_cost = compute_route_duration(data, new_vn, da)
                old_cost = compute_route_duration(data, vn, da)
                delta = new_cost - old_cost

                if delta < delta_min:
                    delta_min = delta
                    best_route = k
                    best_pos = pos

        if best_route is None:
            # Cannot insert i' — skip it (will be handled by dummy variables)
            phi.discard(i_prime)
            continue

        # Step 15: Insert i' into best position
        routes_vn[best_route] = (routes_vn[best_route][:best_pos]
                                 + [i_prime]
                                 + routes_vn[best_route][best_pos:])
        phi.discard(i_prime)

        # Step 16-20: Assign drone-reachable customers from ϕ(i')
        drone_candidates = phi_reachable.get(i_prime, set()) & phi
        if drone_candidates and routes_nd[best_route] > 0:
            # Sort by deadline (non-descending)
            sorted_candidates = sorted(drone_candidates, key=lambda j: data.l[j])

            # Initialize drone completion times for this vehicle node
            nd = routes_nd[best_route]
            if i_prime not in routes_da[best_route]:
                routes_da[best_route][i_prime] = [[] for _ in range(nd)]

            drone_routes = routes_da[best_route][i_prime]

            for j in sorted_candidates:
                if j not in phi:
                    continue

                # Find drone route with minimum completion time
                best_drone = None
                best_completion = float('inf')
                for d_idx in range(nd):
                    # Compute current completion time of drone d_idx at i'
                    dr = drone_routes[d_idx]
                    completion = 0.0
                    for dn in dr:
                        completion += (data.t_0 + data.t_d[i_prime, dn]
                                       + data.ser_d[dn] + data.t_d[dn, i_prime])
                    if completion < best_completion:
                        best_completion = completion
                        best_drone = d_idx

                if best_drone is None:
                    continue

                # Check if inserting j at the end of drone route is feasible
                # 1. Energy check
                if data.E[i_prime, j] != 1:
                    continue

                # 2. Capacity check
                new_da = dict(routes_da[best_route])
                new_da[i_prime] = [list(dr) for dr in drone_routes]
                new_da[i_prime][best_drone] = list(new_da[i_prime][best_drone]) + [j]

                if not check_capacity_feasibility(data, routes_vn[best_route],
                                                  new_da, nd):
                    continue

                # 3. Deadline check: drone arrival at j must be ≤ l[j]
                arrivals = compute_arrival_times(data, routes_vn[best_route],
                                                 routes_da[best_route])
                vehicle_arrival_at_iprime = arrivals.get(i_prime, 0)
                drone_departure = vehicle_arrival_at_iprime + best_completion
                drone_arrival_at_j = drone_departure + data.t_0 + data.t_d[i_prime, j]
                if drone_arrival_at_j > data.l[j]:
                    continue

                # 4. Overall route feasibility
                if not check_deadline_feasibility(data, routes_vn[best_route], new_da):
                    continue

                # Accept insertion
                drone_routes[best_drone].append(j)
                phi.discard(j)

            # Clean up empty drone routes from assignments
            if all(len(dr) == 0 for dr in drone_routes):
                del routes_da[best_route][i_prime]

    # Build Route objects
    result = []
    for k in range(k_v):
        vn = routes_vn[k]
        if len(vn) <= 2:
            # Empty route (only depot-depot), skip
            continue
        da = routes_da[k]
        nd = routes_nd[k]
        # If no drones were actually used, set num_drones to 0
        has_drones = any(len(dr) > 0 for drs in da.values() for dr in drs) if da else False
        actual_drones = nd if has_drones else 0
        cost = compute_route_duration(data, vn, da)
        route = Route(vn, da, actual_drones, cost)
        result.append(route)

    return result


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import CardiffDataLoader

    data = CardiffDataLoader("Cardiff10_01.txt")
    routes = generate_initial_columns(data)

    print(f"Generated {len(routes)} initial routes")
    all_customers = set()
    total_cost = 0.0
    for i, r in enumerate(routes):
        print(f"\n  Route {i+1}: {r}")
        print(f"    Vehicle nodes: {r.vehicle_nodes}")
        print(f"    Drone assignments: {r.drone_assignments}")
        print(f"    Customers: {r.get_all_customers()}")
        print(f"    Cost: {r.cost:.2f}")
        print(f"    Drones: {r.num_drones}")
        all_customers |= r.get_all_customers()
        total_cost += r.cost

    missing = set(range(1, data.n + 1)) - all_customers
    print(f"\nAll customers covered: {len(missing) == 0}")
    if missing:
        print(f"  Missing (need dummy vars): {missing}")
    print(f"Total cost: {total_cost:.2f}")
