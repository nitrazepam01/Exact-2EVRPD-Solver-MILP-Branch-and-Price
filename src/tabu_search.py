"""
Tabu Search heuristic for column generation pricing (§4.1.4).

The TS starts from existing columns with reduced cost = 0.
It uses 3 operators to find negative reduced-cost routes:
  1. Insertion: insert an unserved customer (vehicle or drone node)
  2. Removal: remove a vehicle node not connected to any drone,
              or a drone node
  3. Shift: move the last customer of the drone route with the latest
            completion time to the drone route with the earliest
            completion time, if it reduces the makespan at σ.

Tabu list forbids (arc) for insertion and removal. Shift is never forbidden.
Maximum 100 iterations.
"""

import copy
import random
from route import Route


# ──────────────────────────────────────────────────────────────
# Helper: compute waiting (service) time at a vehicle node
# ──────────────────────────────────────────────────────────────

def _vehicle_wait(data, veh_node, drone_assignments):
    """
    Compute vehicle waiting time at veh_node (= max(s_v, max drone makespan)).
    """
    wait = data.ser_v[veh_node]
    if veh_node in drone_assignments:
        for dr in drone_assignments[veh_node]:
            t = sum(data.t_0 + data.t_d[veh_node, dn] + data.ser_d[dn] + data.t_d[dn, veh_node]
                    for dn in dr)
            wait = max(wait, t)
    return wait


def compute_reduced_cost(data, route, lambda_i, lambda_v0, lambda_d0):
    """Compute reduced cost c̄_r = c_r - Σ α_ir λ_i - λ_v0 - d_r λ_d0."""
    rc = route.cost
    for i in route.get_all_customers():
        rc -= lambda_i.get(i, 0.0)
    rc -= lambda_v0
    rc -= route.num_drones * lambda_d0
    return rc


def _route_cost(data, vehicle_nodes, drone_assignments):
    """Compute route duration (travel + wait at every vehicle node)."""
    total = 0.0
    for idx in range(len(vehicle_nodes) - 1):
        i = vehicle_nodes[idx]
        j = vehicle_nodes[idx + 1]
        total += data.t_v[i, j]
        if j != 0:
            total += _vehicle_wait(data, j, drone_assignments)
    return total


def _check_capacity(data, vehicle_nodes, drone_assignments, num_drones):
    total = data.q1_d * num_drones
    for vn in vehicle_nodes:
        if vn != 0:
            total += data.demand[vn]
    for vn, drs in drone_assignments.items():
        for dr in drs:
            for dn in dr:
                total += data.demand[dn]
    return total <= data.Q


def _check_deadline(data, vehicle_nodes, drone_assignments):
    """Returns True if all deadline constraints are met."""
    arrival = 0.0
    for idx in range(len(vehicle_nodes) - 1):
        i = vehicle_nodes[idx]
        j = vehicle_nodes[idx + 1]
        if idx == 0:
            arrival = data.t_v[i, j]
        else:
            wait_i = _vehicle_wait(data, i, drone_assignments)
            arrival = arrival + wait_i + data.t_v[i, j]
        if j != 0:
            if arrival > data.l[j] + 1e-9:
                return False
            # Check drone deadlines at j
            if j in drone_assignments:
                base = arrival  # vehicle arrives at j
                t_so_far = 0.0
                for dr in drone_assignments[j]:
                    t_round = 0.0
                    for dn in dr:
                        dep = base + t_so_far + data.t_0 + data.t_d[j, dn]
                        if dep > data.l[dn] + 1e-9:
                            return False
                        t_so_far += data.t_0 + data.t_d[j, dn] + data.ser_d[dn] + data.t_d[dn, j]
    if arrival > data.l[0] + 1e-9:
        return False
    return True


def _deep_copy_da(da):
    """Deep copy drone_assignments dict."""
    return {vn: [list(dr) for dr in drs] for vn, drs in da.items()}


# ──────────────────────────────────────────────────────────────
# Operator 1: Insertion
# ──────────────────────────────────────────────────────────────

def _try_insertion(data, vehicle_nodes, drone_assignments, num_drones,
                   unserved, tabu_arcs, best_rc, lambda_i, lambda_v0, lambda_d0):
    """
    Try inserting each unserved customer as a vehicle node or drone node.
    Returns (best_new_state, best_new_rc, tabu_arc_used) or (None, *, None).
    """
    best = None
    best_tabu = None

    for customer in unserved:
        # --- Try as vehicle node ---
        for pos in range(1, len(vehicle_nodes)):
            new_vn = vehicle_nodes[:pos] + [customer] + vehicle_nodes[pos:]
            new_da = _deep_copy_da(drone_assignments)

            if not _check_capacity(data, new_vn, new_da, num_drones):
                continue
            if not _check_deadline(data, new_vn, new_da):
                continue

            prev = vehicle_nodes[pos - 1]
            new_arc = (prev, customer)
            if new_arc in tabu_arcs:
                continue

            cost = _route_cost(data, new_vn, new_da)
            r = Route(new_vn, new_da, num_drones, cost)
            rc = compute_reduced_cost(data, r, lambda_i, lambda_v0, lambda_d0)

            if rc < best_rc:
                best_rc = rc
                best = (new_vn, new_da, num_drones, cost)
                best_tabu = new_arc

        # --- Try as drone node from each vehicle node in the route ---
        if customer in data.Z_d:
            for vn_idx in range(1, len(vehicle_nodes) - 1):
                veh_node = vehicle_nodes[vn_idx]
                if data.E[veh_node, customer] != 1:
                    continue

                new_da = _deep_copy_da(drone_assignments)
                if veh_node not in new_da:
                    if num_drones == 0:
                        continue
                    new_da[veh_node] = [[] for _ in range(num_drones)]

                # Insert at the end of the drone route with min completion time
                nd = max(len(new_da[veh_node]), 1) if veh_node in new_da else num_drones
                best_drone = None
                best_drone_t = float('inf')
                for d_idx, dr in enumerate(new_da[veh_node]):
                    t = sum(data.t_0 + data.t_d[veh_node, dn] + data.ser_d[dn] + data.t_d[dn, veh_node]
                            for dn in dr)
                    if t < best_drone_t:
                        best_drone_t = t
                        best_drone = d_idx

                if best_drone is None:
                    continue

                new_arc = (veh_node, customer)
                if new_arc in tabu_arcs:
                    continue

                new_da[veh_node][best_drone] = list(new_da[veh_node][best_drone]) + [customer]

                if not _check_capacity(data, vehicle_nodes, new_da, num_drones):
                    continue
                if not _check_deadline(data, vehicle_nodes, new_da):
                    continue

                cost = _route_cost(data, vehicle_nodes, new_da)
                r = Route(vehicle_nodes, new_da, num_drones, cost)
                rc = compute_reduced_cost(data, r, lambda_i, lambda_v0, lambda_d0)

                if rc < best_rc:
                    best_rc = rc
                    best = (vehicle_nodes, new_da, num_drones, cost)
                    best_tabu = new_arc

    return best, best_rc, best_tabu


# ──────────────────────────────────────────────────────────────
# Operator 2: Removal
# ──────────────────────────────────────────────────────────────

def _try_removal(data, vehicle_nodes, drone_assignments, num_drones,
                 tabu_arcs, best_rc, lambda_i, lambda_v0, lambda_d0):
    """
    Try removing:
      - A vehicle node NOT connected to any drone (i.e., not in drone_assignments)
      - A drone node from any drone route
    """
    best = None
    best_tabu = None

    # Remove vehicle node
    connected_veh_nodes = set(drone_assignments.keys())
    for pos in range(1, len(vehicle_nodes) - 1):
        vn = vehicle_nodes[pos]
        if vn in connected_veh_nodes:
            continue  # skip if it has drone children

        new_vn = vehicle_nodes[:pos] + vehicle_nodes[pos + 1:]
        new_da = _deep_copy_da(drone_assignments)

        prev = vehicle_nodes[pos - 1]
        rm_arc = (prev, vn)
        if rm_arc in tabu_arcs:
            continue

        if len(new_vn) < 2:
            continue

        cost = _route_cost(data, new_vn, new_da)
        r = Route(new_vn, new_da, num_drones, cost)
        rc = compute_reduced_cost(data, r, lambda_i, lambda_v0, lambda_d0)

        if rc < best_rc:
            best_rc = rc
            best = (new_vn, new_da, num_drones, cost)
            best_tabu = rm_arc

    # Remove drone node
    for veh_node, drs in drone_assignments.items():
        for d_idx, dr in enumerate(drs):
            for customer in dr:
                new_da = _deep_copy_da(drone_assignments)
                new_da[veh_node][d_idx] = [x for x in dr if x != customer]
                # Clean up empty drone slot (keep structure)
                if all(len(r) == 0 for r in new_da[veh_node]):
                    del new_da[veh_node]

                rm_arc = (veh_node, customer)
                if rm_arc in tabu_arcs:
                    continue

                cost = _route_cost(data, vehicle_nodes, new_da)
                r = Route(vehicle_nodes, new_da, num_drones, cost)
                rc = compute_reduced_cost(data, r, lambda_i, lambda_v0, lambda_d0)

                if rc < best_rc:
                    best_rc = rc
                    best = (vehicle_nodes, new_da, num_drones, cost)
                    best_tabu = rm_arc

    return best, best_rc, best_tabu


# ──────────────────────────────────────────────────────────────
# Operator 3: Shift
# ──────────────────────────────────────────────────────────────

def _try_shift(data, vehicle_nodes, drone_assignments, num_drones,
               best_rc, lambda_i, lambda_v0, lambda_d0):
    """
    For each vehicle node with ≥ 2 drone routes, try shifting the last customer
    of the drone route with the latest completion time to the drone route with
    the earliest completion time. Accept if it reduces the makespan at that node.
    """
    best = None

    for veh_node, drs in drone_assignments.items():
        if len(drs) < 2:
            continue

        # Compute completion times for each drone route
        comp_times = []
        for dr in drs:
            t = sum(data.t_0 + data.t_d[veh_node, dn] + data.ser_d[dn] + data.t_d[dn, veh_node]
                    for dn in dr)
            comp_times.append(t)

        old_makespan = max(comp_times)
        latest_idx = comp_times.index(old_makespan)
        earliest_idx = comp_times.index(min(comp_times))

        if latest_idx == earliest_idx:
            continue

        dr_latest = drs[latest_idx]
        if not dr_latest:
            continue

        last_customer = dr_latest[-1]

        # Check energy: drone must be able to fly from veh_node to last_customer
        if data.E[veh_node, last_customer] != 1:
            continue

        # Build new drone assignments after shift
        new_da = _deep_copy_da(drone_assignments)
        new_da[veh_node][latest_idx] = dr_latest[:-1]
        new_da[veh_node][earliest_idx] = list(new_da[veh_node][earliest_idx]) + [last_customer]

        # Compute new completion times
        new_comp_latest = sum(
            data.t_0 + data.t_d[veh_node, dn] + data.ser_d[dn] + data.t_d[dn, veh_node]
            for dn in new_da[veh_node][latest_idx]
        )
        new_comp_earliest = sum(
            data.t_0 + data.t_d[veh_node, dn] + data.ser_d[dn] + data.t_d[dn, veh_node]
            for dn in new_da[veh_node][earliest_idx]
        )
        new_makespan = max(new_comp_latest, new_comp_earliest)

        # Only accept if makespan decreases
        if new_makespan >= old_makespan - 1e-9:
            continue

        # Clean up empty routes
        new_da[veh_node] = [dr for dr in new_da[veh_node] if dr]
        if not new_da[veh_node]:
            del new_da[veh_node]

        if not _check_capacity(data, vehicle_nodes, new_da, num_drones):
            continue
        if not _check_deadline(data, vehicle_nodes, new_da):
            continue

        cost = _route_cost(data, vehicle_nodes, new_da)
        r = Route(vehicle_nodes, new_da, num_drones, cost)
        rc = compute_reduced_cost(data, r, lambda_i, lambda_v0, lambda_d0)

        if rc < best_rc:
            best_rc = rc
            best = (vehicle_nodes, new_da, num_drones, cost)

    return best, best_rc


# ──────────────────────────────────────────────────────────────
# Main TS pricing function
# ──────────────────────────────────────────────────────────────

def tabu_search_pricing(data, solver, lambda_i, lambda_v0, lambda_d0,
                        col_max=10, max_iter=100,
                        forbidden_arcs=None, forced_arcs=None):
    """
    Heuristic column generation via Tabu Search (§4.1.4).

    Starts from columns in solver with reduced cost ≈ 0.
    Returns list of Route objects with negative reduced cost.

    Args:
        data: CardiffDataLoader
        solver: RMPSolver (to get current columns and RMP solution)
        lambda_i, lambda_v0, lambda_d0: dual variables
        col_max: stop if this many negative-RC columns are found
        max_iter: maximum TS iterations (paper: 100)
        forbidden_arcs: set of (i,j) arcs disallowed by branching

    Returns:
        list[Route]: routes with negative reduced cost
    """
    if forbidden_arcs is None:
        forbidden_arcs = set()
    if forced_arcs is None:
        forced_arcs = set()

    # Precompute forced arc lookups (same as in labeling)
    forced_from = {}
    forced_into = {}
    for (src, dst) in forced_arcs:
        forced_from.setdefault(src, set()).add(dst)
        forced_into.setdefault(dst, set()).add(src)

    all_customers = set(data.Z)

    # Collect starting columns: those with RC ≈ 0
    start_routes = []
    for route, mu in solver.get_solution_values():
        if mu > 1e-6:
            rc = compute_reduced_cost(data, route, lambda_i, lambda_v0, lambda_d0)
            if abs(rc) < 1.0:  # approximately zero
                start_routes.append(route)

    if not start_routes:
        return []

    negative_rc_routes = []
    tabu_list = {}  # arc → expiry iteration
    tabu_tenure = 10

    # Run TS from each starting route
    for seed_route in start_routes:
        if len(negative_rc_routes) >= col_max:
            break

        vn = list(seed_route.vehicle_nodes)
        da = _deep_copy_da(seed_route.drone_assignments)
        nd = seed_route.num_drones
        current_rc = compute_reduced_cost(data, seed_route,
                                          lambda_i, lambda_v0, lambda_d0)

        best_in_run_rc = current_rc
        best_in_run = (vn, da, nd, seed_route.cost)

        for it in range(max_iter):
            # Compute served customers
            r_tmp = Route(vn, da, nd, 0)
            served = r_tmp.get_all_customers()
            unserved = all_customers - served

            # Current active tabu arcs
            active_tabu = {arc for arc, expiry in tabu_list.items() if expiry > it}
            # Also include forbidden arcs from branching
            active_tabu |= forbidden_arcs

            # Build forced arc constraints for this current state
            # A vehicle arc (prev→customer) is illegal if it violates forced arcs
            # We pass forced_from/forced_into to insertion operator below

            best_state = None
            best_new_rc = float('inf')
            best_tabu_arc = None
            operator_used = None

            # Operator 1: Insertion
            ins_state, ins_rc, ins_arc = _try_insertion(
                data, vn, da, nd, unserved, active_tabu, float('inf'),
                lambda_i, lambda_v0, lambda_d0)
            if ins_state is not None and ins_rc < best_new_rc:
                best_new_rc = ins_rc
                best_state = ins_state
                best_tabu_arc = ins_arc
                operator_used = 'insertion'

            # Operator 2: Removal
            rm_state, rm_rc, rm_arc = _try_removal(
                data, vn, da, nd, active_tabu, float('inf'),
                lambda_i, lambda_v0, lambda_d0)
            if rm_state is not None and rm_rc < best_new_rc:
                best_new_rc = rm_rc
                best_state = rm_state
                best_tabu_arc = rm_arc
                operator_used = 'removal'

            # Operator 3: Shift (no tabu)
            sh_state, sh_rc = _try_shift(
                data, vn, da, nd, float('inf'),
                lambda_i, lambda_v0, lambda_d0)
            if sh_state is not None and sh_rc < best_new_rc:
                best_new_rc = sh_rc
                best_state = sh_state
                best_tabu_arc = None
                operator_used = 'shift'

            if best_state is None:
                break

            # Accept the best move
            vn, da, nd, cost = best_state

            # Add tabu for arc (insertion/removal only)
            if best_tabu_arc is not None:
                tabu_list[best_tabu_arc] = it + tabu_tenure

            # Collect negative-RC routes
            if best_new_rc < -1e-6:
                real_route = Route(vn, da, nd, cost)
                has_drones = any(len(dr) > 0 for drs in da.values() for dr in drs)
                actual_nd = nd if has_drones else 0
                real_route = Route(vn, da, actual_nd, cost)
                negative_rc_routes.append(real_route)
                if len(negative_rc_routes) >= col_max:
                    return negative_rc_routes

            if best_new_rc < best_in_run_rc:
                best_in_run_rc = best_new_rc
                best_in_run = (vn, da, nd, cost)

    return negative_rc_routes
