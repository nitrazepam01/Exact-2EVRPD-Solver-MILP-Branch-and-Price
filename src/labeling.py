"""
Forward Labeling Algorithm for the Pricing Problem (§4.1.1).
Implements the ESPPDRC (Elementary Shortest Path Problem with Drones
and Resource Constraints) using forward-only label extension.

This is the "pure forward" fallback per user's instruction — no bidirectional
or joining until this version is validated.
"""

from route import Route


class ForwardLabel:
    """
    Forward label L_f = (v, σ, π[0..k̄_d], κ, C, Ω1, Ω2, S, path_vn, path_da).

    Attributes:
        v:     last node added to partial path (vehicle or drone node)
        sigma: last vehicle node visited
        pi:    list[float] of size (k̄_d + 1).
               pi[0] = time when vehicle finishes serving sigma.
               pi[i] = time of drone i returning to sigma (i=1..k̄_d).
        kappa: remaining vehicle capacity
        C:     accumulated dual value  (Σ λ_i for visited customers)
        omega1: frozenset of reachable vehicle nodes from sigma
        omega2: frozenset of reachable drone nodes from sigma
        S:     frozenset of all visited nodes (for elementarity)
        path_vn: list of vehicle nodes in order (for route reconstruction)
        path_da: dict {veh_node: [[drone_route1], ...]} (for route reconstruction)
    """
    __slots__ = ['v', 'sigma', 'pi', 'kappa', 'C',
                 'omega1', 'omega2', 'S', 'path_vn', 'path_da']

    def __init__(self, v, sigma, pi, kappa, C, omega1, omega2, S,
                 path_vn, path_da):
        self.v = v
        self.sigma = sigma
        self.pi = pi          # list of floats, len = k_bar_d + 1
        self.kappa = kappa
        self.C = C
        self.omega1 = omega1  # frozenset
        self.omega2 = omega2  # frozenset
        self.S = S            # frozenset
        self.path_vn = path_vn
        self.path_da = path_da


def dominates_forward(L1, L2, k_bar_d):
    """
    Check if L1 dominates L2 (Proposition 1, with Corollary 1 sorting).

    Conditions:
        (38) σ(L1) = σ(L2)
        (39) π_i(L1) ≤ π_i(L2)  for i=0,...,k̄_d  (after sorting drones)
        (40) κ(L1) ≥ κ(L2)
        (41) Ω1(L2) ⊆ Ω1(L1)
        (42) Ω2(L2) ⊆ Ω2(L1)
        (43) π_i(L1) - C(L1) ≤ π_i(L2) - C(L2)  for i=0,...,k̄_d

    Corollary 1: sort π_i (i=1..k̄_d) in same order before comparing.
    """
    # (38)
    if L1.sigma != L2.sigma:
        return False

    # (40)
    if L1.kappa < L2.kappa:
        return False

    # (41) & (42)
    if not L2.omega1.issubset(L1.omega1):
        return False
    if not L2.omega2.issubset(L1.omega2):
        return False

    # Sort drone π values (Corollary 1): both in ascending order
    if k_bar_d > 0:
        pi1_drones = sorted(L1.pi[1:])
        pi2_drones = sorted(L2.pi[1:])
    else:
        pi1_drones = []
        pi2_drones = []

    # (39): π_0(L1) ≤ π_0(L2) and sorted drone times
    if L1.pi[0] > L2.pi[0]:
        return False
    for a, b in zip(pi1_drones, pi2_drones):
        if a > b:
            return False

    # (43): π_i(L1) - C(L1) ≤ π_i(L2) - C(L2)
    #   ⟺  π_i(L1) - π_i(L2) ≤ C(L1) - C(L2)
    diff_C = L1.C - L2.C
    if L1.pi[0] - L2.pi[0] > diff_C + 1e-9:
        return False
    for a, b in zip(pi1_drones, pi2_drones):
        if a - b > diff_C + 1e-9:
            return False

    return True


def solve_pricing_forward(data, lambda_i, lambda_v0, lambda_d0,
                          k_bar_d, forbidden_arcs=None, forced_arcs=None,
                          col_max=10):
    """
    Solve the pricing problem using pure forward labeling.
    Finds routes with negative reduced cost for a given k̄_d (number of drones).

    Args:
        data: CardiffDataLoader instance
        lambda_i: dict {customer: dual_value} for constraints (33)
        lambda_v0: dual of constraint (34)
        lambda_d0: dual of constraint (35)
        k_bar_d: number of drones carried by the vehicle (0, 1, ..., Γ)
        forbidden_arcs: set of (i,j) arcs that must not be traversed (branching)
        col_max: max number of negative-RC columns to collect

    Returns:
        list[Route]: routes with negative reduced cost
    """
    if forbidden_arcs is None:
        forbidden_arcs = set()
    if forced_arcs is None:
        forced_arcs = set()
    # Note: forced_arcs are enforced via RMP extra_constraints only.
    # Enforcing them in pricing (labeling) would make column generation
    # incomplete (misses some feasible columns), causing LP to be over-estimated
    # and the correct branch to be pruned. RMP constraints handle the semantics.

    n = data.n
    Z = data.Z
    Z_d = data.Z_d
    V_set = frozenset(data.V)
    Z_d_set = frozenset(Z_d)

    # Initial label: start at depot
    # π: all zeros (time 0 at depot)
    # κ: Q - q1_d * k̄_d (capacity minus drone equipment weight)
    # C: 0 (no dual value accumulated)
    # Ω1: all nodes in V (all reachable)
    # Ω2: all drone-eligible customers
    init_pi = [0.0] * (k_bar_d + 1)
    init_kappa = data.Q - data.q1_d * k_bar_d
    init_omega1 = frozenset(i for i in Z if init_kappa >= data.demand[i])
    init_omega2 = Z_d_set

    init_label = ForwardLabel(
        v=0, sigma=0, pi=init_pi, kappa=init_kappa, C=0.0,
        omega1=init_omega1, omega2=init_omega2,
        S=frozenset([0]),
        path_vn=[0], path_da={}
    )

    # Labels stored by sigma (last vehicle node)
    # For forward-only: extend labels, then when a label returns to depot, 
    # compute the full route cost and reduced cost.
    all_labels = {i: [] for i in data.V}
    all_labels[0].append(init_label)

    negative_rc_routes = []

    # Process labels: BFS-like extension
    unprocessed = [init_label]

    while unprocessed and len(negative_rc_routes) < col_max:
        label = unprocessed.pop(0)

        # --- Try extending to depot (complete the route) ---
        if label.sigma != 0 and len(label.path_vn) > 1:
            # Can always return to depot; check time
            max_pi = max(label.pi)
            arrival_depot = max_pi + data.t_v[label.sigma, 0]
            if arrival_depot <= data.l[0] and (label.sigma, 0) not in forbidden_arcs:
                # Complete route found
                # Route cost c_r = total travel time + total waiting time
                # We reconstruct it from the path
                vn = label.path_vn + [0]
                da = dict(label.path_da)
                cost = _compute_label_route_cost(data, vn, da, k_bar_d)

                # Reduced cost: c̄_r = c_r - C(L_f) - λ_v0 - k̄_d * λ_d0
                rc = cost - label.C - lambda_v0 - k_bar_d * lambda_d0

                if rc < -1e-6:
                    # Determine actual drones used
                    has_drones = any(
                        len(dr) > 0
                        for drs in da.values() for dr in drs
                    ) if da else False
                    actual_nd = k_bar_d if has_drones else 0

                    route = Route(vn, da, actual_nd, cost)
                    negative_rc_routes.append(route)
                    if len(negative_rc_routes) >= col_max:
                        break

        # --- Case 1: Extend to a vehicle node w ∈ Ω1 ---
        for w in label.omega1:
            if w in label.S:
                continue
            if (label.sigma, w) in forbidden_arcs:
                continue

            # Feasibility checks
            # Capacity: κ - q_w ≥ 0
            new_kappa = label.kappa - data.demand[w]
            if new_kappa < -1e-9:
                continue

            # Time: max(π_j for j=0..k̄_d) + t_v[σ, w] ≤ l_w
            max_pi = max(label.pi)
            arrival_w = max_pi + data.t_v[label.sigma, w]
            if arrival_w > data.l[w]:
                continue

            # Build new label
            new_pi = [0.0] * (k_bar_d + 1)
            new_pi[0] = arrival_w + data.ser_v[w]  # vehicle finishes serving w
            for i in range(1, k_bar_d + 1):
                new_pi[i] = arrival_w  # drones arrive at w with vehicle

            new_C = label.C + lambda_i.get(w, 0.0)
            new_S = label.S | {w}

            # Update Ω1: remove infeasible vehicle nodes
            new_omega1 = set()
            for wp in label.omega1:
                if wp == w or wp in new_S:
                    continue
                if new_kappa - data.demand[wp] < -1e-9:
                    continue
                if new_pi[0] + data.t_v[w, wp] > data.l[wp]:
                    continue
                new_omega1.add(wp)

            # Update Ω2: remove infeasible drone nodes
            new_omega2 = set()
            if k_bar_d > 0:
                for wp in label.omega2:
                    if wp == w or wp in new_S:
                        continue
                    if new_kappa - data.demand[wp] < -1e-9:
                        continue
                    # Check if any drone can reach wp in time
                    # Use the earliest drone (min pi_i, i≠0)
                    min_drone_pi = min(new_pi[1:]) if k_bar_d > 0 else float('inf')
                    if min_drone_pi + data.t_0 + data.t_d[w, wp] > data.l[wp]:
                        continue
                    new_omega2.add(wp)

            new_label = ForwardLabel(
                v=w, sigma=w, pi=list(new_pi), kappa=new_kappa, C=new_C,
                omega1=frozenset(new_omega1), omega2=frozenset(new_omega2),
                S=new_S,
                path_vn=label.path_vn + [w],
                path_da=dict(label.path_da)
            )

            # Dominance check
            if not _is_dominated(new_label, all_labels.get(w, []), k_bar_d):
                # Remove labels dominated by new_label
                all_labels.setdefault(w, [])
                all_labels[w] = [
                    L for L in all_labels[w]
                    if not dominates_forward(new_label, L, k_bar_d)
                ]
                all_labels[w].append(new_label)
                unprocessed.append(new_label)

        # --- Case 2: Extend to a drone node w ∈ Ω2 ---
        if k_bar_d > 0:
            for w in label.omega2:
                if w in label.S:
                    continue

                # Feasibility: capacity
                new_kappa = label.kappa - data.demand[w]
                if new_kappa < -1e-9:
                    continue

                # Energy feasibility: E[σ, w] = 1
                if data.E[label.sigma, w] != 1:
                    continue

                # Time: ∃ drone i s.t. π_i + t_0 + t_d[σ,w] ≤ l_w
                # Proposition 2: use drone with min π_i (i=1..k̄_d)
                best_drone = None
                best_pi = float('inf')
                for i in range(1, k_bar_d + 1):
                    if label.pi[i] < best_pi:
                        best_pi = label.pi[i]
                        best_drone = i

                drone_arrival = best_pi + data.t_0 + data.t_d[label.sigma, w]
                if drone_arrival > data.l[w]:
                    continue

                # Build new label
                sigma = label.sigma  # unchanged for drone extension
                new_pi = list(label.pi)
                new_pi[0] = label.pi[0]  # vehicle time unchanged
                # Update the assigned drone's return time
                new_pi[best_drone] = (best_pi + data.t_0
                                      + data.t_d[sigma, w]
                                      + data.ser_d[w]
                                      + data.t_d[w, sigma])

                new_C = label.C + lambda_i.get(w, 0.0)
                new_S = label.S | {w}

                # Update Ω1
                new_omega1 = set()
                max_new_pi = max(new_pi)
                for wp in label.omega1:
                    if wp in new_S:
                        continue
                    if new_kappa - data.demand[wp] < -1e-9:
                        continue
                    if max_new_pi + data.t_v[sigma, wp] > data.l[wp]:
                        continue
                    new_omega1.add(wp)

                # Update Ω2
                new_omega2 = set()
                min_drone_pi_new = min(new_pi[1:])
                for wp in label.omega2:
                    if wp == w or wp in new_S:
                        continue
                    if new_kappa - data.demand[wp] < -1e-9:
                        continue
                    if min_drone_pi_new + data.t_0 + data.t_d[sigma, wp] > data.l[wp]:
                        continue
                    new_omega2.add(wp)

                # Update path: add w to drone assignments at sigma
                new_da = {}
                for k, v in label.path_da.items():
                    new_da[k] = [list(dr) for dr in v]
                if sigma not in new_da:
                    new_da[sigma] = [[] for _ in range(k_bar_d)]
                new_da[sigma][best_drone - 1] = list(new_da[sigma][best_drone - 1]) + [w]

                new_label = ForwardLabel(
                    v=w, sigma=sigma, pi=new_pi, kappa=new_kappa, C=new_C,
                    omega1=frozenset(new_omega1), omega2=frozenset(new_omega2),
                    S=new_S,
                    path_vn=list(label.path_vn),
                    path_da=new_da
                )

                # Dominance check
                if not _is_dominated(new_label, all_labels.get(sigma, []), k_bar_d):
                    all_labels.setdefault(sigma, [])
                    all_labels[sigma] = [
                        L for L in all_labels[sigma]
                        if not dominates_forward(new_label, L, k_bar_d)
                    ]
                    all_labels[sigma].append(new_label)
                    unprocessed.append(new_label)

    return negative_rc_routes


def _is_dominated(new_label, existing_labels, k_bar_d):
    """Check if new_label is dominated by any existing label."""
    for L in existing_labels:
        if dominates_forward(L, new_label, k_bar_d):
            return True
    return False


def _compute_label_route_cost(data, vehicle_nodes, drone_assignments, k_bar_d):
    """Compute the total duration of a route from label path info."""
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


def solve_pricing_all(data, lambda_i, lambda_v0, lambda_d0,
                      forbidden_arcs=None, forced_arcs=None, col_max=10):
    """
    Run labeling for all k̄_d = 0, 1, ..., Γ.
    Returns all routes with negative reduced cost (up to col_max).
    """
    all_routes = []
    for k_bar_d in range(data.Gamma + 1):
        if len(all_routes) >= col_max:
            break
        remaining = col_max - len(all_routes)
        routes = solve_pricing_forward(
            data, lambda_i, lambda_v0, lambda_d0,
            k_bar_d, forbidden_arcs, forced_arcs, remaining
        )
        all_routes.extend(routes)
    return all_routes[:col_max]


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import CardiffDataLoader
    from initial_columns import generate_initial_columns
    from rmp_solver import RMPSolver

    data = CardiffDataLoader("Cardiff10_01.txt")

    # Generate initial columns
    init_routes = generate_initial_columns(data)
    print(f"Initial columns: {len(init_routes)}, cost={sum(r.cost for r in init_routes):.2f}")

    # Solve RMP to get duals
    solver = RMPSolver(data, routes=init_routes)
    solver.build()
    obj = solver.solve_lp()
    print(f"RMP LP obj: {obj:.2f}")

    lambda_i, lambda_v0, lambda_d0 = solver.get_duals()
    print(f"Duals: λ_v0={lambda_v0:.4f}, λ_d0={lambda_d0:.4f}")

    # Solve pricing for each k̄_d
    neg_rc_routes = solve_pricing_all(data, lambda_i, lambda_v0, lambda_d0)
    print(f"\nFound {len(neg_rc_routes)} negative-RC routes:")
    for i, r in enumerate(neg_rc_routes):
        rc = r.cost - sum(r.get_alpha(c) * lambda_i.get(c, 0) for c in data.Z)
        rc -= lambda_v0 + r.num_drones * lambda_d0
        print(f"  Route {i}: veh={r.vehicle_nodes}, drones={r.drone_assignments}, "
              f"nd={r.num_drones}, cost={r.cost:.2f}, RC={rc:.4f}")
