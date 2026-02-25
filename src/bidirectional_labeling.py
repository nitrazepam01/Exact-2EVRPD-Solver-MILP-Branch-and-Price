"""
Bidirectional Labeling Algorithm for the Pricing Problem (§4.1.1-4.1.3).

Implements:
  - Forward labels (§4.1.1, updated from labeling.py with cutoff condition)
  - Backward labels (§4.1.2, new)
  - Label Joining with Algorithm 1 pairing (§4.1.3)
  - Proposition 1 (forward dominance), Corollary 1 (sort trick)
  - Proposition 2 (use min-pi drone for forward)
  - Proposition 3 (backward dominance, NO sort trick)

Forward labels extend while: min_i{π_i(L_f)} < l_0 / 2  (for k̄_d ≥ 1)
                         or: π_0(L_f) - s_v[σ] < l_0 / 2  (for k̄_d = 0)
Backward labels extend while: min_i{π_i(L_b)} > l_0 / 2  (for k̄_d ≥ 1)
                          or: π_0(L_b) > l_0 / 2           (for k̄_d = 0)
"""

from route import Route


# ─────────────────────────────────────────────────────────────────────────────
# Forward Label
# ─────────────────────────────────────────────────────────────────────────────

class ForwardLabel:
    """
    L_f = (v, σ, π[0..k̄_d], κ, C, Ω1, Ω2, S, path_vn, path_da)

    π_0: time when vehicle finishes serving σ (= arrival + s_v[σ])
    π_i (i≥1): return time of drone i to σ after its last trip
    κ: remaining vehicle capacity
    C: accumulated dual value Σ λ_i for visited customers
    """
    __slots__ = ['v', 'sigma', 'pi', 'kappa', 'C',
                 'omega1', 'omega2', 'S', 'path_vn', 'path_da']

    def __init__(self, v, sigma, pi, kappa, C, omega1, omega2, S,
                 path_vn, path_da):
        self.v = v
        self.sigma = sigma
        self.pi = pi          # list[float], len = k̄_d + 1
        self.kappa = kappa
        self.C = C
        self.omega1 = omega1  # frozenset
        self.omega2 = omega2  # frozenset
        self.S = S            # frozenset (visited nodes)
        self.path_vn = path_vn
        self.path_da = path_da


class BackwardLabel:
    """
    L_b = (v, σ, π[0..k̄_d], ρ[0..k̄_d], κ, C1, C2, Ω1, Ω2, S, path_vn, path_da)

    π_0: latest arrival time of vehicle at σ
    π_i (i≥1): latest departure time of drone i from σ to serve first customer
    ρ_0: service time s_v[σ] of vehicle at σ
    ρ_i (i≥1): total duration of drone route i
    κ: remaining vehicle capacity
    C1: accumulated dual value of backward partial path
    C2: time duration from σ to end depot (0)
    """
    __slots__ = ['v', 'sigma', 'pi', 'rho', 'kappa', 'C1', 'C2',
                 'omega1', 'omega2', 'S', 'path_vn', 'path_da']

    def __init__(self, v, sigma, pi, rho, kappa, C1, C2,
                 omega1, omega2, S, path_vn, path_da):
        self.v = v
        self.sigma = sigma
        self.pi = pi          # list[float], len = k̄_d + 1
        self.rho = rho        # list[float], len = k̄_d + 1
        self.kappa = kappa
        self.C1 = C1
        self.C2 = C2
        self.omega1 = omega1  # frozenset
        self.omega2 = omega2  # frozenset
        self.S = S            # frozenset
        self.path_vn = path_vn
        self.path_da = path_da


# ─────────────────────────────────────────────────────────────────────────────
# Forward Dominance (Proposition 1 + Corollary 1)
# ─────────────────────────────────────────────────────────────────────────────

def _dominates_forward(L1, L2, k_bar_d):
    """
    L1 ≺ L2 (L1 dominates L2) if:
      (38) σ(L1) = σ(L2)
      (39) π_i(L1) ≤ π_i(L2) for i=0..k̄_d  (after Corollary 1 sort of drone parts)
      (40) κ(L1) ≥ κ(L2)
      (41) Ω1(L2) ⊆ Ω1(L1)
      (42) Ω2(L2) ⊆ Ω2(L1)
      (43) π_i(L1) - C(L1) ≤ π_i(L2) - C(L2) for i=0..k̄_d
           ⟺ max_i{π_i(L1) - π_i(L2)} ≤ max{0, C(L1) - C(L2)}
    """
    if L1.sigma != L2.sigma:
        return False
    if L1.kappa < L2.kappa:
        return False
    if not L2.omega1.issubset(L1.omega1):
        return False
    if not L2.omega2.issubset(L1.omega2):
        return False

    # Compare π_0 separately (vehicle, no sorting)
    if L1.pi[0] > L2.pi[0] + 1e-9:
        return False
    diff_C = L1.C - L2.C
    if L1.pi[0] - L2.pi[0] > max(0.0, diff_C) + 1e-9:
        return False

    if k_bar_d > 0:
        # Corollary 1: sort drone π values in same order before comparing
        pi1_drones = sorted(L1.pi[1:])
        pi2_drones = sorted(L2.pi[1:])
        for a, b in zip(pi1_drones, pi2_drones):
            if a > b + 1e-9:
                return False
            if a - b > max(0.0, diff_C) + 1e-9:
                return False

    return True


def _is_forward_dominated(new_label, existing_labels, k_bar_d):
    for L in existing_labels:
        if _dominates_forward(L, new_label, k_bar_d):
            return True
    return False


def _prune_forward_dominated(new_label, existing_labels, k_bar_d):
    return [L for L in existing_labels
            if not _dominates_forward(new_label, L, k_bar_d)]


# ─────────────────────────────────────────────────────────────────────────────
# Backward Dominance (Proposition 3)
# ─────────────────────────────────────────────────────────────────────────────

def _dominates_backward(L1, L2, k_bar_d):
    """
    L1_b ≺ L2_b if:
      σ(L1) = σ(L2)
      π_i(L1) ≥ π_i(L2)   for i=0..k̄_d   (later departure = better for backward)
      κ(L1) ≥ κ(L2)
      Ω1(L2) ⊆ Ω1(L1)
      Ω2(L2) ⊆ Ω2(L1)
      ρ_i(L1) + C2(L1) - C1(L1) - ρ_i(L2) - C2(L2) + C1(L2) ≤ 0  for i=0..k̄_d
    Note: Corollary 1 does NOT apply for backward labels (§ text: "we cannot use Corollary 1")
    """
    if L1.sigma != L2.sigma:
        return False
    if L1.kappa < L2.kappa:
        return False
    if not L2.omega1.issubset(L1.omega1):
        return False
    if not L2.omega2.issubset(L1.omega2):
        return False

    # π_i(L1) ≥ π_i(L2) for all i
    for a, b in zip(L1.pi, L2.pi):
        if a < b - 1e-9:
            return False

    # ρ_i(L1) + C2(L1) - C1(L1) ≤ ρ_i(L2) + C2(L2) - C1(L2) for all i
    adj1 = L1.C2 - L1.C1
    adj2 = L2.C2 - L2.C1
    for r1, r2 in zip(L1.rho, L2.rho):
        if r1 + adj1 > r2 + adj2 + 1e-9:
            return False

    return True


def _is_backward_dominated(new_label, existing_labels, k_bar_d):
    for L in existing_labels:
        if _dominates_backward(L, new_label, k_bar_d):
            return True
    return False


def _prune_backward_dominated(new_label, existing_labels, k_bar_d):
    return [L for L in existing_labels
            if not _dominates_backward(new_label, L, k_bar_d)]


# ─────────────────────────────────────────────────────────────────────────────
# Forward Label Extension
# ─────────────────────────────────────────────────────────────────────────────

def _extend_forward(data, label, k_bar_d, lambda_i, forbidden_arcs, forced_from, forced_into,
                    all_fwd_labels, unprocessed_fwd, half_time):
    """
    Extend a forward label with the half-time cutoff condition.
    For k̄_d ≥ 1: extend while min_i{π_i(L_f), i=1..k̄_d} < l_0 / 2
    For k̄_d = 0: extend while π_0(L_f) - s_v[σ] < l_0 / 2
    """
    if k_bar_d >= 1:
        cutoff_val = min(label.pi[1:])
    else:
        cutoff_val = label.pi[0] - data.ser_v[label.sigma] if label.sigma != 0 else 0.0

    if cutoff_val >= half_time:
        return  # Do not extend further

    # Case 1: Extend to vehicle node w ∈ Ω1
    for w in label.omega1:
        if w in label.S:
            continue
        if (label.sigma, w) in forbidden_arcs:
            continue

        new_kappa = label.kappa - data.demand[w]
        if new_kappa < -1e-9:
            continue

        max_pi = max(label.pi)
        arrival_w = max_pi + data.t_v[label.sigma, w]
        if arrival_w > data.l[w] + 1e-9:
            continue

        new_pi = [0.0] * (k_bar_d + 1)
        new_pi[0] = arrival_w + data.ser_v[w]
        for i in range(1, k_bar_d + 1):
            new_pi[i] = arrival_w  # drone returns to w with vehicle

        new_C = label.C + lambda_i.get(w, 0.0)
        new_S = label.S | {w}

        # Update Ω1 (paper eq after Case 1)
        new_omega1 = set()
        for wp in label.omega1:
            if wp in new_S:
                continue
            if new_kappa - data.demand[wp] < -1e-9:
                continue
            if max(new_pi) + data.t_v[w, wp] > data.l[wp] + 1e-9:
                continue
            new_omega1.add(wp)

        # Update Ω2
        new_omega2 = set()
        if k_bar_d > 0:
            min_drone_pi = min(new_pi[1:])
            for wp in label.omega2:
                if wp in new_S:
                    continue
                if new_kappa - data.demand[wp] < -1e-9:
                    continue
                if min_drone_pi + data.t_0 + data.t_d[w, wp] > data.l[wp] + 1e-9:
                    continue
                new_omega2.add(wp)

        new_da = {k: [list(dr) for dr in v] for k, v in label.path_da.items()}

        new_label = ForwardLabel(
            v=w, sigma=w, pi=new_pi, kappa=new_kappa, C=new_C,
            omega1=frozenset(new_omega1), omega2=frozenset(new_omega2),
            S=new_S, path_vn=label.path_vn + [w], path_da=new_da
        )

        bucket = all_fwd_labels.setdefault(w, [])
        if not _is_forward_dominated(new_label, bucket, k_bar_d):
            all_fwd_labels[w] = _prune_forward_dominated(new_label, bucket, k_bar_d)
            all_fwd_labels[w].append(new_label)
            unprocessed_fwd.append(new_label)

    # Case 2: Extend to drone node w ∈ Ω2 (Proposition 2: use min-π drone)
    if k_bar_d > 0:
        for w in label.omega2:
            if w in label.S:
                continue
            if data.E[label.sigma, w] != 1:
                continue

            new_kappa = label.kappa - data.demand[w]
            if new_kappa < -1e-9:
                continue

            # Find drone with min π_i (i=1..k̄_d) — Proposition 2
            min_drone_idx = 1 + label.pi[1:].index(min(label.pi[1:]))
            min_drone_pi = label.pi[min_drone_idx]
            drone_arrival = min_drone_pi + data.t_0 + data.t_d[label.sigma, w]
            if drone_arrival > data.l[w] + 1e-9:
                continue

            sigma = label.sigma
            new_pi = list(label.pi)
            new_pi[min_drone_idx] = (min_drone_pi + data.t_0
                                     + data.t_d[sigma, w]
                                     + data.ser_d[w]
                                     + data.t_d[w, sigma])

            new_C = label.C + lambda_i.get(w, 0.0)
            new_S = label.S | {w}

            max_new_pi = max(new_pi)
            min_drone_pi_new = min(new_pi[1:])

            new_omega1 = set()
            for wp in label.omega1:
                if wp in new_S:
                    continue
                if new_kappa - data.demand[wp] < -1e-9:
                    continue
                if max_new_pi + data.t_v[sigma, wp] > data.l[wp] + 1e-9:
                    continue
                new_omega1.add(wp)

            new_omega2 = set()
            for wp in label.omega2:
                if wp == w or wp in new_S:
                    continue
                if new_kappa - data.demand[wp] < -1e-9:
                    continue
                if min_drone_pi_new + data.t_0 + data.t_d[sigma, wp] > data.l[wp] + 1e-9:
                    continue
                new_omega2.add(wp)

            new_da = {k: [list(dr) for dr in v] for k, v in label.path_da.items()}
            if sigma not in new_da:
                new_da[sigma] = [[] for _ in range(k_bar_d)]
            new_da[sigma][min_drone_idx - 1] = (
                list(new_da[sigma][min_drone_idx - 1]) + [w]
            )

            new_label = ForwardLabel(
                v=w, sigma=sigma, pi=new_pi, kappa=new_kappa, C=new_C,
                omega1=frozenset(new_omega1), omega2=frozenset(new_omega2),
                S=new_S, path_vn=list(label.path_vn), path_da=new_da
            )

            bucket = all_fwd_labels.setdefault(sigma, [])
            if not _is_forward_dominated(new_label, bucket, k_bar_d):
                all_fwd_labels[sigma] = _prune_forward_dominated(new_label, bucket, k_bar_d)
                all_fwd_labels[sigma].append(new_label)
                unprocessed_fwd.append(new_label)


# ─────────────────────────────────────────────────────────────────────────────
# Backward Label Extension
# ─────────────────────────────────────────────────────────────────────────────

def _extend_backward(data, label, k_bar_d, lambda_i, forbidden_arcs, forced_from, forced_into,
                     all_bwd_labels, unprocessed_bwd, half_time):
    """
    Extend a backward label.
    For k̄_d ≥ 1: extend while min_i{π_i(L_b), i=1..k̄_d} > l_0 / 2
    For k̄_d = 0: extend while π_0(L_b) > l_0 / 2
    """
    if k_bar_d >= 1:
        cutoff_val = min(label.pi[1:])
    else:
        cutoff_val = label.pi[0]

    if cutoff_val <= half_time:
        return  # Do not extend further

    # Case 1: Extend to vehicle node w ∈ Ω1
    for w in label.omega1:
        if w in label.S:
            continue
        if (w, label.sigma) in forbidden_arcs:
            continue

        new_kappa = label.kappa - data.demand[w]
        if new_kappa < -1e-9:
            continue

        # Feasibility: min_i{π_i} - t_v[w, σ] - s_v[w] ≥ 0
        min_pi = min(label.pi)
        available = min_pi - data.t_v[w, label.sigma] - data.ser_v[w]
        if available < -1e-9:
            continue

        # Time w deadline: vehicle must arrive at w before l_w
        # In backward labeling, π_0 is the latest vehicle arrival at σ
        # The vehicle arrives at w from w's predecessor, and then travels to σ
        # The vehicle's latest arrival at w = min_pi - t_v[w, σ]
        vehicle_arrive_w = min_pi - data.t_v[w, label.sigma]
        if vehicle_arrive_w > data.l[w] + 1e-9:
            # Actually the latest arrival must be ≤ l_w (deadline), we compute:
            # π_0(L_b) for new label = min{min_j{π_j(L'_b)} - t_v[w,σ] - s_v[w], l_w}
            pass  # handled by formula below

        sigma_new = w

        # Build new π (Case 1 of backward)
        new_pi = [0.0] * (k_bar_d + 1)
        for i in range(1, k_bar_d + 1):
            new_pi[i] = min_pi - data.t_v[w, label.sigma]  # drones available at w
        new_pi[0] = min(min_pi - data.t_v[w, label.sigma] - data.ser_v[w],
                        data.l[w])

        if new_pi[0] < -1e-9:
            continue

        # ρ (Case 1)
        new_rho = [0.0] * (k_bar_d + 1)
        new_rho[0] = data.ser_v[w]
        for i in range(1, k_bar_d + 1):
            new_rho[i] = 0.0

        new_kappa_new = label.kappa - data.demand[w]
        new_C1 = label.C1 + lambda_i.get(w, 0.0)
        # C2: time duration from σ_new (=w) to end depot
        # = max_i{ρ_i(L'_b)} + t_v[w, σ(L'_b)] + C2(L'_b)
        new_C2 = max(label.rho) + data.t_v[w, label.sigma] + label.C2
        new_S = label.S | {w}

        # Update Ω1
        new_omega1 = set()
        for wp in label.omega1:
            if wp in new_S:
                continue
            if new_kappa_new - data.demand[wp] < -1e-9:
                continue
            # Check feasibility: min_i{new_pi[i]} - t_v[wp, w] - s_v[wp] ≥ 0
            if min(new_pi) - data.t_v[wp, w] - data.ser_v[wp] < -1e-9:
                continue
            new_omega1.add(wp)

        # Update Ω2
        new_omega2 = set()
        for wp in label.omega2:
            if wp in new_S:
                continue
            if new_kappa_new - data.demand[wp] < -1e-9:
                continue
            new_omega2.add(wp)

        new_da = {k: [list(dr) for dr in v] for k, v in label.path_da.items()}

        new_label = BackwardLabel(
            v=w, sigma=w, pi=new_pi, rho=new_rho,
            kappa=new_kappa_new, C1=new_C1, C2=new_C2,
            omega1=frozenset(new_omega1), omega2=frozenset(new_omega2),
            S=new_S, path_vn=[w] + label.path_vn, path_da=new_da
        )

        bucket = all_bwd_labels.setdefault(w, [])
        if not _is_backward_dominated(new_label, bucket, k_bar_d):
            all_bwd_labels[w] = _prune_backward_dominated(new_label, bucket, k_bar_d)
            all_bwd_labels[w].append(new_label)
            unprocessed_bwd.append(new_label)

    # Case 2: Extend to drone node w ∈ Ω2
    if k_bar_d > 0:
        for w in label.omega2:
            if w in label.S:
                continue
            if data.E[label.sigma, w] != 1:
                continue

            new_kappa = label.kappa - data.demand[w]
            if new_kappa < -1e-9:
                continue

            # Feasibility: ∃ i s.t. min{π_i - t_d[w,σ] - s_d[w], l_w} - t_d[σ,w] - t_0 ≥ 0
            best_drone = None
            best_drone_pi_lb = -float('inf')
            for i in range(1, k_bar_d + 1):
                val = min(label.pi[i] - data.t_d[w, label.sigma] - data.ser_d[w],
                          data.l[w]) - data.t_d[label.sigma, w] - data.t_0
                if val >= -1e-9 and val > best_drone_pi_lb:
                    best_drone_pi_lb = val
                    best_drone = i

            if best_drone is None:
                continue

            sigma = label.sigma

            # Build new π (Case 2 of backward)
            new_pi = list(label.pi)
            new_pi[best_drone] = min(
                label.pi[best_drone] - data.t_d[w, sigma] - data.ser_d[w],
                data.l[w]
            ) - data.t_d[sigma, w] - data.t_0

            # Build new ρ (Case 2)
            new_rho = list(label.rho)
            new_rho[best_drone] = (label.rho[best_drone]
                                   + data.t_d[sigma, w] + data.ser_d[w]
                                   + data.t_d[w, sigma] + data.t_0)

            new_C1 = label.C1 + lambda_i.get(w, 0.0)
            new_C2 = label.C2  # no change (drone, not vehicle)
            new_S = label.S | {w}

            new_omega1 = set()
            for wp in label.omega1:
                if wp in new_S:
                    continue
                if new_kappa - data.demand[wp] < -1e-9:
                    continue
                if min(new_pi) - data.t_v[wp, sigma] - data.ser_v[wp] < -1e-9:
                    continue
                new_omega1.add(wp)

            new_omega2 = set()
            for wp in label.omega2:
                if wp == w or wp in new_S:
                    continue
                if new_kappa - data.demand[wp] < -1e-9:
                    continue
                new_omega2.add(wp)

            new_da = {k: [list(dr) for dr in v] for k, v in label.path_da.items()}
            if sigma not in new_da:
                new_da[sigma] = [[] for _ in range(k_bar_d)]
            # Prepend w to drone route (backward: new node goes to the front)
            new_da[sigma][best_drone - 1] = (
                [w] + list(new_da[sigma][best_drone - 1])
            )

            new_label = BackwardLabel(
                v=w, sigma=sigma, pi=new_pi, rho=new_rho,
                kappa=new_kappa, C1=new_C1, C2=new_C2,
                omega1=frozenset(new_omega1), omega2=frozenset(new_omega2),
                S=new_S, path_vn=list(label.path_vn), path_da=new_da
            )

            bucket = all_bwd_labels.setdefault(sigma, [])
            if not _is_backward_dominated(new_label, bucket, k_bar_d):
                all_bwd_labels[sigma] = _prune_backward_dominated(new_label, bucket, k_bar_d)
                all_bwd_labels[sigma].append(new_label)
                unprocessed_bwd.append(new_label)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1: Pairing Algorithm for Drone Routes
# ─────────────────────────────────────────────────────────────────────────────

def _pair_drone_routes(Lf_pi, Lb_pi_list, Lb_rho_list, k_bar_d):
    """
    Algorithm 1: Pair forward and backward drone routes.
    Input:
      Lf_pi: list of π_i(L_f) for i=1..k̄_d (forward drone return times)
      Lb_pi_list: list of π_i(L_b) for i=1..k̄_d (backward drone departure times)
      Lb_rho_list: list of ρ_i(L_b) for i=1..k̄_d (backward drone durations)
    Returns:
      list of (fwd_idx, bwd_idx) pairs, or None if infeasible
    
    Sort F descending by π_i(L_f), sort B ascending by ρ_i(L_b).
    For each f in F, find first b in B such that π_i(L_f) ≤ π_i(L_b), pair them.
    """
    F = sorted(range(k_bar_d), key=lambda i: Lf_pi[i], reverse=True)
    B = sorted(range(k_bar_d), key=lambda i: Lb_rho_list[i])

    pairs = []
    B_remaining = list(B)

    for fi in F:
        pi_f = Lf_pi[fi]
        paired = False
        for j, bi in enumerate(B_remaining):
            pi_b = Lb_pi_list[bi]
            if pi_f <= pi_b + 1e-9:
                pairs.append((fi, bi))
                B_remaining.pop(j)
                paired = True
                break
        if not paired:
            return None  # Cannot pair — infeasible

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Label Joining (§4.1.3)
# ─────────────────────────────────────────────────────────────────────────────

def _join_labels(data, Lf, Lb, k_bar_d, lambda_i, lambda_v0, lambda_d0):
    """
    Try to join forward label Lf and backward label Lb into a complete route.
    Returns Route or None.
    
    Conditions (44-47):
      (44) σ(Lf) = σ(Lb)
      (45) π_i(Lf) ≤ π_i(Lb) for i=0..k̄_d
      (46) κ(Lf) + κ(Lb) + q[σ] + q1_d·k̄_d ≥ Q
      (47) S(Lf) ∩ S(Lb) = ∅
    
    Route duration: c_r = C2(Lb) + max{max_i{π_i(Lf) + ρ_i(Lb)}, π_0(Lf)}
    Reduced cost: c̄_r = c_r - C(Lf) - C1(Lb) - λ_v0 - k̄_d·λ_d0
    """
    # (44) same sigma
    if Lf.sigma != Lb.sigma:
        return None

    # (45) π_i(Lf) ≤ π_i(Lb) for i=0..k̄_d
    for i in range(k_bar_d + 1):
        if Lf.pi[i] > Lb.pi[i] + 1e-9:
            return None

    # (46) capacity: κ_f + κ_b + q[σ] + q1_d·k̄_d ≥ Q
    sigma = Lf.sigma
    cap_used = (data.Q - Lf.kappa) + (data.Q - Lb.kappa)
    # cap_used = total demand served by both halves
    # The depot has q[0] = 0, vehicle equip is q1_d * k_bar_d
    # Condition: κ(Lf) + κ(Lb) + q[σ] + q1_d·k̄_d ≥ Q
    cap_check = Lf.kappa + Lb.kappa + data.demand.get(sigma, 0) + data.q1_d * k_bar_d
    if cap_check < data.Q - 1e-9:
        return None

    # (47) elementarity: S(Lf) ∩ S(Lb) = ∅  (excluding depot 0 — always in both)
    S_customers_f = Lf.S - {0, sigma}
    S_customers_b = Lb.S - {0, sigma}
    if S_customers_f & S_customers_b:
        return None

    # Compute route duration
    if k_bar_d == 0:
        c_r = Lb.C2 + Lf.pi[0]
    else:
        # Use Algorithm 1 to pair drone routes
        Lf_pi_drones = Lf.pi[1:]   # length k_bar_d
        Lb_pi_drones = Lb.pi[1:]   # length k_bar_d
        Lb_rho_drones = Lb.rho[1:] # length k_bar_d

        pairs = _pair_drone_routes(Lf_pi_drones, Lb_pi_drones, Lb_rho_drones, k_bar_d)
        if pairs is None:
            return None

        # Compute makespan contributions from each pair
        max_combo = Lf.pi[0]
        for fi, bi in pairs:
            max_combo = max(max_combo, Lf.pi[1 + fi] + Lb.rho[1 + bi])
        c_r = Lb.C2 + max_combo

    # Build path: forward ends at sigma, backward starts at sigma → skip one sigma
    # Lf.path_vn = [0, ..., sigma], Lb.path_vn = [sigma, ..., 0]
    vn = Lf.path_vn + Lb.path_vn[1:]
    # Merge drone assignments from both halves
    da = {}
    for k, v in Lf.path_da.items():
        da[k] = [list(dr) for dr in v]
    for k, v in Lb.path_da.items():
        if k not in da:
            da[k] = [list(dr) for dr in v]
        else:
            # Merge drone routes at sigma: pair and concatenate
            if k_bar_d > 0 and pairs is not None and k == sigma:
                Lf_pi_drones = Lf.pi[1:]
                Lb_pi_drones = Lb.pi[1:]
                Lb_rho_drones = Lb.rho[1:]
                for fi, bi in pairs:
                    da[sigma][fi] = list(da[sigma][fi]) + list(v[bi])
            else:
                da[k] = [list(dr) for dr in v]

    # Clean up empty drone trips and compute actual num_drones
    actual_nd = 0
    cleaned_da = {}
    for hub, trips in da.items():
        non_empty = [t for t in trips if len(t) > 0]
        if non_empty:
            cleaned_da[hub] = non_empty
            actual_nd = max(actual_nd, len(non_empty))
    da = cleaned_da

    # Reduced cost: c̄_r = c_r - C(Lf) - C1(Lb) - λ_v0 - actual_nd·λ_d0
    # Both Lf.C and Lb.C1 include λ[sigma] (the joining vehicle node),
    # so we add it back once to correct the double-counting.
    # Use actual_nd (not k_bar_d) to avoid over-penalizing routes that
    # use fewer drones than the pricing iteration's k̄_d.
    rc = (c_r - Lf.C - Lb.C1
          + lambda_i.get(sigma, 0.0)  # undo one copy of lambda[sigma]
          - lambda_v0 - actual_nd * lambda_d0)
    if rc >= -1e-6:
        return None

    route = Route(vn, da, actual_nd, c_r)
    return route


# ─────────────────────────────────────────────────────────────────────────────
# Main Bidirectional Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_pricing_bidirectional(data, lambda_i, lambda_v0, lambda_d0,
                                k_bar_d, forbidden_arcs=None, forced_arcs=None,
                                col_max=10):
    """
    Solve pricing for a single k̄_d value using bidirectional labeling (§4.1.1-4.1.3).
    
    Returns list[Route] with negative reduced cost.
    """
    if forbidden_arcs is None:
        forbidden_arcs = set()
    if forced_arcs is None:
        forced_arcs = set()
    # Note: forced_arcs enforced via RMP only; see labeling.py for rationale.
    forced_from = {}
    forced_into = {}

    half_time = data.l[0] / 2.0  # l_0 / 2

    Z = data.Z
    Z_d_set = frozenset(data.Z_d)

    # ── Initialize forward label at depot ──
    init_kappa = data.Q - data.q1_d * k_bar_d
    init_pi_f = [0.0] * (k_bar_d + 1)  # all zeros at depot
    init_omega1_f = frozenset(
        i for i in Z if init_kappa >= data.demand[i]
    )
    init_omega2_f = Z_d_set if k_bar_d > 0 else frozenset()

    root_fwd = ForwardLabel(
        v=0, sigma=0, pi=init_pi_f, kappa=init_kappa, C=0.0,
        omega1=init_omega1_f, omega2=init_omega2_f,
        S=frozenset([0]), path_vn=[0], path_da={}
    )

    # ── Initialize backward label at depot ──
    # π_0 = l_0 (latest vehicle arrival at depot)
    # π_i = l_0 for i=1..k̄_d (latest drone departure time = l_0)
    # ρ = [0, 0, ...] (zero duration at initialization)
    # κ = Q - q1_d * k̄_d
    init_pi_b = [data.l[0]] * (k_bar_d + 1)
    init_rho_b = [0.0] * (k_bar_d + 1)
    init_omega1_b = frozenset(i for i in Z if init_kappa >= data.demand[i])
    init_omega2_b = Z_d_set if k_bar_d > 0 else frozenset()

    root_bwd = BackwardLabel(
        v=0, sigma=0, pi=init_pi_b, rho=init_rho_b,
        kappa=init_kappa, C1=0.0, C2=0.0,
        omega1=init_omega1_b, omega2=init_omega2_b,
        S=frozenset([0]), path_vn=[0], path_da={}
    )

    # Label collections: keyed by sigma
    all_fwd_labels = {0: [root_fwd]}
    all_bwd_labels = {0: [root_bwd]}
    unprocessed_fwd = [root_fwd]
    unprocessed_bwd = [root_bwd]

    negative_rc_routes = []

    # ── Phase 1: Extend forward and backward labels alternately ──
    while (unprocessed_fwd or unprocessed_bwd) and len(negative_rc_routes) < col_max:
        # Process one forward label
        if unprocessed_fwd:
            Lf = unprocessed_fwd.pop(0)
            _extend_forward(data, Lf, k_bar_d, lambda_i, forbidden_arcs,
                           forced_from, forced_into,
                           all_fwd_labels, unprocessed_fwd, half_time)

        # Process one backward label
        if unprocessed_bwd:
            Lb = unprocessed_bwd.pop(0)
            _extend_backward(data, Lb, k_bar_d, lambda_i, forbidden_arcs,
                            forced_from, forced_into,
                            all_bwd_labels, unprocessed_bwd, half_time)

    # ── Phase 2: Join forward and backward labels ──
    for sigma in set(all_fwd_labels.keys()) & set(all_bwd_labels.keys()):
        if sigma == 0:
            continue  # skip depot
        fwd_bucket = all_fwd_labels[sigma]
        bwd_bucket = all_bwd_labels[sigma]

        for Lf in fwd_bucket:
            for Lb in bwd_bucket:
                if len(negative_rc_routes) >= col_max:
                    break
                route = _join_labels(data, Lf, Lb, k_bar_d,
                                     lambda_i, lambda_v0, lambda_d0)
                if route is not None:
                    negative_rc_routes.append(route)
            if len(negative_rc_routes) >= col_max:
                break

    return negative_rc_routes[:col_max]


def solve_pricing_bidirectional_all(data, lambda_i, lambda_v0, lambda_d0,
                                    forbidden_arcs=None, forced_arcs=None,
                                    col_max=10, existing_sigs=None):
    """Run bidirectional labeling for all k̄_d = 0, 1, ..., Γ.
    
    Only genuinely new (non-duplicate) routes count toward col_max,
    preventing CG degeneracy where k̄_d=0 duplicates block k̄_d>0 exploration.
    """
    if existing_sigs is None:
        existing_sigs = set()

    all_new_routes = []  # only new (non-duplicate) routes

    for k_bar_d in range(data.Gamma + 1):
        if len(all_new_routes) >= col_max:
            break

        remaining = col_max - len(all_new_routes)
        routes = solve_pricing_bidirectional(
            data, lambda_i, lambda_v0, lambda_d0,
            k_bar_d, forbidden_arcs, forced_arcs, remaining
        )

        for r in routes:
            sig = r.signature()
            if sig not in existing_sigs:
                all_new_routes.append(r)
                existing_sigs.add(sig)
                if len(all_new_routes) >= col_max:
                    break

    return all_new_routes


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import CardiffDataLoader
    from initial_columns import generate_initial_columns
    from rmp_solver import RMPSolver

    data = CardiffDataLoader("Cardiff10_01.txt")

    # Solve RMP to get duals
    init_routes = generate_initial_columns(data)
    solver = RMPSolver(data, routes=init_routes)
    solver.build()
    obj = solver.solve_lp()
    print(f"RMP LP obj: {obj:.2f}")

    lambda_i, lambda_v0, lambda_d0 = solver.get_duals()

    print(f"\nTesting bidirectional labeling (l_0/2 = {data.l[0]/2:.1f}):")
    for k_bar_d in range(data.Gamma + 1):
        routes = solve_pricing_bidirectional(
            data, lambda_i, lambda_v0, lambda_d0, k_bar_d, col_max=5
        )
        print(f"  k̄_d={k_bar_d}: found {len(routes)} negative-RC routes")
        for r in routes:
            rc = r.cost - sum(r.get_alpha(c) * lambda_i.get(c, 0) for c in data.Z)
            rc -= lambda_v0 + r.num_drones * lambda_d0
            print(f"    veh={r.vehicle_nodes}, da={r.drone_assignments}, "
                  f"nd={r.num_drones}, cost={r.cost:.2f}, RC={rc:.4f}")
