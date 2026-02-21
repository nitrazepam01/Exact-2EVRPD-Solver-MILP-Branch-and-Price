"""
Column Generation Manager for the B&P algorithm.
Implements the CG loop: RMP → Pricing (TS heuristic first, then exact bidirectional) → Add columns.

Per §4.1.4: TS is applied first. If ≥ col_max columns found, skip exact labeling.
Otherwise, run exact bidirectional labeling.
"""

import sys
import time

from data_loader import CardiffDataLoader
from route import Route
from rmp_solver import RMPSolver
from initial_columns import generate_initial_columns
from bidirectional_labeling import solve_pricing_bidirectional_all
from tabu_search import tabu_search_pricing
from labeling import solve_pricing_all as solve_pricing_forward_only


def _route_signature(route):
    """Create a hashable signature for a route to detect duplicates."""
    vn = tuple(route.vehicle_nodes)
    da_items = []
    for k in sorted(route.drone_assignments.keys()):
        for dr in route.drone_assignments[k]:
            da_items.append((k, tuple(dr)))
    return (vn, tuple(da_items), route.num_drones)


def run_column_generation(data, initial_routes=None, forbidden_arcs=None,
                          forced_arcs=None, extra_constraints=None, col_max=10,
                          verbose=True, max_iterations=500,
                          use_ts=True, use_bidir=True):
    """
    Run the column generation loop at a single B&B node.

    Per §4.1.4: Try Tabu Search first. If col_max negative-RC routes found,
    skip exact labeling. Otherwise run exact bidirectional labeling.

    Args:
        data: CardiffDataLoader instance
        initial_routes: list of Route objects (if None, uses cheapest insertion)
        forbidden_arcs: set of (i,j) forbidden arcs from branching
        extra_constraints: list of branching constraint dicts for RMP
        col_max: max columns to add per iteration
        verbose: print progress
        max_iterations: max CG iterations to prevent infinite loops
        use_ts: use Tabu Search heuristic pricing (§4.1.4)
        use_bidir: use bidirectional labeling (§4.1.1-4.1.3)

    Returns:
        tuple: (lp_bound, rmp_solver, is_integer)
    """
    if forbidden_arcs is None:
        forbidden_arcs = set()
    if forced_arcs is None:
        forced_arcs = set()
    if extra_constraints is None:
        extra_constraints = []

    # Step 1: Generate initial columns if needed
    if initial_routes is None:
        initial_routes = generate_initial_columns(data)
        if verbose:
            covered = set()
            for r in initial_routes:
                covered |= r.get_all_customers()
            missing = set(range(1, data.n + 1)) - covered
            print(f"[CG] Initial columns: {len(initial_routes)}, "
                  f"cost={sum(r.cost for r in initial_routes):.2f}, "
                  f"missing={missing if missing else 'none'}")

    # Step 2: Build RMP
    solver = RMPSolver(data, routes=initial_routes,
                       extra_constraints=extra_constraints)
    solver.build()

    # Track existing column signatures for deduplication
    existing_sigs = set()
    for r in initial_routes:
        existing_sigs.add(_route_signature(r))

    iteration = 0
    start_time = time.time()
    obj = None

    while iteration < max_iterations:
        iteration += 1

        # Step 3: Solve RMP
        obj = solver.solve_lp()
        if obj is None:
            if verbose:
                print(f"[CG] Iter {iteration}: RMP infeasible!")
            return None, solver, False

        # Get dual variables
        lambda_i, lambda_v0, lambda_d0 = solver.get_duals()

        if verbose:
            active_dummies = sum(1 for v in solver.get_dummy_values().values() if v > 1e-6)
            print(f"[CG] Iter {iteration}: LP obj = {obj:.4f}, "
                  f"cols = {len(solver.routes)}, "
                  f"active dummies = {active_dummies}")

        # Step 4: Heuristic pricing — Tabu Search (§4.1.4)
        new_routes = []
        if use_ts and iteration > 1:  # TS needs existing solution values
            ts_routes = tabu_search_pricing(
                data, solver, lambda_i, lambda_v0, lambda_d0,
                col_max=col_max, max_iter=100,
                forbidden_arcs=forbidden_arcs, forced_arcs=forced_arcs
            )
            new_routes = ts_routes
            if verbose and ts_routes:
                pass  # TS found routes, will be logged below

        # Step 5: Exact pricing — bidirectional labeling (only if TS found < col_max)
        if len(new_routes) < col_max and use_bidir:
            remaining = col_max - len(new_routes)
            exact_routes = solve_pricing_bidirectional_all(
                data, lambda_i, lambda_v0, lambda_d0,
                forbidden_arcs=forbidden_arcs, forced_arcs=forced_arcs,
                col_max=remaining
            )
            new_routes.extend(exact_routes)

        # Step 5b: Forward-only exact labeling as supplement for full coverage
        # (Bidirectional with half-time cutoff may miss asymmetric routes;
        #  forward-only guarantees optimality — paper §5.2.3 uses both)
        if len(new_routes) < col_max:
            remaining = col_max - len(new_routes)
            fwd_routes = solve_pricing_forward_only(
                data, lambda_i, lambda_v0, lambda_d0,
                forbidden_arcs=forbidden_arcs, forced_arcs=forced_arcs,
                col_max=remaining
            )
            new_routes.extend(fwd_routes)

        if not new_routes:
            # No negative reduced cost columns → CG converged
            elapsed = time.time() - start_time
            if verbose:
                print(f"[CG] Converged in {iteration} iterations, "
                      f"{elapsed:.2f}s, LRP = {obj:.4f}")
            is_int = solver.is_integer_solution()
            return obj, solver, is_int

        # Filter out duplicate columns
        unique_routes = []
        for r in new_routes:
            sig = _route_signature(r)
            if sig not in existing_sigs:
                existing_sigs.add(sig)
                unique_routes.append(r)

        if not unique_routes:
            # All columns are duplicates → converged
            elapsed = time.time() - start_time
            if verbose:
                print(f"[CG] All new columns are duplicates, converged in "
                      f"{iteration} iterations, {elapsed:.2f}s, LRP = {obj:.4f}")
            is_int = solver.is_integer_solution()
            return obj, solver, is_int

        # Step 6: Add new columns to RMP
        if verbose:
            best_rc = min(
                solver.compute_reduced_cost(r, lambda_i, lambda_v0, lambda_d0)
                for r in unique_routes
            )
            print(f"[CG]   Added {len(unique_routes)} columns, "
                  f"best RC = {best_rc:.4f}")

        for r in unique_routes:
            solver.add_column(r)

    # Hit iteration limit
    elapsed = time.time() - start_time
    if verbose:
        print(f"[CG] Hit iteration limit ({max_iterations}), "
              f"{elapsed:.2f}s, LRP = {obj:.4f}")
    is_int = solver.is_integer_solution()
    return obj, solver, is_int


if __name__ == '__main__':
    data_file = sys.argv[1] if len(sys.argv) > 1 else "Cardiff10_01.txt"
    data = CardiffDataLoader(data_file)
    print(f"Loaded {data_file}: n={data.n}, k_v={data.k_v}, k_d={data.k_d}, Γ={data.Gamma}")

    lp_bound, solver, is_integer = run_column_generation(data)

    if lp_bound is not None:
        print(f"\n{'='*50}")
        print(f"CG Result: LRP bound = {lp_bound:.4f}")
        print(f"Integer solution: {is_integer}")
        print(f"Total columns: {len(solver.routes)}")

        # Show selected routes
        vals = solver.get_solution_values()
        print("\nSelected routes:")
        for route, mu in vals:
            if mu > 1e-6:
                print(f"  μ={mu:.4f}, cost={route.cost:.2f}, "
                      f"veh={route.vehicle_nodes}, "
                      f"drones={route.drone_assignments}, "
                      f"nd={route.num_drones}")
