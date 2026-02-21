"""
Branch-and-Bound tree for the B&P algorithm (§4.3).
Implements 4-level hierarchical branching with best-first search.
"""

import heapq
import time

from data_loader import CardiffDataLoader
from route import Route
from rmp_solver import RMPSolver
from initial_columns import generate_initial_columns
from column_generation import run_column_generation


class BBNode:
    """A node in the Branch-and-Bound tree."""
    __slots__ = ['id', 'lb', 'extra_constraints', 'forbidden_arcs',
                 'forced_arcs', 'parent_id', 'depth', 'initial_routes']

    _counter = 0

    def __init__(self, extra_constraints=None, forbidden_arcs=None,
                 forced_arcs=None, parent_id=None, depth=0,
                 initial_routes=None):
        BBNode._counter += 1
        self.id = BBNode._counter
        self.lb = -float('inf')
        self.extra_constraints = list(extra_constraints) if extra_constraints else []
        self.forbidden_arcs = set(forbidden_arcs) if forbidden_arcs else set()
        self.forced_arcs = set(forced_arcs) if forced_arcs else set()
        self.parent_id = parent_id
        self.depth = depth
        self.initial_routes = initial_routes

    def __lt__(self, other):
        """For min-heap: node with lower lb is explored first (best-first)."""
        return self.lb < other.lb


def find_branching_decision(solver, data):
    """
    Determine the branching variable using 4-level hierarchy (§4.3).

    Returns:
        dict or None: branching info, or None if solution is integer.
    """
    vals = solver.get_solution_values()

    # Level 1: Number of vehicles Σ μ_r
    total_vehicles = sum(mu for _, mu in vals if mu > 1e-8)
    frac_v = total_vehicles - int(total_vehicles + 0.5)
    if abs(frac_v) > 1e-5 and total_vehicles != int(total_vehicles):
        floor_v = int(total_vehicles)
        return {
            'level': 1,
            'description': f'vehicles: {total_vehicles:.4f}',
            'branches': [
                {'type': 'vehicle_count', 'sense': 'L', 'rhs': float(floor_v)},
                {'type': 'vehicle_count', 'sense': 'G', 'rhs': float(floor_v + 1)},
            ]
        }

    # Level 2: Number of drones Σ d_r μ_r
    total_drones = sum(r.num_drones * mu for r, mu in vals if mu > 1e-8)
    frac_d = total_drones - int(total_drones + 0.5)
    if abs(frac_d) > 1e-5 and total_drones != int(total_drones):
        floor_d = int(total_drones)
        return {
            'level': 2,
            'description': f'drones: {total_drones:.4f}',
            'branches': [
                {'type': 'drone_count', 'sense': 'L', 'rhs': float(floor_d)},
                {'type': 'drone_count', 'sense': 'G', 'rhs': float(floor_d + 1)},
            ]
        }

    # Level 3: Vehicle arc flow Σ γ_ijr μ_r closest to 0.5
    best_arc = None
    best_frac_dist = float('inf')
    for i in data.V:
        for j in data.V:
            if i == j:
                continue
            flow = sum(r.get_gamma(i, j) * mu for r, mu in vals if mu > 1e-8)
            if 1e-5 < flow < 1.0 - 1e-5:
                dist = abs(flow - 0.5)
                if dist < best_frac_dist:
                    best_frac_dist = dist
                    best_arc = (i, j, flow)

    if best_arc is not None:
        i, j, flow = best_arc
        return {
            'level': 3,
            'description': f'arc ({i},{j}): {flow:.4f}',
            'branches': [
                # x_ij = 0 → forbid arc (i,j) in pricing
                {'type': 'arc_flow', 'sense': 'L', 'rhs': 0.0, 'arc': (i, j),
                 'forbid_arc': (i, j)},
                # x_ij = 1 → force arc (i,j)
                {'type': 'arc_flow', 'sense': 'G', 'rhs': 1.0, 'arc': (i, j),
                 'force_arc': (i, j)},
            ]
        }

    # Level 4: Drone dispatch flow Σ ξ_ijr μ_r closest to 0.5
    best_dispatch = None
    best_frac_dist = float('inf')
    for i in data.Z:
        for j in data.Z_d:
            if i == j:
                continue
            flow = sum(r.get_xi(i, j) * mu for r, mu in vals if mu > 1e-8)
            if 1e-5 < flow < 1.0 - 1e-5:
                dist = abs(flow - 0.5)
                if dist < best_frac_dist:
                    best_frac_dist = dist
                    best_dispatch = (i, j, flow)

    if best_dispatch is not None:
        i, j, flow = best_dispatch
        return {
            'level': 4,
            'description': f'drone ({i},{j}): {flow:.4f}',
            'branches': [
                {'type': 'drone_dispatch', 'sense': 'L', 'rhs': 0.0, 'arc': (i, j),
                 'forbid_arc': (i, j)},
                {'type': 'drone_dispatch', 'sense': 'G', 'rhs': 1.0, 'arc': (i, j),
                 'force_arc': (i, j)},
            ]
        }

    # No fractional variables found — solution is integer
    return None


def solve_bnp(data, verbose=True, time_limit=300):
    """
    Solve the 2E-VRP-D using Branch-and-Price.

    Args:
        data: CardiffDataLoader instance
        verbose: print progress
        time_limit: max time in seconds

    Returns:
        tuple: (best_ub, best_solution, stats)
    """
    start_time = time.time()

    # Generate initial columns (shared across all nodes)
    init_routes = generate_initial_columns(data)

    # Initialize
    best_ub = float('inf')
    best_solution = None
    nodes_explored = 0

    # Create root node
    root = BBNode(initial_routes=init_routes)
    heap = [root]  # min-heap by lower bound

    while heap:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            if verbose:
                print(f"[B&P] Time limit reached ({elapsed:.1f}s)")
            break

        # Pop node with lowest lower bound (best-first)
        node = heapq.heappop(heap)

        # Prune: if node's lb >= best_ub, skip
        if node.lb >= best_ub - 1e-6:
            continue

        nodes_explored += 1

        if verbose:
            print(f"\n[B&P] Node {node.id} (depth={node.depth}, "
                  f"lb={node.lb:.4f}, UB={best_ub:.4f})")

        # Solve CG at this node
        lp_bound, solver, is_integer = run_column_generation(
            data,
            initial_routes=node.initial_routes,
            forbidden_arcs=node.forbidden_arcs,
            forced_arcs=node.forced_arcs,
            extra_constraints=node.extra_constraints,
            verbose=verbose
        )

        if lp_bound is None:
            if verbose:
                print(f"[B&P]   Node {node.id}: infeasible")
            continue

        node.lb = lp_bound

        # Prune by bound
        if lp_bound >= best_ub - 1e-6:
            if verbose:
                print(f"[B&P]   Node {node.id}: pruned (lb={lp_bound:.4f} >= UB={best_ub:.4f})")
            continue

        # Check if integer
        if is_integer:
            if lp_bound < best_ub:
                best_ub = lp_bound
                best_solution = solver.get_solution_values()
                if verbose:
                    print(f"[B&P]   *** New UB = {best_ub:.4f} ***")
            continue

        # Find branching decision
        branch = find_branching_decision(solver, data)
        if branch is None:
            # Integer solution (no fractional vars)
            if lp_bound < best_ub:
                best_ub = lp_bound
                best_solution = solver.get_solution_values()
                if verbose:
                    print(f"[B&P]   *** New UB = {best_ub:.4f} (integer) ***")
            continue

        if verbose:
            print(f"[B&P]   Branching on level {branch['level']}: "
                  f"{branch['description']}")

        # Create child nodes
        # Collect existing columns that are compatible with each branch
        existing_routes = list(solver.routes)

        for bi, binfo in enumerate(branch['branches']):
            child = BBNode(
                extra_constraints=node.extra_constraints + [binfo],
                forbidden_arcs=set(node.forbidden_arcs),
                forced_arcs=set(node.forced_arcs),
                parent_id=node.id,
                depth=node.depth + 1
            )

            # Add forbidden/forced arcs for arc-flow branching
            if 'forbid_arc' in binfo:
                child.forbidden_arcs.add(binfo['forbid_arc'])
            if 'force_arc' in binfo:
                child.forced_arcs.add(binfo['force_arc'])

            # Filter existing columns: remove routes that violate new constraints
            child_routes = []
            for r in existing_routes:
                if _route_satisfies_constraint(r, binfo):
                    child_routes.append(r)
            child.initial_routes = child_routes if child_routes else init_routes

            child.lb = lp_bound  # inherit parent's lower bound
            heapq.heappush(heap, child)

    elapsed = time.time() - start_time
    gap = (best_ub - root.lb) / best_ub * 100 if best_ub < float('inf') else float('inf')

    stats = {
        'ub': best_ub,
        'lb': root.lb,
        'gap': gap,
        'nodes': nodes_explored,
        'time': elapsed,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"B&P Result:")
        print(f"  UB = {best_ub:.4f}")
        print(f"  LB = {root.lb:.4f}")
        print(f"  Gap = {gap:.2f}%")
        print(f"  Nodes explored = {nodes_explored}")
        print(f"  Time = {elapsed:.2f}s")
        if best_solution:
            print(f"\nOptimal routes:")
            for route, mu in best_solution:
                if mu > 0.5:
                    print(f"  μ={mu:.4f}, cost={route.cost:.2f}, "
                          f"veh={route.vehicle_nodes}, "
                          f"drones={route.drone_assignments}")
        print(f"{'='*60}")

    return best_ub, best_solution, stats


def _route_satisfies_constraint(route, constraint):
    """Check if a route satisfies a branching constraint."""
    ctype = constraint['type']
    sense = constraint['sense']
    rhs = constraint['rhs']

    if ctype == 'vehicle_count':
        # This is about aggregate Σμ_r, not individual route
        return True
    elif ctype == 'drone_count':
        return True
    elif ctype == 'arc_flow':
        arc = constraint['arc']
        val = route.get_gamma(arc[0], arc[1])
        if sense == 'L' and val > rhs + 1e-6:
            return False
        if sense == 'G' and val < rhs - 1e-6:
            return False
        return True
    elif ctype == 'drone_dispatch':
        arc = constraint['arc']
        val = route.get_xi(arc[0], arc[1])
        if sense == 'L' and val > rhs + 1e-6:
            return False
        if sense == 'G' and val < rhs - 1e-6:
            return False
        return True
    return True


if __name__ == '__main__':
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "Cardiff15_01.txt"
    data = CardiffDataLoader(data_file)
    print(f"Solving {data_file}: n={data.n}, k_v={data.k_v}, k_d={data.k_d}, Γ={data.Gamma}")
    print(f"Expected optimal: ")
    print()

    best_ub, best_solution, stats = solve_bnp(data, time_limit=300)
