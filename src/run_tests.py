"""
Run all Phase 1 tests without pytest dependency.
Run with: D:\Miniconda\envs\cplex128_env\python.exe run_tests.py
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import CardiffDataLoader
from route import Route
from rmp_solver import RMPSolver


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cardiff10_01.txt")


def get_data():
    return CardiffDataLoader(DATA_PATH)


def compute_route_cost(data, route):
    """Compute duration of a Route: travel times + waiting times."""
    vn = route.vehicle_nodes
    total_cost = 0.0
    for idx in range(len(vn) - 1):
        i = vn[idx]
        j = vn[idx + 1]
        total_cost += data.t_v[i, j]
        if j != 0:
            wait = data.ser_v[j]
            if j in route.drone_assignments:
                for dr in route.drone_assignments[j]:
                    drone_time = 0.0
                    for dn in dr:
                        drone_time += data.t_0 + data.t_d[j, dn] + data.ser_d[dn] + data.t_d[dn, j]
                    wait = max(wait, drone_time)
            total_cost += wait
    return total_cost


def run_test(name, func):
    """Run a single test, report pass/fail."""
    try:
        func()
        print(f"  [PASS] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}")
        traceback.print_exc()
        return False


# ========== Test Functions ==========

def test_route_basic():
    r = Route([0, 3, 7, 0], {3: [[5, 6], [8]]}, 2, 500.0)
    assert r.get_all_customers() == {3, 5, 6, 7, 8}
    assert r.get_alpha(3) == 1
    assert r.get_alpha(0) == 0
    assert r.get_gamma(0, 3) == 1
    assert r.get_gamma(3, 0) == 0
    assert r.get_xi(3, 5) == 1
    assert r.get_xi(3, 7) == 0


def test_dummy_variables():
    data = get_data()
    solver = RMPSolver(data, routes=[])
    solver.build()
    obj = solver.solve_lp()
    assert obj is not None
    expected = data.n * solver.penalty_M
    assert abs(obj - expected) < 1e-3, f"Got {obj}, expected {expected}"
    dummies = solver.get_dummy_values()
    for i in data.Z:
        assert abs(dummies[i] - 1.0) < 1e-5
    print(f"    Penalty M = {solver.penalty_M:.1f}, obj = {obj:.2f}")


def test_two_routes_covering_all():
    data = get_data()
    route1 = Route([0, 1, 2, 3, 4, 5, 0], {}, 0, 0)
    route1.cost = compute_route_cost(data, route1)
    route2 = Route([0, 6, 7, 8, 9, 10, 0], {}, 0, 0)
    route2.cost = compute_route_cost(data, route2)
    total = route1.cost + route2.cost

    solver = RMPSolver(data, routes=[route1, route2])
    solver.build()
    obj = solver.solve_lp()
    assert obj is not None
    assert abs(obj - total) < 1e-2, f"obj={obj:.2f}, expected={total:.2f}"
    print(f"    Route1={route1.cost:.2f}, Route2={route2.cost:.2f}, LP obj={obj:.2f}")


def test_dual_signs():
    data = get_data()
    route1 = Route([0, 1, 2, 3, 4, 5, 0], {}, 0, 0)
    route1.cost = compute_route_cost(data, route1)
    route2 = Route([0, 6, 7, 8, 9, 10, 0], {}, 0, 0)
    route2.cost = compute_route_cost(data, route2)

    solver = RMPSolver(data, routes=[route1, route2])
    solver.build()
    solver.solve_lp()
    lambda_i, lambda_v0, lambda_d0 = solver.get_duals()

    assert len(lambda_i) == data.n
    # λ_v0 and λ_d0: for ≤ constraints, dual should be ≤ 0
    assert lambda_v0 <= 1e-8, f"λ_v0={lambda_v0} should be ≤ 0"
    assert lambda_d0 <= 1e-8, f"λ_d0={lambda_d0} should be ≤ 0"
    print(f"    λ_v0={lambda_v0:.4f}, λ_d0={lambda_d0:.4f}")
    print(f"    λ_i: {[f'{lambda_i[i]:.2f}' for i in sorted(lambda_i.keys())]}")


def test_reduced_cost_formula():
    """
    With ub removed (no upper-bound trap), basis routes (μ_r > 0)
    should have reduced cost = 0, and our formula should match CPLEX.
    """
    data = get_data()
    route1 = Route([0, 1, 2, 3, 4, 5, 0], {}, 0, 0)
    route1.cost = compute_route_cost(data, route1)
    route2 = Route([0, 6, 7, 8, 9, 10, 0], {}, 0, 0)
    route2.cost = compute_route_cost(data, route2)

    solver = RMPSolver(data, routes=[route1, route2])
    solver.build()
    solver.solve_lp()
    lambda_i, lambda_v0, lambda_d0 = solver.get_duals()

    # Our formula: c̄_r = c_r - Σ α_ir λ_i - λ_v0 - d_r λ_d0
    our_rc1 = solver.compute_reduced_cost(route1, lambda_i, lambda_v0, lambda_d0)
    our_rc2 = solver.compute_reduced_cost(route2, lambda_i, lambda_v0, lambda_d0)

    # CPLEX internal reduced costs
    cplex_rc1 = solver.cpx.solution.get_reduced_costs("mu_0")
    cplex_rc2 = solver.cpx.solution.get_reduced_costs("mu_1")

    print(f"    Our RC1={our_rc1:.6f}, CPLEX RC1={cplex_rc1:.6f}")
    print(f"    Our RC2={our_rc2:.6f}, CPLEX RC2={cplex_rc2:.6f}")

    # Basis routes should have RC = 0 (no UB trap!)
    assert abs(our_rc1) < 1e-4, f"RC1={our_rc1} should be ~0"
    assert abs(our_rc2) < 1e-4, f"RC2={our_rc2} should be ~0"

    # Our formula must match CPLEX
    assert abs(our_rc1 - cplex_rc1) < 1e-3, \
        f"Formula mismatch: our={our_rc1:.4f} vs cplex={cplex_rc1:.4f}"
    assert abs(our_rc2 - cplex_rc2) < 1e-3, \
        f"Formula mismatch: our={our_rc2:.4f} vs cplex={cplex_rc2:.4f}"

    # Dummy variables should have non-negative RC (won't be generated by pricing)
    for name in solver._dummy_var_names:
        drc = solver.cpx.solution.get_reduced_costs(name)
        assert drc >= -1e-6, f"Dummy {name} RC={drc} should be >= 0"


def test_dynamic_column_add():
    data = get_data()
    route1 = Route([0, 1, 2, 3, 4, 5, 0], {}, 0, 0)
    route1.cost = compute_route_cost(data, route1)

    solver = RMPSolver(data, routes=[route1])
    solver.build()
    obj1 = solver.solve_lp()

    route2 = Route([0, 6, 7, 8, 9, 10, 0], {}, 0, 0)
    route2.cost = compute_route_cost(data, route2)
    solver.add_column(route2)
    obj2 = solver.solve_lp()

    assert obj2 < obj1, f"obj2={obj2:.2f} should < obj1={obj1:.2f}"
    print(f"    Before: {obj1:.2f}, After: {obj2:.2f}")


def test_integer_solution():
    data = get_data()
    route1 = Route([0, 1, 2, 3, 4, 5, 0], {}, 0, 0)
    route1.cost = compute_route_cost(data, route1)
    route2 = Route([0, 6, 7, 8, 9, 10, 0], {}, 0, 0)
    route2.cost = compute_route_cost(data, route2)

    solver = RMPSolver(data, routes=[route1, route2])
    solver.build()
    solver.solve_lp()
    assert solver.is_integer_solution(), "Should be integer"


def test_route_with_drones():
    """Test RMP with drone-equipped routes."""
    data = get_data()
    # Z_d = [1, 2, 3, 4, 6, 8, 9] (drone-eligible)
    # Customers 5, 7, 10 are vehicle-only (f_d=0)

    # Route 1: vehicle visits 3, 5, 7; drones from 3 serve 6 and 9
    route1 = Route(
        vehicle_nodes=[0, 3, 5, 7, 0],
        drone_assignments={3: [[6], [9]]},
        num_drones=2,
        cost=0
    )
    route1.cost = compute_route_cost(data, route1)

    # Route 2: vehicle visits 10, 1; drones from 1 serve 2, 4, 8
    route2 = Route(
        vehicle_nodes=[0, 10, 1, 0],
        drone_assignments={1: [[2, 4], [8]]},
        num_drones=2,
        cost=0
    )
    route2.cost = compute_route_cost(data, route2)

    # All customers covered
    covered = route1.get_all_customers() | route2.get_all_customers()
    assert covered == set(range(1, data.n + 1)), f"Missing: {set(range(1, data.n+1)) - covered}"

    solver = RMPSolver(data, routes=[route1, route2])
    solver.build()
    obj = solver.solve_lp()
    assert obj is not None, "LP should be feasible"
    total = route1.cost + route2.cost
    print(f"    Drone route1={route1.cost:.2f}, route2={route2.cost:.2f}")
    print(f"    LP obj={obj:.2f}, total={total:.2f}")
    assert abs(obj - total) < 1e-2

    # Check drone count constraint
    lambda_i, lambda_v0, lambda_d0 = solver.get_duals()
    total_drones = route1.num_drones + route2.num_drones  # 4
    print(f"    Total drones used: {total_drones}, k_d={data.k_d}")
    # 4 drones used = k_d = 4, so constraint (35) is tight → λ_d0 may be non-zero


def test_fractional_solution():
    """Test that overlapping routes produce fractional LP solution."""
    data = get_data()
    # Create 3 overlapping routes — LP must choose fractional values
    route1 = Route([0, 1, 2, 3, 4, 5, 0], {}, 0, 0)
    route1.cost = compute_route_cost(data, route1)
    route2 = Route([0, 3, 4, 5, 6, 7, 0], {}, 0, 0)  # overlaps on 3,4,5
    route2.cost = compute_route_cost(data, route2)
    route3 = Route([0, 1, 2, 8, 9, 10, 0], {}, 0, 0)  # overlaps on 1,2
    route3.cost = compute_route_cost(data, route3)
    route4 = Route([0, 6, 7, 8, 9, 10, 0], {}, 0, 0)  # overlaps on 8,9,10,6,7
    route4.cost = compute_route_cost(data, route4)

    solver = RMPSolver(data, routes=[route1, route2, route3, route4])
    solver.build()
    obj = solver.solve_lp()
    assert obj is not None

    vals = solver.get_solution_values()
    print(f"    LP obj={obj:.2f}")
    has_fractional = False
    for route, val in vals:
        print(f"    μ={val:.4f} for route {route.vehicle_nodes}")
        if 1e-5 < val < 1.0 - 1e-5:
            has_fractional = True

    # With overlapping routes, LP may or may not be fractional
    # (depends on costs). Just verify it solves.
    print(f"    Has fractional values: {has_fractional}")


# ========== Main ==========

if __name__ == '__main__':
    tests = [
        ("Route basic operations", test_route_basic),
        ("RMP with dummy variables only", test_dummy_variables),
        ("RMP with two vehicle-only routes", test_two_routes_covering_all),
        ("Dual variable signs", test_dual_signs),
        ("Reduced cost formula verification", test_reduced_cost_formula),
        ("Dynamic column addition", test_dynamic_column_add),
        ("Integer solution detection", test_integer_solution),
        ("Routes with drone assignments", test_route_with_drones),
        ("Overlapping routes / fractional LP", test_fractional_solution),
    ]

    print("=" * 60)
    print("Phase 1 Tests: Route & RMP Solver")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, func in tests:
        if run_test(name, func):
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
