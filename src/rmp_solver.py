"""
Restricted Master Problem (RMP) Solver for the B&P algorithm.
Implements the Set Partitioning formulation (eqs 32-36) using CPLEX.
"""

import cplex
import sys


class RMPSolver:
    """
    Solves the LP relaxation of the Set Partitioning model.

    The model:
        min  Σ c_r μ_r                                            (32)
        s.t. Σ α_ir μ_r = 1          ∀ i ∈ Z                     (33)
             Σ μ_r ≤ k_v                                          (34)
             Σ d_r μ_r ≤ k_d                                      (35)
             0 ≤ μ_r ≤ 1             ∀ r  (LP relaxation of 36)

    Also supports:
        - Dummy variables χ_i with penalty M for feasibility
        - Extra branching constraints from B&B nodes
    """

    def __init__(self, data, routes=None, extra_constraints=None):
        """
        Args:
            data: CardiffDataLoader instance
            routes: initial list of Route objects (can be empty)
            extra_constraints: list of dicts for branching constraints, each:
                {'type': 'vehicle_count_ub'|'vehicle_count_lb'|...,
                 'rhs': float, 'sense': 'L'|'G'|'E', ...}
        """
        self.data = data
        self.routes = list(routes) if routes else []
        self.extra_constraints = extra_constraints or []

        # Sensible penalty M: 50 × max possible single-route duration
        # max single-route ≈ l_0 (the planning horizon)
        self.penalty_M = 50.0 * self.data.l[0]

        self.cpx = None
        self._cst_customer_names = []  # constraint names for (33)
        self._cst_vehicle_name = None  # constraint name for (34)
        self._cst_drone_name = None    # constraint name for (35)
        self._extra_cst_names = []     # branching constraint names
        self._route_var_names = []     # μ_r variable names
        self._dummy_var_names = []     # χ_i variable names
        self._branch_dummy_names = []  # artificial vars for branching constraints
        self._built = False

    def build(self):
        """Build the CPLEX model from scratch."""
        self.cpx = cplex.Cplex()
        self.cpx.objective.set_sense(self.cpx.objective.sense.minimize)
        self.cpx.set_log_stream(None)
        self.cpx.set_results_stream(None)
        self.cpx.set_warning_stream(None)
        self.cpx.set_error_stream(None)

        n = self.data.n
        Z = self.data.Z  # customers 1..n

        # --- Add constraints (empty initially, columns fill them) ---

        # (33): Σ α_ir μ_r + χ_i = 1  ∀ i ∈ Z
        self._cst_customer_names = []
        for i in Z:
            name = f"cst_customer_{i}"
            self._cst_customer_names.append(name)
            self.cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair([], [])],
                senses=['E'],
                rhs=[1.0],
                names=[name]
            )

        # (34): Σ μ_r ≤ k_v
        self._cst_vehicle_name = "cst_vehicle_count"
        self.cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair([], [])],
            senses=['L'],
            rhs=[float(self.data.k_v)],
            names=[self._cst_vehicle_name]
        )

        # (35): Σ d_r μ_r ≤ k_d
        self._cst_drone_name = "cst_drone_count"
        self.cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair([], [])],
            senses=['L'],
            rhs=[float(self.data.k_d)],
            names=[self._cst_drone_name]
        )

        # --- Add extra branching constraints ---
        self._extra_cst_names = []
        for idx, ec in enumerate(self.extra_constraints):
            name = f"cst_branch_{idx}"
            self._extra_cst_names.append(name)
            self.cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair([], [])],
                senses=[ec['sense']],
                rhs=[ec['rhs']],
                names=[name]
            )

        # --- Add artificial variables for G/E branching constraints ---
        # Without these, RMP becomes infeasible when no existing columns
        # satisfy a >= constraint, preventing CG from producing duals.
        self._branch_dummy_names = []
        for idx, ec in enumerate(self.extra_constraints):
            if ec['sense'] in ['G', 'E']:
                name = f"branch_dummy_{idx}"
                self._branch_dummy_names.append(name)
                cst_name = self._extra_cst_names[idx]
                self.cpx.variables.add(
                    names=[name],
                    obj=[self.penalty_M],
                    lb=[0.0],
                    columns=[cplex.SparsePair([cst_name], [1.0])]
                )

        # --- Add dummy variables χ_i for each customer ---
        self._dummy_var_names = []
        for i in Z:
            name = f"dummy_{i}"
            self._dummy_var_names.append(name)
            col_coeffs = []
            col_rows = []
            # Coefficient 1.0 in customer constraint (33) for customer i
            cst_idx = Z.index(i)
            col_rows.append(self._cst_customer_names[cst_idx])
            col_coeffs.append(1.0)

            self.cpx.variables.add(
                names=[name],
                obj=[self.penalty_M],
                lb=[0.0],
                columns=[cplex.SparsePair(col_rows, col_coeffs)]
            )

        # --- Add route variables μ_r ---
        self._route_var_names = []
        for route in self.routes:
            self._add_route_variable(route)

        self._built = True

    def _add_route_variable(self, route):
        """Add a single μ_r variable for the given route."""
        Z = self.data.Z
        name = f"mu_{len(self._route_var_names)}"
        self._route_var_names.append(name)

        col_rows = []
        col_coeffs = []

        # (33): α_ir coefficient for each customer
        for i in Z:
            alpha = route.get_alpha(i)
            if alpha > 0:
                cst_idx = Z.index(i)
                col_rows.append(self._cst_customer_names[cst_idx])
                col_coeffs.append(float(alpha))

        # (34): coefficient 1.0 for vehicle count
        col_rows.append(self._cst_vehicle_name)
        col_coeffs.append(1.0)

        # (35): coefficient d_r for drone count
        col_rows.append(self._cst_drone_name)
        col_coeffs.append(float(route.num_drones))

        # Extra branching constraints
        for idx, ec in enumerate(self.extra_constraints):
            coeff = self._get_branching_coeff(route, ec)
            if coeff != 0.0:
                col_rows.append(self._extra_cst_names[idx])
                col_coeffs.append(coeff)

        self.cpx.variables.add(
            names=[name],
            obj=[route.cost],
            lb=[0.0],
            columns=[cplex.SparsePair(col_rows, col_coeffs)]
        )

    def _get_branching_coeff(self, route, ec):
        """Compute the coefficient of route in a branching constraint."""
        btype = ec['type']
        if btype == 'vehicle_count':
            return 1.0
        elif btype == 'drone_count':
            return float(route.num_drones)
        elif btype == 'arc_flow':
            i, j = ec['arc']
            return float(route.get_gamma(i, j))
        elif btype == 'drone_dispatch':
            i, j = ec['arc']
            return float(route.get_xi(i, j))
        return 0.0

    def add_column(self, route):
        """Dynamically add a new column (route) to the model."""
        self.routes.append(route)
        if self._built:
            self._add_route_variable(route)

    def solve_lp(self):
        """
        Solve the LP relaxation.

        Returns:
            float: optimal objective value, or None if infeasible.
        """
        if not self._built:
            self.build()

        try:
            self.cpx.solve()
            status = self.cpx.solution.get_status()
            if status == self.cpx.solution.status.optimal or \
               status == self.cpx.solution.status.optimal_tolerance:
                return self.cpx.solution.get_objective_value()
            else:
                print(f"RMP LP solve status: {status}")
                return None
        except cplex.CplexSolverError as e:
            print(f"CPLEX error: {e}")
            return None

    def get_duals(self):
        """
        Extract dual variables from the solved LP.

        Returns:
            tuple: (lambda_i_dict, lambda_v0, lambda_d0)
                lambda_i_dict: {customer_i: dual_value}  (for constraints 33)
                lambda_v0: dual of constraint (34) + vehicle_count branching duals
                lambda_d0: dual of constraint (35) + drone_count branching duals
        """
        Z = self.data.Z

        # Get all dual values
        lambda_i = {}
        for idx, i in enumerate(Z):
            dual_val = self.cpx.solution.get_dual_values(self._cst_customer_names[idx])
            lambda_i[i] = dual_val

        lambda_v0 = self.cpx.solution.get_dual_values(self._cst_vehicle_name)
        lambda_d0 = self.cpx.solution.get_dual_values(self._cst_drone_name)

        # Accumulate duals from branching constraints into pricing duals.
        # vehicle_count: coeff = 1.0 for all routes → same structure as (34)
        # drone_count:   coeff = d_r for each route → same structure as (35)
        # arc_flow/drone_dispatch: handled by forbidden/forced arcs, no dual needed
        for idx, ec in enumerate(self.extra_constraints):
            cst_name = self._extra_cst_names[idx]
            dual_val = self.cpx.solution.get_dual_values(cst_name)
            if ec['type'] == 'vehicle_count':
                lambda_v0 += dual_val
            elif ec['type'] == 'drone_count':
                lambda_d0 += dual_val

        return lambda_i, lambda_v0, lambda_d0

    def get_solution_values(self):
        """
        Get the solution values of all route variables μ_r.

        Returns:
            list of (Route, float): pairs of (route, μ_r value)
        """
        result = []
        for idx, route in enumerate(self.routes):
            var_name = self._route_var_names[idx]
            val = self.cpx.solution.get_values(var_name)
            result.append((route, val))
        return result

    def get_dummy_values(self):
        """
        Get values of dummy variables.

        Returns:
            dict: {customer_i: dummy_value}
        """
        result = {}
        for idx, i in enumerate(self.data.Z):
            var_name = self._dummy_var_names[idx]
            val = self.cpx.solution.get_values(var_name)
            result[i] = val
        # Include branch dummies
        for name in self._branch_dummy_names:
            val = self.cpx.solution.get_values(name)
            result[name] = val
        return result

    def is_integer_solution(self, tol=1e-5):
        """Check if the current LP solution is integer."""
        for idx in range(len(self.routes)):
            var_name = self._route_var_names[idx]
            val = self.cpx.solution.get_values(var_name)
            if tol < val < 1.0 - tol:
                return False
        # Also check dummy variables are all zero
        for name in self._dummy_var_names:
            val = self.cpx.solution.get_values(name)
            if val > tol:
                return False
        # Also check branch dummies are all zero
        for name in self._branch_dummy_names:
            val = self.cpx.solution.get_values(name)
            if val > tol:
                return False
        return True

    def compute_reduced_cost(self, route, lambda_i, lambda_v0, lambda_d0):
        """
        Compute the reduced cost of a route given dual variables.

        c̄_r = c_r - Σ α_ir λ_i - λ_v0 - d_r λ_d0    (eq 37)
        """
        rc = route.cost
        for i in self.data.Z:
            rc -= route.get_alpha(i) * lambda_i.get(i, 0.0)
        rc -= lambda_v0
        rc -= route.num_drones * lambda_d0
        return rc
