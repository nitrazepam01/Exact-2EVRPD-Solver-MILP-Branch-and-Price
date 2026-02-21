"""
Route data structure for the B&P algorithm.
A Route represents a complete vehicle route with optional drone assignments.
"""


class Route:
    """
    Represents a feasible route for one vehicle (possibly with drones).

    Attributes:
        vehicle_nodes: list[int]
            Ordered sequence of vehicle nodes including depot (0).
            e.g. [0, 3, 7, 0]
        drone_assignments: dict[int, list[list[int]]]
            Maps each vehicle node -> list of drone routes.
            Each drone route is a list of drone nodes served sequentially.
            e.g. {3: [[5, 6], [8]], 7: [[4]]}
            means at vehicle node 3: drone1 serves [5,6], drone2 serves [8];
                 at vehicle node 7: drone1 serves [4].
        num_drones: int
            Number of drones allocated to this route (k̄_d for this route).
        cost: float
            Total duration of the route c_r (travel + waiting times).
    """

    def __init__(self, vehicle_nodes, drone_assignments, num_drones, cost):
        self.vehicle_nodes = vehicle_nodes  # e.g. [0, 3, 7, 0]
        self.drone_assignments = drone_assignments  # {veh_node: [[drone_route1], [drone_route2], ...]}
        self.num_drones = num_drones  # d_r in the paper
        self.cost = cost  # c_r in the paper

        # --- Precompute cached sets for O(1) lookup ---
        self._customers_served = set()
        for node in self.vehicle_nodes:
            if node != 0:
                self._customers_served.add(node)
        for vn, drone_routes in self.drone_assignments.items():
            for dr in drone_routes:
                for dn in dr:
                    self._customers_served.add(dn)

        self._vehicle_arcs = set()
        for idx in range(len(self.vehicle_nodes) - 1):
            self._vehicle_arcs.add((self.vehicle_nodes[idx], self.vehicle_nodes[idx + 1]))

        self._drone_dispatches = set()
        for vn, drone_routes in self.drone_assignments.items():
            for dr in drone_routes:
                for dn in dr:
                    self._drone_dispatches.add((vn, dn))

    def get_all_customers(self):
        """Return set of all customers (vehicle + drone) served by this route. O(1)."""
        return self._customers_served

    def get_alpha(self, customer):
        """α_ir: 1 if customer i is visited by this route, 0 otherwise. O(1)."""
        return 1 if customer in self._customers_served else 0

    def get_vehicle_arcs(self):
        """Returns cached set of (i, j) vehicle arcs. O(1)."""
        return self._vehicle_arcs

    def get_gamma(self, i, j):
        """γ_ijr: 1 if route r traverses vehicle arc (i, j), 0 otherwise. O(1)."""
        return 1 if (i, j) in self._vehicle_arcs else 0

    def get_drone_dispatches(self):
        """Returns cached set of (vehicle_node, drone_node) pairs. O(1)."""
        return self._drone_dispatches

    def get_xi(self, i, j):
        """ξ_ijr: 1 if drone node j is served from vehicle node i in this route. O(1)."""
        return 1 if (i, j) in self._drone_dispatches else 0

    def __repr__(self):
        return (f"Route(veh={self.vehicle_nodes}, drones={self.drone_assignments}, "
                f"num_drones={self.num_drones}, cost={self.cost:.2f})")
