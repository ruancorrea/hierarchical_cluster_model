import logging
from tqdm import tqdm
from sklearn.cluster import KMeans
from app.shared.ortools import (
    solve as ortools_solve,
    ORToolsParams
)
from app.types import (
    CVRPInstance,
    CVRPSolution,
    CVRPSolutionVehicle
)
from app.eval import evaluate_solution
from app.utils import create_CVRPInstance
import numpy as np


class HCM_ONLINE:
    def __init__(
        self,
        n_unit_loads: int,
        data: CVRPInstance,
        clustering: KMeans,
        subclusterings: dict,
        alpha_criteria: float,
        beta_distance: float,
        distribution_unit_loads: list,
        time_limit_ms_tsp: int= 1_000
    ):
        """Initialize the HCM_ONLINE class with the provided parameters.
        This class is responsible for solving the CVRP problem using the Hierarchical Cluster Model (HCM)."""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.n_unit_loads=n_unit_loads
        self.data=data
        self.data.vehicle_capacity=180
        self.clustering=clustering
        self.subclusterings=subclusterings
        self.alpha_criteria=alpha_criteria
        self.beta_distance=beta_distance
        self.distribution_unit_loads=distribution_unit_loads
        self.ortools_tsp_params=ORToolsParams(
            max_vehicles=1,
            time_limit_ms=time_limit_ms_tsp
        )
        self.routes = list()

    def define_unit_load_of_package(self, current_point: list, unit_loads: dict):
        """Define the unit load of a package based on its location.
        This function finds the nearest unit load based on the distance to the package's location."""
        def calculate_distance(point_A, point_B):
            """Calculate the Euclidean distance between two points."""
            return np.sqrt(np.sum((point_A - point_B) ** 2))

        min_distance=np.inf
        unit_load_min_distance=None
        for unit_load, packages in unit_loads.items():
            if len(packages) == 0:
                continue
            distance = min(
                [
                    calculate_distance(current_point, np.array([point_unit_load.point.lat, point_unit_load.point.lng])) 
                    for point_unit_load in packages
                ]
            )
            if min_distance > distance:
                min_distance=distance
                unit_load_min_distance=unit_load

        if min_distance > self.beta_distance:
            cluster=self.clustering.predict([current_point])[0]
            index=0
            if cluster in self.subclusterings.keys():
                sub_cluster=self.subclusterings[cluster].predict([current_point])[0]
                index=sub_cluster
            unit_load_min_distance=self.distribution_unit_loads.index(cluster) + index

        return unit_load_min_distance

    def generate_route(self, instance: CVRPInstance):
        """Generate route from a tsp solution."""
        self.logger.info(f"Generate route.")
        solution=None
        while True:
            solution=ortools_solve(instance, self.ortools_tsp_params)# TSP
            if isinstance(solution, CVRPSolution):
                break
        return CVRPSolutionVehicle(instance.origin, solution.deliveries)

    def run(self):
        """Run the HCM_ONLINE algorithm to solve the CVRP problem."""
        unit_loads_packages = {unit_load: [] for unit_load in range(self.n_unit_loads)}
        unit_loads_capacity = {unit_load: 0 for unit_load in range(self.n_unit_loads)}
        routes=list()
        self.logger.info(f"Packages Allocation")
        for package in tqdm(self.data.deliveries):
            package_current_point = np.array([package.point.lat, package.point.lng])
            unit_load = self.define_unit_load_of_package(package_current_point, unit_loads_packages)
            unit_loads_packages[unit_load].append(package)
            unit_loads_capacity[unit_load] += package.size
            if unit_loads_capacity[unit_load] >= self.data.vehicle_capacity * self.alpha_criteria:
                routes.append(unit_loads_packages[unit_load])
                unit_loads_packages[unit_load] = []
                unit_loads_capacity[unit_load] = 0

        for unit_load, packages in unit_loads_packages.items():
            if unit_loads_capacity[unit_load] > 0:
                routes.append(packages)

        vehicles=list()
        self.logger.info(f"Generate Routes")
        for route in tqdm(routes):
            vehicles.append(self.generate_route(create_CVRPInstance(self.data, route)))

        self.solution = CVRPSolution(
            name=self.data.name,
            vehicles=vehicles
        )

        self.logger.info(f"Routes distance calculate.")
        self.distance = evaluate_solution(self.data, self.solution)
        return self.solution, self.distance

