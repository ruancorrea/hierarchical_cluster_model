import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from app.hcm_online import HCM_ONLINE
from app.utils import create_CVRPInstance
from app.types import CVRPInstance
from collections import defaultdict
import numpy as np

class HCM_OFFLINE:
    def __init__(
        self,
        data: list,
        n_clusters: list,
        n_unit_loads: int,
        E_RATES: list,
        alpha_criteria: float,
        beta_distance: float,
        test_size: float,
        seed: int
    ):
        """Initialize the HCM_OFFLINE class with the provided parameters."""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.instance_basic_data = data[0]
        self.H=[
                package
                for instance in data
                for package in instance.deliveries
            ]
        self.n_clusters=n_clusters
        self.n_unit_loads=n_unit_loads
        self.E_RATES=E_RATES
        self.alpha_criteria=alpha_criteria
        self.beta_distance=beta_distance
        self.test_size=test_size
        self.seed=seed


    def apply_remove_outliers(self, data: CVRPInstance, rate: float) -> tuple:
        """Remove outliers from the dataset using Isolation Forest."""
        deliveries = np.array([delivery for delivery in data.deliveries])
        points = np.array([[delivery.point.lat, delivery.point.lng] for delivery in data.deliveries])
        if rate==0.0:
            return deliveries, points
        iso_forest = IsolationForest(contamination=rate, random_state=self.seed)
        outliers = iso_forest.fit_predict(points)
        return deliveries[outliers == 1], points[outliers == 1]

    def apply_create_model(self, n_clusters: int, data: list):
        """Create a clustering model using KMeans."""
        model=KMeans(n_clusters=n_clusters, init='k-means++', random_state=self.seed, n_init='auto')
        labels=model.fit_predict(data)
        return model, labels

    def UL_allocation(self, labels: list):
        """ Distribution of unit loads
        based on the number of points in each cluster of the level one model.
        """

        _, clusters_counts = np.unique(labels, return_counts=True)
        total_packages = np.sum(clusters_counts)
        clusters_counts = dict(enumerate(clusters_counts))
        allocation = defaultdict(int)
        total_allocation = 0
        for cluster, count in clusters_counts.items():
            packages_percentage=(count / total_packages)
            allocation[cluster]=int(np.ceil(self.n_unit_loads * packages_percentage))
            total_allocation = total_allocation + allocation[cluster]

        while total_allocation > self.n_unit_loads:
            cluster = max(allocation, key=allocation.get)
            if allocation[cluster] > 1:
                allocation[cluster] = allocation[cluster] - 1
                total_allocation = total_allocation - 1

        distribution = list()
        for cluster, n_unit_loads in allocation.items():
            for _ in range(n_unit_loads):
                distribution.append(cluster)

        return allocation, distribution

    def define_H(self) -> tuple:
        """Split the dataset into two parts H1 and H2.
        H1 will be used for clustering and H2 for testing."""
        H1, H2 = train_test_split(self.H, test_size=self.test_size, random_state=self.seed)
        instance_H1 = create_CVRPInstance(instance=self.instance_basic_data, deliveries=H1, factor=1)
        instance_H2 = create_CVRPInstance(instance=self.instance_basic_data, deliveries=H2, factor=1)
        return instance_H1, instance_H2

    def run(self):
        """Run the HCM_OFFLINE algorithm."""
        H1, H2 = self.define_H()
        min_distance=np.inf
        choose_clustering=None
        choose_subclusterings=None
        choose_distribution_unit_loads=None

        for e in self.E_RATES:
            deliveries_Hc, points_Hc = self.apply_remove_outliers(data=H1, rate=e)
            Hc = create_CVRPInstance(instance=self.instance_basic_data,deliveries=list(deliveries_Hc), factor=1)
            for c in self.n_clusters:
                clustering, labels=self.apply_create_model(n_clusters=c, data=points_Hc)
                allocation_unit_loads, distribution_unit_loads=self.UL_allocation(labels)
                subclusterings=defaultdict()
                for subcluster, n_unit_loads in allocation_unit_loads.items():
                    subclusterings[subcluster], _ = self.apply_create_model(n_clusters=n_unit_loads, data=points_Hc[labels == subcluster])
                _, distance = HCM_ONLINE(
                    n_unit_loads=self.n_unit_loads,
                    data=H2,
                    clustering=clustering,
                    subclusterings=subclusterings,
                    alpha_criteria=self.alpha_criteria,
                    beta_distance=self.beta_distance,
                    distribution_unit_loads=distribution_unit_loads
                ).run()

                if min_distance > distance:
                    min_distance=distance
                    choose_clustering=clustering
                    choose_subclusterings=subclusterings
                    choose_distribution_unit_loads=distribution_unit_loads

        return choose_clustering, choose_subclusterings, choose_distribution_unit_loads
