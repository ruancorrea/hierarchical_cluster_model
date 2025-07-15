from dataclasses import dataclass
from typing import Iterable, Optional, Any, List

import requests
import numpy as np

from app.types import Point


EARTH_RADIUS_METERS = 6371000


@dataclass
class OSRMConfig:
    host: str = "http://localhost:5000"
    timeout_s: int = 600


def calculate_distance_matrix_m(
    points: Iterable[Point], config: Optional[OSRMConfig] = None
):
    config = config or OSRMConfig()

    if len(points) < 2:
        return 0

    coords_uri = ";".join(
        ["{},{}".format(point.lng, point.lat) for point in points]
    )

    response = requests.get(
        f"{config.host}/table/v1/driving/{coords_uri}?annotations=distance",
        timeout=config.timeout_s,
    )

    response.raise_for_status()

    return np.array(response.json()["distances"])


# Função OSRM modificada
def calculate_distance_sub_matrix(
    coords_list: List[Point],  # Lista de pontos únicos para esta chamada OSRM
    source_indices: List[int], # Índices (em coords_list) para as origens
    dest_indices: List[int],   # Índices (em coords_list) para os destinos
    config: Optional[OSRMConfig] = None,
):
    config = config or OSRMConfig()

    if not coords_list:
        return np.array([])
    if not source_indices or not dest_indices: # Precisa de origens e destinos
        return np.array([[] for _ in source_indices]) # Retorna matriz com N_sources linhas e 0 colunas, ou similar

    coords_uri = ";".join(
        [f"{point.lng},{point.lat}" for point in coords_list]
    )
    sources_uri = ";".join(map(str, source_indices))
    destinations_uri = ";".join(map(str, dest_indices))

    request_url = (
        f"{config.host}/table/v1/driving/{coords_uri}"
        f"?annotations=distance&sources={sources_uri}&destinations={destinations_uri}"
    )
    
    # print(f"  OSRM Request URL (first few chars): {request_url[:150]}...")
    # print(f"  Coords: {len(coords_list)}, Sources: {len(source_indices)}, Dests: {len(dest_indices)}")

    response = requests.get(request_url, timeout=config.timeout_s)
    response.raise_for_status()
    
    # A resposta["distances"] será uma matriz len(source_indices) x len(dest_indices)
    return np.array(response.json()["distances"])


def calculate_route_distance_m(
    points: Iterable[Point], config: Optional[OSRMConfig] = None
):
    config = config or OSRMConfig()

    if len(points) < 2:
        return 0

    coords_uri = ";".join(
        "{},{}".format(point.lng, point.lat) for point in points
    )

    response = requests.get(
        f"{config.host}/route/v1/driving/{coords_uri}?annotations=distance&continue_straight=false",
        timeout=config.timeout_s,
    )

    response.raise_for_status()

    return min(r["distance"] for r in response.json()["routes"])


def calculate_distance_matrix_great_circle_m(
    points: Iterable[Point], config: Any = None
) -> np.ndarray:
    """Distance matrix using the Great Circle distance
    This is an Euclidean-like distance but on spheres [1]. In this case it is
    used to estimate the distance in meters between locations in the Earth.

    Parameters
    ----------
    points
        Iterable with `lat` and `lng` properties with the coordinates of a
        delivery

    Returns
    -------
    distance_matrix
        Array with the (i, j) entry indicating the Great Circle distance (in
        meters) between the `i`-th and the `j`-th point

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Great-circle_distance
    Using the third computational formula
    """
    points_rad = np.radians([(point.lat, point.lng) for point in points])

    delta_lambda = points_rad[:, [1]] - points_rad[:, 1]  # (N x M) lng
    phi1 = points_rad[:, [0]]  # (N x 1) array of source latitudes
    phi2 = points_rad[:, 0]  # (1 x M) array of destination latitudes

    delta_sigma = np.arctan2(
        np.sqrt(
            (np.cos(phi2) * np.sin(delta_lambda)) ** 2
            + (
                np.cos(phi1) * np.sin(phi2)
                - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
            )
            ** 2
        ),
        (
            np.sin(phi1) * np.sin(phi2)
            + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda)
        ),
    )

    return EARTH_RADIUS_METERS * delta_sigma


def calculate_route_distance_great_circle_m(points: Iterable[Point]) -> float:
    """Compute total distance from moving from starting point to final
    The total distance will be from point 0 to 1, from 1 to 2, and so on in
    the order provided.

    Parameters
    ----------
    points
        Iterable with `lat` and `lng` properties with the coordinates of a
        delivery

    Returns
    -------
    route_distance
        Total distance from going to the first point to the next until the last
        one
    """

    distance_matrix = calculate_distance_matrix_great_circle_m(points)

    point_indices = np.arange(len(points))

    return distance_matrix[point_indices[:-1], point_indices[1:]].sum()

def get_nearest_road(point: Point, config: Optional[OSRMConfig] = None, number: Optional[int] = 1):
    """Encontra a rua mais próxima a um ponto (lat, lon) usando OSRM Nearest."""
    config = config or OSRMConfig()

    response = requests.get(
        f"{config.host}/nearest/v1/driving/{point.lng},{point.lat}?number={number}",
        timeout=config.timeout_s,
    )
    response.raise_for_status()
    response = response.json()
    
    if 'waypoints' in response and response['waypoints']:
        locations = np.array([ w['location'] for w in response['waypoints'] ])[0]
        point = Point(lng=locations[0], lat=locations[1])
        return point  
    
    
    return None
