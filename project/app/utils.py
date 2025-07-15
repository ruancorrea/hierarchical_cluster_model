from app.types import (
    CVRPInstance,
    JSONDataclassMixin, 
    Delivery,
    CVRPSolution,
    CVRPSolutionVehicle
)

def json_read(path: str) -> dict:
    try:
        with open(path, "r") as file:
            data = json.load(file)
        return data
    except:
        return None

def create_CVRPInstance(instance, deliveries: list[Delivery], factor=3) -> CVRPInstance:
    """ Creating an instanceCVRP with deliveries. """

    return CVRPInstance(
        name = instance.name,
        region= instance.region,
        origin= instance.origin,
        vehicle_capacity = factor* instance.vehicle_capacity,
        deliveries= deliveries
    )