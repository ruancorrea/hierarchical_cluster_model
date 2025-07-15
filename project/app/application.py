import os
import time
import logging
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
from app.types import CVRPInstance
from app.hcm_offline import HCM_OFFLINE
from app.hcm_online import HCM_ONLINE
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_data(args):

    train_path = Path(args.train_instances)
    train_path_dir = train_path if train_path.is_dir() else train_path.parent
    train_files = (
        [train_path] if train_path.is_file() else list(train_path.iterdir())
    )

    train_instances = [CVRPInstance.from_file(f) for f in train_files[:240]]
    
    test_path = Path(args.eval_instances)
    test_path_dir = test_path if test_path.is_dir() else test_path.parent
    test_files = (
        [test_path] if test_path.is_file() else list(test_path.iterdir())
    )

    return train_instances, test_files

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--train_instances", type=str, required=True)
    parser.add_argument("--eval_instances", type=str, required=True)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    train_files, test_files=get_data(args)
    output_dir = Path(args.output or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_unit_loads=28
    n_clusters = range(3, n_unit_loads+1)
    E_RATES=[0.0, 0.025, 0.05, 0.1]
    alpha_criteria=0.94
    beta_distance=0.005
    test_size=0.05
    seed=42
    region="pa-0"

    hcm_offline= HCM_OFFLINE(
        data=train_files,
        n_clusters=n_clusters, 
        n_unit_loads=n_unit_loads, 
        E_RATES=E_RATES, 
        alpha_criteria=alpha_criteria, 
        beta_distance=beta_distance,
        test_size=test_size,
        seed=seed
    )
    clustering, subclusterings, distribution_unit_loads = hcm_offline.run()
    def solve(file):
        instance = CVRPInstance.from_file(file)
        start = time.perf_counter()
        #logging.info('solution', instance.name)
        hcm_online=HCM_ONLINE(
            n_unit_loads= n_unit_loads, 
            data= instance, 
            clustering= clustering, 
            subclusterings= subclusterings, 
            alpha_criteria= alpha_criteria,
            beta_distance= beta_distance,
            distribution_unit_loads= distribution_unit_loads
        )

        solution, distance = hcm_online.run()
        solution.to_file(output_dir / f"{instance.name}.json")
        print("region", region, instance.name, "distance", distance)

    with Pool(os.cpu_count()) as pool:
        list(tqdm(pool.imap(solve, test_files), total=len(test_files)))
        
