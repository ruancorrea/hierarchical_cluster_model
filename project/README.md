# Hierarchical Cluster Model Application

## Installation

```bash
pip install -r requirements.txt
``` 

### Running example

```bash
python3 -m app.application \
    --train_instances data/cvrp-instances-1.0/train/pa-0 \
    --eval_instances batches/instances/pa-0 \
    --output app/solutions
```