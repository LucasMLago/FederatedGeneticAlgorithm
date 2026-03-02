# FederatedGeneticAlgorithm

## Setup

```bash
pip install -e .
```

## Run

```bash
# CPU (default)
flwr run .

# GPU
flwr run . local-simulation-gpu
```

## Google Colab (GPU)

1. Select **Runtime → Change runtime type → T4 GPU**.
2. Install and run:

```python
!pip install -e .
!flwr run . local-simulation-gpu
```

> The `local-simulation-gpu` federation allocates `num-gpus = 0.5` per Ray worker, which is required
> for `torch.cuda.is_available()` to return `True` inside each client. The default federation uses
> `num-gpus = 0.0` and runs on CPU only.
