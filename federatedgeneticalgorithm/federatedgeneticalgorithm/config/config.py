import torch

# Pytorch config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed config
SEED = 0

# Genetic Algorithm config
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.2
TOURNAMENT_SIZE = 3
POPULATION_SIZE = 10
NUMBER_OF_GENERATIONS = 5

# Multi-fidelity & Proximal Config
RUNG0_EPOCHS = 2
RUNG0_SUBSET_RATIO = 0.6
RUNG0_MU = 0.5

RUNG1_EPOCHS = 3
RUNG1_MU = 0.05

SURROGATE_RETRAIN_INTERVAL = 3
NUM_CANDIDATES_TO_EVALUATE = 5