import torch

# Pytorch config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed config
SEED = 0

# MLP config
INPUT_DIM = 11
OUTPUT_DIM = 1
EPOCHS = 30

# Genetic Algorithm config
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.2
POPULATION_SIZE = 40
NUMBER_OF_GENERATIONS = 10