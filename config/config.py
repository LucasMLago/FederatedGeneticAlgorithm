import torch

# Pytorch config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed config
SEED = 0

# Model config
EPOCHS = 10

# Genetic Algorithm config
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.2
POPULATION_SIZE = 10
NUMBER_OF_GENERATIONS = 5