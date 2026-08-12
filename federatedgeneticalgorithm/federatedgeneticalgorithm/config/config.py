SEED = 0

# --- Federated runtime ---
NUM_SERVER_ROUNDS = 15
FRACTION_TRAIN = 0.5
LOCAL_EPOCHS = 8

# --- Genetic Algorithm ---
EPOCHS = 10
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.3
TOURNAMENT_SIZE = 3
POPULATION_SIZE = 4
NUMBER_OF_GENERATIONS = 3

# --- Surrogate ---
SURROGATE_RETRAIN_INTERVAL = 2
NUM_CANDIDATES_TO_EVALUATE = 4
SURROGATE_TRIAL_POOL_SIZE = 40
SURROGATE_TOP_K = 3
# about one rung1 sample per round, so 4 lines up with the end of warmup
SURROGATE_MIN_HIGH_FIDELITY_SAMPLES = 4
# after warmup rung0 uses the surrogate prediction, plus this many real evals per round
SURROGATE_FIRST_WARMUP_ROUNDS = 3
SURROGATE_FIRST_REAL_EVALS_PER_ROUND = 2

# Stagnation adaptation (per client)
GA_STAGNATION_WINDOW = 3
GA_STAGNATION_MIN_DELTA = 0.002
ADAPTIVE_EXTRA_MUTANTS = 2

# Run GA during warmup, then every N visits; reuse last-best HP in between.
GA_WARMUP_ROUNDS = 3
GA_RUN_EVERY_N_ROUNDS = 2

# reuse a cached fitness if the same HP was evaluated at most this many rounds ago
FITNESS_CACHE_MAX_ROUND_AGE = 2

# --- Rungs ---
RUNG0_EPOCHS = 1
RUNG0_SUBSET_RATIO = 0.25
RUNG0_MU = 0.1
RUNG0_USE_SHARED_SUBSET = True

RUNG1_EPOCHS = 4
RUNG1_MU = 0.05
RUNG1_TOP_K = 1
ELITE_REEVAL_COUNT = 1
ELITE_REEVAL_EVERY_N_ROUNDS = 3

# FedProx mu for the final local training pass
LOCAL_TRAIN_MU = 0.1

# GA fitness = val_acc - lambda * drift_rms
FITNESS_DRIFT_PENALTY_LAMBDA = 0.1
# lambda goes linearly from WARMUP_START to the value above over the first rounds
FITNESS_LAMBDA_WARMUP_ROUNDS = 3
FITNESS_LAMBDA_WARMUP_START = 0.1

# "fitness" or "post_agg_proxy" (val_acc - lambda*drift)
SURROGATE_TARGET = "post_agg_proxy"

# False = clients train with the DEFAULT_* values below
ENABLE_GA = False
ENABLE_SURROGATE_GA = False

# server picks one HP per round and sends it to every client, overrides ENABLE_GA
ENABLE_FED_GA = False
FED_GA_POPULATION_SIZE = 4

# same one-HP-per-round setup as FedGA; at most one ENABLE_FED_* can be True
ENABLE_FED_RANDOM_SEARCH = False
ENABLE_FED_TPE = False

# seed the initial population with DEFAULT_*. It tends to score near chance
# in round 1 (untrained global model) and get dropped
FED_GA_SEED_BASELINE = True
# fitness = eval-acc gain over the previous round instead of absolute eval-acc
FED_GA_USE_DELTA_FITNESS = False

# --- Telemetry ---
ENABLE_TELEMETRY_EXPORT = True
ENABLE_GA_CANDIDATE_EXPORT = True
TELEMETRY_BASE_DIR = "artifacts"

# --- Defaults used when ENABLE_GA=False ---
# plain textbook values, not tuned
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 0.01
DEFAULT_OPTIMIZER = "sgd"
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MOMENTUM = 0.9

# --- Partition ---
# "iid" or "dirichlet"
PARTITION_MODE = "dirichlet"
# smaller = more skewed clients; 0.5 is the usual moderate non-IID (Hsu et al. 2019)
DIRICHLET_ALPHA = 0.5
# not read anywhere, client_app always splits the test set IID
TEST_PARTITION_MODE = "iid"

# --- Dataset ---
# "cifar10", "cifar10_small" or "femnist".
# FEMNIST also goes through the Dirichlet split, not its per-writer split
DATASET_NAME = "cifar10"
# FEMNIST subsample, close to CIFAR-10 size so run times are comparable
FEMNIST_TRAIN_SAMPLES = 60000
FEMNIST_TEST_SAMPLES = 10000


# FGA_CONFIG_PATH is the JSON the runner writes. Ray workers re-import this
# module with the same env var, so server and clients get the same overrides.
import os as _os
import json as _json


def _apply_runtime_overrides() -> None:
    path = _os.environ.get("FGA_CONFIG_PATH")
    if not path:
        return
    with open(path, encoding="utf-8") as fh:
        overrides = _json.load(fh)
    known = {name for name in list(globals()) if name.isupper() and not name.startswith("_")}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise RuntimeError(
            f"FGA_CONFIG_PATH={path} contains unknown config keys: {unknown}. "
            f"Add them to config.py first or remove from the YAML."
        )
    for key, value in overrides.items():
        globals()[key] = value


_apply_runtime_overrides()
