# Reproducibility
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
# Each round yields ~1 rung1 + occasional elite-reval; 4 matches the warmup
# window so surrogate-first activates right after warmup ends.
SURROGATE_MIN_HIGH_FIDELITY_SAMPLES = 4
# After warmup, Rung0 uses the surrogate prediction; a few real evals per round
# keep the surrogate calibrated.
SURROGATE_FIRST_WARMUP_ROUNDS = 3
SURROGATE_FIRST_REAL_EVALS_PER_ROUND = 2

# Stagnation adaptation (per client)
GA_STAGNATION_WINDOW = 3
GA_STAGNATION_MIN_DELTA = 0.002
ADAPTIVE_EXTRA_MUTANTS = 2

# Run GA during warmup, then every N visits; reuse last-best HP in between.
GA_WARMUP_ROUNDS = 3
GA_RUN_EVERY_N_ROUNDS = 2

# Skip retraining when a recently-evaluated HP signature is still fresh.
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

# FedProx proximal term applied in the final local SGD pass.
LOCAL_TRAIN_MU = 0.1

# Drift penalty in the GA fitness: fitness = val_acc - lambda * drift_rms.
# Warmup keeps lambda small for the first K rounds so round-1 exploration
# isn't penalized by the (inevitable) initial drift of an untrained global.
FITNESS_DRIFT_PENALTY_LAMBDA = 0.1
FITNESS_LAMBDA_WARMUP_ROUNDS = 3
FITNESS_LAMBDA_WARMUP_START = 0.1

# Surrogate target: "fitness" (raw) or "post_agg_proxy" (val_acc - lambda*drift,
# steering the GA toward HPs whose local gain survives FedAvg).
SURROGATE_TARGET = "post_agg_proxy"

# Enable the GA+Surrogate HP tuner. When False, clients train with DEFAULT_*.
ENABLE_GA = False
ENABLE_SURROGATE_GA = False

# Server-side federated GA. When True, the server picks one HP per round
# and broadcasts to ALL clients (eliminates inter-client HP heterogeneity).
# Takes precedence over ENABLE_GA on the client side.
ENABLE_FED_GA = False
FED_GA_POPULATION_SIZE = 4

# Non-GA server-side HP searchers. Both share the
# same coupling regime as FedGA (one HP per round, broadcast to all clients)
# so the comparison isolates the choice of search algorithm. Mutually exclusive
# with ENABLE_FED_GA and with each other; server_app raises if >1 is True.
ENABLE_FED_RANDOM_SEARCH = False
ENABLE_FED_TPE = False

# FedGA fitness-signal knobs.
#   FED_GA_SEED_BASELINE: when True (default), the initial population includes
#     the textbook expert HP (DEFAULT_*). Note the cold-start caveat: the
#     seeded HP is evaluated in round 1 against a randomly-initialized
#     aggregate model, receives near-chance fitness, and tends to be discarded
#     as generation-worst. Set False for a fully random initial population.
#   FED_GA_USE_DELTA_FITNESS: when True, fitness is the per-round Δeval-acc
#     instead of absolute accuracy. Caveat: an HP evaluated right after a
#     crash (or on the steep early learning curve) inherits an inflated Δ.
FED_GA_SEED_BASELINE = True
FED_GA_USE_DELTA_FITNESS = False

# --- Telemetry ---
ENABLE_TELEMETRY_EXPORT = True
ENABLE_GA_CANDIDATE_EXPORT = True
TELEMETRY_BASE_DIR = "artifacts"

# --- Defaults used when ENABLE_GA=False ---
# Textbook settings a practitioner would pick without HP search. Deliberately
# not tuned -- the point of the ENABLE_GA=True path is exactly to do better.
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 0.01
DEFAULT_OPTIMIZER = "sgd"
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MOMENTUM = 0.9

# --- Partition ---
# "iid": equal split; "dirichlet": class-conditional Dirichlet with
# DIRICHLET_ALPHA. Smaller alpha => more heterogeneity.
# alpha=0.5 follows Hsu, Qi & Brown (2019) as the standard "moderate non-IID"
# setting in FL benchmarks (FedML, Flower). alpha=1.0 is mild, 0.1 is hard,
# 0.01 pathological; 0.5 challenges the global model without collapsing it.
PARTITION_MODE = "dirichlet"
DIRICHLET_ALPHA = 0.5
# Test partition is always IID so per-client eval is an unbiased task sample.
TEST_PARTITION_MODE = "iid"

# --- Dataset ---
# Which dataset+model pair the run uses. Each name resolves to a (trainset,
# testset, model_factory) triple via task._build_dataset().
#   "cifar10"  -> torchvision CIFAR-10 + CIFARResNet (~11M params)
#   "femnist"  -> HuggingFace flwrlabs/femnist (62-class) + FEMNISTCNN (LEAF spec)
# We keep the Dirichlet partitioner across datasets so α=0.5 has consistent
# meaning. FEMNIST's natural writer-level non-IID is documented but not used
# as the partition signal.
DATASET_NAME = "cifar10"
# Subsample size when loading FEMNIST (matches CIFAR-10 scale to keep per-run
# wall-time comparable). Ignored for cifar10.
FEMNIST_TRAIN_SAMPLES = 60000
FEMNIST_TEST_SAMPLES = 10000


# ---------------------------------------------------------------------------
# Runtime override loader.
#
# When `FGA_CONFIG_PATH` is set, the runner CLI has written a resolved JSON
# snapshot there; apply it as the last step of module import. Keys must match
# UPPER_CASE constants already defined above (typos error out; silent drops
# would let YAML mistakes nullify experiments). Ray actor workers re-import
# this module fresh and inherit the env var, so server and clients converge
# on the same resolved values without any explicit propagation step.
# ---------------------------------------------------------------------------
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
