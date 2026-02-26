#!/bin/bash
set -e

# project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Run Federated Learning with GPU
source .venv/bin/activate
flwr run . --federation-config "backend.client-resources.num-gpus=1.0"
