#!/bin/bash
#SBATCH --job-name=ADGAM-ablE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=linux46,linux45,linux48,linux49,linux50,linux52,linux51,linux56,linux57,linux58,linux53,linux47,compsci-cluster-fitz-22,compsci-cluster-fitz-24,compsci-cluster-fitz-26,compsci-cluster-fitz-30

set -euo pipefail
mkdir -p logs

# =========================
# User-configurable lists
# =========================
DATASETS=${DATASETS:- "blood_transfusion wine compas telescope"}
GRID_MIN_SAMPLES=${GRID_MIN_SAMPLES:-"50"}

# =========================
# Fixed params (same as main 4-12-version, except backward_fit=0)
# =========================
RUNS=${RUNS:-5}
RESULTS_ROOT=${RESULTS_ROOT:-"results"}
M=${M:-100}
ETA=${ETA:-1}
STEP_MODE=${STEP_MODE:-"grad"}
MAX_DEPTH=${MAX_DEPTH:-4}
MIN_GAIN=${MIN_GAIN:-0.01}
TOPK=${TOPK:-5}

GRID_MAX_DEPTHS=${GRID_MAX_DEPTHS:-"3,5"}
GRID_MIN_GAIN=${GRID_MIN_GAIN:-"0.01"}
GRID_STRUCTURE_ALPHA=${GRID_STRUCTURE_ALPHA:-"0.5,0.75"}
GRID_MAX_THRESHOLDS_FOR_TREE=${GRID_MAX_THRESHOLDS_FOR_TREE:-"15"}
GRID_LEAF_FSG_MAX_SUPPORT_SIZE=${GRID_LEAF_FSG_MAX_SUPPORT_SIZE:-"8"}

# Ablation E: backward_fit is explicitly OFF
BACKWARD_FIT=0

STOCHASTIC_COORD=${STOCHASTIC_COORD:-1}
STOCHASTIC_TOPK=${STOCHASTIC_TOPK:-3}
SILENCE_PARENT=${SILENCE_PARENT:-1}
RANDOM_STATE=${RANDOM_STATE:-42}

PY=${PY:-"ADGAM_cv_sparsity.py"}
EXTRA_ARGS="${EXTRA_ARGS:-${*:-}}"

# =========================
# Build dataset and min_samples lists
# =========================
read -ra DATASET_LIST <<< "${DATASETS}"
nDATA=${#DATASET_LIST[@]}

MSL_LIST=()
if [[ -n "${GRID_MIN_SAMPLES}" ]]; then
  IFS=',' read -ra MSL_LIST <<< "${GRID_MIN_SAMPLES}"
else
  MSL_LIST=("${MIN_SAMPLES_LEAF:-100}")
fi
nMSL=${#MSL_LIST[@]}

# =========================
# Self-submit array (phase 1)
# =========================
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TOTAL=$(( nDATA * nMSL ))
  echo "Submitting array with ${TOTAL} tasks (dataset × min_samples_leaf)"
  sbatch --export=ALL,EXTRA_ARGS="${EXTRA_ARGS}" --array=0-$((TOTAL-1)) "$0"
  exit 0
fi

# =========================
# Decode array task id (phase 2)
# =========================
tid=${SLURM_ARRAY_TASK_ID}
idxDATA=$(( tid % nDATA )); tid=$(( tid / nDATA ))
idxMSL=$(( tid % nMSL ))

DATASET="${DATASET_LIST[$idxDATA]}"
MSL="${MSL_LIST[$idxMSL]}"

ABL_TAG="ablE_nobwf_msl${MSL}"
OUT_ROOT="${RESULTS_ROOT}"
echo "[RUN] dataset=${DATASET}  msl=${MSL}  backward_fit=${BACKWARD_FIT}  ablation_tag=${ABL_TAG}"

# =========================
# Thread env
# =========================
export PYTHONUNBUFFERED=TRUE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

STOCHASTIC_COORD_FLAG=$([ "${STOCHASTIC_COORD}" -eq 1 ] && echo "--stochastic_coord" || echo "")
# NOTE: no --backward_fit flag (ablation_e tests without backward fitting)

# =========================
# Run
# =========================
srun python3 -u "${PY}" \
  --dataset "${DATASET}" \
  --runs "${RUNS}" \
  --random_state "${RANDOM_STATE}" \
  --results_root "${OUT_ROOT}" \
  --ablation_tag "${ABL_TAG}" \
  --step_mode "${STEP_MODE}" \
  --eta "${ETA}" \
  --M "${M}" \
  --max_depth "${MAX_DEPTH}" \
  --min_gain_fraction "${MIN_GAIN}" \
  --min_samples_leaf "${MSL}" \
  --k "${TOPK}" \
  --leaf_fitter "fsg" \
  ${STOCHASTIC_COORD_FLAG} \
  --stochastic_topk "${STOCHASTIC_TOPK}" \
  --silence_parent "${SILENCE_PARENT}" \
  --grid_min_gain "${GRID_MIN_GAIN}" \
  --grid_structure_alpha "${GRID_STRUCTURE_ALPHA}" \
  --grid_max_depths "${GRID_MAX_DEPTHS}" \
  --grid_max_thresholds_for_tree "${GRID_MAX_THRESHOLDS_FOR_TREE}" \
  --grid_leaf_fsg_max_support_size "${GRID_LEAF_FSG_MAX_SUPPORT_SIZE}" \
  ${EXTRA_ARGS}
