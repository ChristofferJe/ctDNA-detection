#!/bin/bash
#SBATCH --account MutationAnalysis 
#SBATCH --mem 20g
#SBATCH --time 48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=20

echo "Running on node: $(hostname)"

module purge

source /home/christofferjensen/miniforge3/etc/profile.d/conda.sh
conda activate cf

echo "Using Python from: $(which python)"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Testing torch import..."
python -c "import torch; print('Torch OK')"

PROJECT="cfDNA"
ENTITY="christoffer-je-aarhus-university"

echo "Creating sweep..."

SWEEP_OUTPUT=$(wandb sweep --project $PROJECT sweep_config_MLP_combined_extended.yaml 2>&1)

echo "$SWEEP_OUTPUT"

SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep "Run sweep agent with" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to extract sweep ID."
    exit 1
fi

echo "Sweep created: $SWEEP_ID"
echo "Starting $SLURM_NTASKS agents..."

# Prevent CPU oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Launch parallel agents
srun wandb agent $SWEEP_ID

