#!/bin/bash
#SBATCH --job-name=aemwater-gpu
#SBATCH --output=aemwater-%j.out
#SBATCH --error=aemwater-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH --account=gts-rramprasad3-coda20
# Free backfill avoids the research group's exhausted inferno billing-minute
# allocation. The explicit V100 request prevents generic-GPU misrouting.
#SBATCH --qos=embers

set -euo pipefail

module reset
module load anaconda3/2023.03
eval "$(conda shell.bash hook)"
conda activate aem

# Loading this after Conda ensures that the CUDA build, rather than the
# CPU-only conda-forge lmp in the aem environment, is selected.
module load gcc/12.3.0 mvapich2/2.3.7-1
module load lammps/20240829-mva2-cuda-openmp-cc70

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export AEMWATER_LAMMPS_ARGS="-sf gpu -pk gpu 1"

cd /storage/home/hcoda1/6/wschertzer3/aem-water-uptake

echo "Host: $(hostname)"
echo "GPU allocation: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "LAMMPS: $(command -v lmp)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
lmp -h | grep -qE '(^| )GPU( |$)' || {
    echo "ERROR: selected LAMMPS binary was not built with the GPU package" >&2
    exit 2
}

# One MPI rank drives one GPU. GPU-enabled pair and PPPM styles run on it;
# OpenMP threads handle work for which this LAMMPS build has no GPU variant.
aemwater bulk
