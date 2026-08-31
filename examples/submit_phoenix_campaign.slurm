#!/bin/bash
# Multi-morphology uptake campaign on Phoenix.
#
# Separate from submit_phoenix_gpu.slurm because the resource shape is
# different in kind, not degree: that script runs a single bulk reference in
# an hour on one GPU, this runs M complete hydration loops, each containing a
# few hundred fixed-lambda FEP states. Reusing the 1-hour walltime would put
# the job in the queue and then kill it mid-morphology.
#SBATCH --job-name=aem-campaign
#SBATCH --output=aem-campaign-%j.out
#SBATCH --error=aem-campaign-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --account=gts-rramprasad3-coda20
# Free backfill avoids the research group's exhausted inferno billing-minute
# allocation. NOTE: embers is preemptible. The campaign checkpoints per
# morphology and resumes by default, so a preemption costs the current
# morphology and not the campaign -- requeue the same script and it continues.
#SBATCH --qos=embers
#SBATCH --requeue

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

# Preflight the FEP package specifically. The GPU check above does not cover
# it, and a build without it fails only once the first lambda state launches --
# after the dry membrane has already been built and annealed. Checking here
# costs a second and converts hours of wasted allocation into an instant exit.
for style in lj/cut/soft lj/cut/coul/long/soft coul/long/soft; do
    lmp -h 2>/dev/null | tr ' ' '\n' | grep -qx "$style" || {
        echo "ERROR: pair style $style missing -- LAMMPS lacks the FEP package." >&2
        echo "       Rebuild LAMMPS with -D PKG_FEP=yes. See the 'Fallback if the" >&2
        echo "       FEP package is absent' section of docs/fep_design.md." >&2
        exit 3
    }
done

# Screening resolution is the default and is what makes M-fold replication
# affordable (~6.4x cheaper per morphology than production resolution).
# Add --production-resolution only for a final quotable number on one system.
aemwater campaign --config membrane.yaml --morphologies 3

echo "campaign_result.json written; per-morphology spread is in the report."
