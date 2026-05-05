#!/bin/bash
# Generic SLURM template for an OrthoReg training job.
#
# Usage:
#   - Replace the placeholders below with values appropriate for your cluster
#     (partition, qos, account, mail-user, paths, etc.). Lines marked
#     "REPLACE_ME" must be set or the job will not submit.
#   - Override any Hydra config from the command line, e.g.
#       sbatch template.sh dataset=pendulum training.regularization=orthogonal +training.seed=0
#
# This template intentionally contains no cluster-specific identifiers.

#SBATCH --job-name=orthoreg
#SBATCH --partition=REPLACE_ME           # e.g. gpu, normal
#SBATCH --qos=REPLACE_ME                 # e.g. gpu_normal (delete if your site has no QOS)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# Activate your Python environment. Adjust to match your install
# (conda, venv, module load, etc.).
# source ~/.bashrc
# conda activate orthoreg

# Run from the repository root (the directory that contains pyproject.toml).
cd "$(dirname "$0")/../.."

python -m orthoreg.training.train "$@"
