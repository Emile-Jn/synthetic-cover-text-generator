#!/usr/bin/env bash

# Exit on error
set -e

# Local and remote base paths
LOCAL_BASE="/Users/emilejohnston/DataspellProjects/synthetic-cover-text-generator"
REMOTE="e12229987@cluster.datalab.tuwien.ac.at"
REMOTE_BASE="synthetic-cover-text-generator"

# Pull uv env changes from the slurm cluster to the local device using rsync
# include only .out files in the repository root (do not traverse into subdirectories)
# Place .out files into local slurm_output/ while keeping pyproject.toml and uv.lock in repo root

# Ensure local slurm_output directory exists
mkdir -p "$LOCAL_BASE/slurm_output"

# Sync config files to repo root (pyproject.toml, uv.lock) and generated_samples
rsync -avP -e ssh \
  --include='pyproject.toml' \
  --include='uv.lock' \
  --include='generated_samples' \
  --include='generated_samples/**' \
  --exclude='*' \
  "$REMOTE:$REMOTE_BASE/" \
  "$LOCAL_BASE/"

# Sync .out files from remote repo root into local slurm_output directory
rsync -avP -e ssh \
  --include='*.out' \
  --exclude='*' \
  "$REMOTE:$REMOTE_BASE/" \
  "$LOCAL_BASE/slurm_output/"
