#!/usr/bin/env bash

# Exit on error
set -e

# Local and remote base paths
LOCAL_BASE="/Users/emilejohnston/DataspellProjects/synthetic-cover-text-generator"
REMOTE="e12229987@cluster.datalab.tuwien.ac.at"
REMOTE_BASE="synthetic-cover-text-generator"

# Pull uv env changes from the slurm cluster to the local device using rsync
rsync -avP -e ssh \
  --include='pyproject.toml' \
  --include='uv.lock' \
  --exclude='*' \
  "$REMOTE:$REMOTE_BASE/" \
  "$LOCAL_BASE/"
