#!/usr/bin/env bash

# Before running this script, make sure the TU Wien VPN is active if needed.

# Exit on error
set -e

LOCAL_DIR="/Users/emilejohnston/DataspellProjects/synthetic-cover-text-generator"
REMOTE="e12229987@cluster.datalab.tuwien.ac.at"
REMOTE_DIR=""

# Synchronize files on cluster from local device using rsync
rsync -avP -e ssh \
  --exclude-from='.rsyncignore' \
  --exclude='slurm_output/' \
  "$LOCAL_DIR" \
  "$REMOTE:$REMOTE_DIR"
