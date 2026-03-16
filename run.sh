#!/bin/bash

# The following lines are SBATCH directives, they are read by the SLURM scheduler

# #SBATCH --partition=GPU-a100  # jobs run on the L0S GPU partition by default
#SBATCH --job-name=cover-generation
#SBATCH --gres=gpu:1      # request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00  # request 2 h of max runtime
#SBATCH --output=slurm_output/slurm-%j.out

# always include this, may provide useful information to the admins
echo "Running on: $SLURM_JOB_NODELIST"
# make sure uv environment is loaded
source .venv/bin/activate

# run the python script ($@ represents all the arguments passed to this bash script, e.g. test.py -nmax 3)
python3 "$@"

echo "Finished."

# Example usage:
# sbatch run.sh hello_world.py
# sbatch --partition=GPU-a100 run.sh -m src.vibe_check --prompt ""
# sbatch --partition=GPU-a100 run.sh -m src.train_lora --data-file amanneo/enron-mail-corpus-mini --model-name unsloth/Qwen3.5-4B-Base
# sbatch --partition=GPU-a100 run.sh -m src.train_lora --data-file stanfordnlp/imdb --model-name unsloth/Qwen3.5-4B-Base
# sbatch --partition=GPU-a100 run.sh -m src.train_lora --data-file stanfordnlp/imdb --model-name unsloth/Qwen3.5-4B-Base --n 64
# sbatch --partition=GPU-a100 run.sh -m src.train_lora --data-file stanfordnlp/imdb --model-name unsloth/Qwen3-4B-Base --n 64
