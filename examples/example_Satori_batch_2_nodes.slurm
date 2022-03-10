#!/bin/bash
#
#SBATCH -J example_Satori_batch_2_nodes
#SBATCH -o SALIENT/job_output/%x_%j.err
#SBATCH -e SALIENT/job_output/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --array=1-2
#SBATCH --mem=1T
#SBATCH --exclusive
#SBATCH --time 00:03:00
#SBATCH -p sched_system_all_8

# Script to run examples in the batch mode on Satori (2 nodes). READ
# ALL INSTRUCTIONS!!
#
# Submit the job using
#
# $ sbatch SALIENT/examples/example_Satori_batch_2_nodes.slurm
#
# Watch the job using
#
# $ squeue

# Activate environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=salient
source $HOME2/anaconda3/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

# Get JOB_NAME (set at the beginning of this script)
JOB_NAME=$SLURM_JOB_NAME

# The current program hardcodes using the environment variable
# SLURMD_NODENAME to distinguish machines. On Satori, the scheduler
# will set this variable.

# Set SALIENT root and PYTHONPATH
SALIENT_ROOT=$HOME/SALIENT
export PYTHONPATH=$SALIENT_ROOT

# Set the data paths
DATASET_ROOT=$HOME2/dataset
OUTPUT_ROOT=$SALIENT_ROOT/job_output

# Speficy directory for --ddp_dir. It must be an empty dir. For
# example, it is not wise to use $OUTPUT_ROOT/$JOB_NAME, if prior
# results are there
DDP_DIR=$OUTPUT_ROOT/$JOB_NAME/ddp
#
# MANUALLY CREATE DDP_DIR IF NOT EXISTENT, OR CLEAR ALL CONTENTS
# INSIDE IF EXISTENT!! CANNOT DO SO IN THIS SCRIPT, BECAUSE MULTIPLE
# NODES WILL TRY TO CREATE/CLEAR, INTERFERING WITH THE NEXT TOUCH.
#
# Then, under the dir, create one empty file for each node, where the
# file name is the node name
touch $DDP_DIR/$SLURMD_NODENAME

# Run examples. For the full list of options, see driver/parser.py
#
# Turn on --verbose to see timing statistics
#
# 2 node, 2 GPUs/node, must do ddp
python -m driver.main ogbn-arxiv $JOB_NAME \
       --dataset_root $DATASET_ROOT --output_root $OUTPUT_ROOT \
       --trials 2 --epochs 3 --test_epoch_frequency 2 \
       --model_name SAGE --test_type batchwise \
       --overwrite_job_dir \
       --num_workers 30 --max_num_devices_per_node 2 --total_num_nodes 2 \
       --ddp_dir $DDP_DIR
