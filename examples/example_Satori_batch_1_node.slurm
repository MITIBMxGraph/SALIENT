#!/bin/bash
#
#SBATCH -J example_Satori_batch_1_node
#SBATCH -o SALIENT/job_output/%x_%j.err
#SBATCH -e SALIENT/job_output/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=1T
#SBATCH --exclusive
#SBATCH --time 00:10:00
#SBATCH -p sched_system_all_8

# Script to run examples in the batch mode on Satori (1 node). READ
# ALL INSTRUCTIONS!!
#
# Submit the job using
#
# $ sbatch SALIENT/examples/example_Satori_batch_1_node.slurm
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

# Run examples. For the full list of options, see driver/parser.py
#
# Turn on --verbose to see timing statistics
#
# 1 node, 1 GPU, no ddp
echo 'Example 1'
echo ''
python -m driver.main ogbn-arxiv $JOB_NAME \
       --dataset_root $DATASET_ROOT --output_root $OUTPUT_ROOT \
       --trials 2 --epochs 3 --test_epoch_frequency 2 \
       --model_name SAGE --test_type batchwise \
       --overwrite_job_dir \
       --num_workers 30 --max_num_devices_per_node 1 --total_num_nodes 1
echo ''
echo '======================================================================'
echo ''
echo 'Example 2'
echo ''
#
# 1 node, 1 GPU, ddp
python -m driver.main ogbn-arxiv $JOB_NAME \
       --dataset_root $DATASET_ROOT --output_root $OUTPUT_ROOT \
       --trials 2 --epochs 3 --test_epoch_frequency 2 \
       --model_name SAGE --test_type batchwise \
       --overwrite_job_dir \
       --num_workers 30 --max_num_devices_per_node 1 --total_num_nodes 1 \
       --one_node_ddp
echo ''
echo '======================================================================'
echo ''
echo 'Example 3'
echo ''
#
# 1 node, 2 GPUs, must do ddp
python -m driver.main ogbn-arxiv $JOB_NAME \
       --dataset_root $DATASET_ROOT --output_root $OUTPUT_ROOT \
       --trials 2 --epochs 3 --test_epoch_frequency 2 \
       --model_name SAGE --test_type batchwise \
       --overwrite_job_dir \
       --num_workers 30 --max_num_devices_per_node 2 --total_num_nodes 1 \
       --one_node_ddp
echo ''
echo '======================================================================'
echo ''
echo 'Example 4'
echo ''
#
# only run inference
python -m driver.main ogbn-arxiv $JOB_NAME \
       --dataset_root $DATASET_ROOT --output_root $OUTPUT_ROOT \
       --trials 1 --do_test_run \
       --do_test_run_filename $OUTPUT_ROOT/$JOB_NAME/model_0_3.pt \
       $OUTPUT_ROOT/$JOB_NAME/model_1_3.pt \
       --model_name SAGE --test_type batchwise \
       --num_workers 30 --max_num_devices_per_node 2 --total_num_nodes 1 \
       --one_node_ddp
echo ''
echo '======================================================================'
echo ''
echo 'Example 5'
echo ''
#
# ogbn-products
python -m driver.main ogbn-products $JOB_NAME \
       --dataset_root $DATASET_ROOT --output_root $OUTPUT_ROOT \
       --trials 2 --epochs 3 --test_epoch_frequency 2 \
       --model_name SAGE --test_type batchwise \
       --overwrite_job_dir \
       --num_workers 30 --max_num_devices_per_node 2 --total_num_nodes 1 \
       --one_node_ddp
echo ''
echo '======================================================================'
echo ''
echo 'Example 6'
echo ''
#
# GIN
python -m driver.main ogbn-arxiv $JOB_NAME \
       --dataset_root $DATASET_ROOT --output_root $OUTPUT_ROOT \
       --trials 2 --epochs 3 --test_epoch_frequency 2 \
       --model_name GIN --test_type batchwise \
       --overwrite_job_dir \
       --num_workers 30 --max_num_devices_per_node 2 --total_num_nodes 1 \
       --one_node_ddp
echo ''
echo '======================================================================'
echo ''
echo 'Example 7'
echo ''
#
# GAT
python -m driver.main ogbn-arxiv $JOB_NAME \
       --dataset_root $DATASET_ROOT --output_root $OUTPUT_ROOT \
       --trials 2 --epochs 3 --test_epoch_frequency 2 \
       --model_name GAT --test_type batchwise \
       --overwrite_job_dir \
       --num_workers 30 --max_num_devices_per_node 2 --total_num_nodes 1 \
       --one_node_ddp
echo ''
echo '======================================================================'
echo ''
echo 'Example 8'
echo ''
#
# SAGEResInception
python -m driver.main ogbn-arxiv $JOB_NAME \
       --dataset_root $DATASET_ROOT --output_root $OUTPUT_ROOT \
       --trials 2 --epochs 3 --test_epoch_frequency 2 \
       --model_name SAGEResInception --test_type batchwise \
       --overwrite_job_dir \
       --num_workers 30 --max_num_devices_per_node 2 --total_num_nodes 1 \
       --one_node_ddp
echo ''
echo '======================================================================'
echo ''
echo 'Example 9'
echo ''
#
# layerwise test
python -m driver.main ogbn-arxiv $JOB_NAME \
       --dataset_root $DATASET_ROOT --output_root $OUTPUT_ROOT \
       --trials 2 --epochs 3 --test_epoch_frequency 2 \
       --model_name SAGE --test_type layerwise \
       --overwrite_job_dir \
       --num_workers 30 --max_num_devices_per_node 2 --total_num_nodes 1 \
       --one_node_ddp
