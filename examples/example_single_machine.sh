#!/bin/bash

# Script to run examples on a single machine. READ ALL INSTRUCTIONS!!

# Set JOB_NAME used for this script
JOB_NAME=example_single_machine

# Set environment variable SLURMD_NODENAME so that os.environ can get
# it. Usually, on a cluster with SLURM scheduler, the scheduler will
# set this variable. In other cases, manually set it like the
# following:
export SLURMD_NODENAME=`hostname`

# Set SALIENT root and PYTHONPATH
SALIENT_ROOT=$HOME/SALIENT
export PYTHONPATH=$SALIENT_ROOT

# Set the data paths
DATASET_ROOT=$HOME/dataset
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
