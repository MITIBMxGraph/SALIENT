# Setup Instructions on the [Satori Cluster](https://mit-satori.github.io)

### 0. Log in

We will use RHEL8 nodes to setup SALIENT and run experiments. Log in `satori-login-001.mit.edu`.

### 1. Install Conda

On the login node, follow instructions on the [Satori user documentation](https://mit-satori.github.io/satori-ai-frameworks.html#install-anaconda) (step 1 therein) to install Conda.

Then, create a Conda environment (for example, call it `salient`):

```bash
conda create -n salient python=3.9 -y
conda activate salient
```

### 2. Check Conda Channels

Check that Conda has the following channels:

```bash
$ conda config --show channels
channels:
  - https://opence.mit.edu
  - https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
  - defaults
  - conda-forge
```

If some channels are missing:

```bash
conda config --prepend channels [missing_channel]
```

### 3. Install PyTorch & CUDA

```bash
conda install pytorch==1.9.0 cudatoolkit=10.2
```

Note: Before `cudatoolkit=11.2` is compatible with the PyTorch available on Satori, we will need to use a lower version and hack some of the subsequent installation steps.

### 4. Log in an Interactive Compute Node

Request a GPU compute node (RHEL8) without exclusive access:

```bash
srun --gres=gpu:1 -N 1 --mem=1T --time 8:00:00 -p sched_system_all_8 --pty /bin/bash
```

or request the full node:

```bash
srun --gres=gpu:4 -N 1 -c 40 --exclusive --mem=1T --time 8:00:00 -p sched_system_all_8 --pty /bin/bash
```

After getting on the node, activate the Conda environment again:

```bash
conda activate salient
```

The subsequent steps are done on the compute node.

### 5. Install PyTorch-Geometric (PyG)

#### 5.1 Load CUDA Module

```bash
module load cuda/11.2
```

Note: This module has the nvcc compiler needed subsequently. The compiler version does not match the cudatoolkit version (see step 3). However, CUDA 10.2 does not come with `cublas_v2.h`, which is needed to compile some of the packages subsequently. Hence, we load CUDA 11.2 instead.

#### 5.2 Install PyG Dependencies

```bash
export FORCE_CUDA=1
pip install git+git://github.com/rusty1s/pytorch_scatter.git@2.0.7
pip install git+git://github.com/rusty1s/pytorch_cluster.git@1.5.9
pip install git+git://github.com/rusty1s/pytorch_spline_conv.git@1.2.1
pip install git+git://github.com/rusty1s/pytorch_sparse.git@master
```

Note: We install from source here (there are no pre-built wheels for PowerPC).

After `pip install`, start python and try to load a package (e.g., `torch-scatter`):

```python
>>> import torch
>>> import torch_scatter
```

An error will occur:

```
RuntimeError: Detected that PyTorch and torch_scatter were compiled with different CUDA versions. PyTorch has CUDA version 10.2 and torch_scatter has CUDA version 11.2. Please reinstall the torch_scatter that matches your PyTorch install.
```

This error is raised by the `__init__.py` file of this package. Open this file and comment the block that raises the error:

```python
    if t_major != major:
        raise RuntimeError(
            f'Detected that PyTorch and torch_scatter were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_scatter has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_scatter that '
            f'matches your PyTorch install.')
```

Do the same hack for all packages `torch_scatter`, `torch_cluster`, `torch_spline_conv`, and `torch_sparse`.

#### 5.3 Install PyG

```bash
pip install torch-geometric
```

### 6. Install OGB

```bash
pip install ogb
```

### 7. Install SALIENT's fast_sampler

Go to the folder `fast_sampler` and install:

```bash
cd fast_sampler
python setup.py install
cd ..
```

To check that it is properly installed, start python and type:

```python
>>> import torch
>>> import fast_sampler
>>> help(fast_sampler)
```

One should see information of the package.

### 8. Install Other Dependencies

```bash
conda install prettytable -c conda-forge
```

### 9. Try Examples

Congratulations! SALIENT has been installed. The folder `examples` contains several example scripts to use SALIENT.

Tips: Create a folder under `/nobackup/` to store the datasets. Pass the folder path to `--dataset_root` in the example scripts. The first time an OGB dataset is used, it will be automatically downloaded to that folder (which may take some time depending on size).

Alternatively, to pre-download an OGB dataset before trying the examples, start python and type:

```python
>>> name = # type dataset name here, such as 'ogbn-arxiv'
>>> root = # type dataset root here, such as '/nobackup/users/username/dataset'
>>> from ogb.nodeproppred import PygNodePropPredDataset
>>> dataset = PygNodePropPredDataset(name=name, root=root)
```

Note: When an OGB dataset is used the first time, SALIENT will process it after downloading and will store the processed data under a `processed` subfolder of that dataset. Subsequent uses of SALIENT will directly load the processed data.

Tips: To see all command-line arguments of SALIENT, set `PYTHONPATH` to be the root of SALIENT and type

```bash
python -m driver.main --help
```

#### 8.1 Interactive Job on One Compute Node

Log in an interactive compute node (RHEL8) with exclusive access (see step 4). Under the folder `examples`, read `example_Satori_interactive.sh` with care, edit as appropriate (e.g., set the correct `SALIENT_ROOT` and `DATASET_ROOT`), and run:

```bash
./example_Satori_interactive.sh
```

#### 8.2 Batch Job on One Compute Node

On a login node, under the folder `examples`, read `example_Satori_batch_1_node.slurm` with care, edit as appropriate (e.g., set the correct `SALIENT_ROOT` and `DATASET_ROOT`), and submit the job:

```bash
sbatch example_Satori_batch_1_node.slurm
```

#### 8.3 Batch Job on Multiple Compute Nodes

On a login node, under the folder `examples`, read `example_Satori_batch_2_nodes.slurm` with care, edit as appropriate (e.g., set the correct `SALIENT_ROOT` and `DATASET_ROOT`), and submit the job:

```bash
sbatch example_Satori_batch_2_nodes.slurm
```
