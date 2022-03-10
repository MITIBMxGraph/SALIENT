# Setup Instructions on a GPU Machine

### 1. Install Conda

Follow instructions on the [Conda user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). For example, to install Miniconda on a linux machine:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh
```

Or, to install Anaconda:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
```

SALIENT has been tested on Python 3.8.10.

It is highly recommended to create a new environment and do the subsequent steps therein. Example with a new environment called `salient`:

```bash
conda create -n salient python=3.8 -y
conda activate salient
```

### 2. Install PyTorch

Follow instructions on the [PyTorch homepage](https://pytorch.org). For example, to install on a linux machine with CUDA 11.3:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

SALIENT has been tested on PyTorch 1.10.0.

### 3. Install PyTorch-Geometric (PyG)

Follow instructions on the [PyG Github page](https://github.com/pyg-team/pytorch_geometric). For example, with PyTorch >= 1.8.0, it suffices to do the following:

```bash
conda install pyg -c pyg -c conda-forge
```

SALIENT has been tested on PyG 2.0.2.

### 4. Install Latest PyTorch-Sparse

PyTorch-Sparse is usually installed together with PyG. We made a slight modification to support faster data transfer. See this [pull request](https://github.com/rusty1s/pytorch_sparse/pull/195) for detail. Currently, this change has been merged but not yet distributed. Hence, one should uninstall the prior version of PyTorch-Sparse and install the latest one from source:

```bash
pip uninstall torch_sparse
export FORCE_CUDA=1
pip install git+git://github.com/rusty1s/pytorch_sparse.git@master
```

Note: Compilation requires nvcc >= 11.

### 5. Install OGB

```bash
pip install ogb
```

SALIENT has been tested on OGB 1.3.2.

### 6. Install SALIENT's fast_sampler

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

Note: Compilation requires a c++ compiler that supports c++17 (e.g., gcc >= 8).

### 7. Install Other Dependencies

```bash
conda install prettytable -c conda-forge
```

### 8. Try an Example

Congratulations! SALIENT has been installed. The script `example_single_machine.sh` under the folder `examples` contains examples to use SALIENT. Read it with care, edit as appropriate (e.g., set the correct `SALIENT_ROOT` and `DATASET_ROOT`), and run under this folder:

```bash
./example_single_machine.sh
```

It may take several minutes to run all examples therein.

#### 8.1 Tips on Datasets

Create a folder somewhere to store the datasets. Pass the folder path to `--dataset_root`. The first time an OGB dataset is used, it will be automatically downloaded to that folder (which may take some time depending on size).

Alternatively, to pre-download an OGB dataset before trying the examples, start python and type:

```python
>>> name = # type dataset name here, such as 'ogbn-arxiv'
>>> root = # type dataset root here, such as '/Users/username/dataset'
>>> from ogb.nodeproppred import PygNodePropPredDataset
>>> dataset = PygNodePropPredDataset(name=name, root=root)
```

Note: When an OGB dataset is used the first time, SALIENT will process it after downloading and will store the processed data under a `processed` subfolder of that dataset. Subsequent uses of SALIENT will directly load the processed data.

#### 8.2 Tips on Command Line Arguments

To see all command-line arguments of SALIENT, set `PYTHONPATH` to be the root of SALIENT and type

```bash
python -m driver.main --help
```