# Adding a Dataset

The following files are involved with datasets.

`driver/dataset.py` defines a class `FastDataset`, where the loading of a dataset is handled by the class method `from_path()`. Typically, for an OGB (or PyG) dataset, it is first loaded by the corresponding API. Then, certain processing is done, such as converting the node feature matrix to row-major order and half-precision, and symmetrizing the graph adjacency matrix. To expedite the loading of the dataset in subsequent uses, we store pertinent data in `.pt` files so that subsequent uses do not need to perform the transformation again.

`driver/main.py` uses the class `FastDataset` in the function `get_dataset()`.