# Adding a GNN Architecture

The following files are involved with GNN architectures.

`driver/models.py` defines architectures. Note that the `inference()` method inside each GNN class is supposed to perform layer-wise inference, which is less favored compared with batch-wise inference. Hence, it does not need to be implemented. The layer-wise inference code generally differs across architectures (because it needs to match the `forward()` method), while layer-wise inference code is one for all (which has already been implemented).

`driver/main.py` involves architectures in two places. At the top, architectures are imported from `.models`. In the function `get_model_type()`, architectures are enumerated.