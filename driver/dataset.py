from typing import Mapping, NamedTuple, Any
from pathlib import Path
import torch
from torch_sparse import SparseTensor
from ogb.nodeproppred import PygNodePropPredDataset

from fast_sampler import to_row_major


def get_sparse_tensor(edge_index, num_nodes=None, return_e_id=False):
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if return_e_id:
            value = torch.arange(adj_t.nnz())
            adj_t = adj_t.set_value(value, layout='coo')
        return adj_t

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    value = torch.arange(edge_index.size(1)) if return_e_id else None
    return SparseTensor(row=edge_index[0], col=edge_index[1],
                        value=value,
                        sparse_sizes=(num_nodes, num_nodes)).t()


class FastDataset(NamedTuple):
    name: str
    x: torch.Tensor
    y: torch.Tensor
    rowptr: torch.Tensor
    col: torch.Tensor
    split_idx: Mapping[str, torch.Tensor]
    meta_info: Mapping[str, Any]

    @classmethod
    def from_ogb(self, name: str, root='dataset'):
        print('Obtaining dataset from ogb...')
        return self.process_ogb(PygNodePropPredDataset(name=name, root=root))

    @classmethod
    def process_ogb(self, dataset):
        print('Converting to fast dataset format...')
        data = dataset.data
        x = to_row_major(data.x).to(torch.float16)
        y = data.y.squeeze()

        if y.is_floating_point():
            y = y.nan_to_num_(-1)
            y = y.long()

        adj_t = get_sparse_tensor(data.edge_index, num_nodes=x.size(0))
        rowptr, col, _ = adj_t.to_symmetric().csr()
        return self(name=dataset.name, x=x, y=y,
                   rowptr=rowptr, col=col,
                   split_idx=dataset.get_idx_split(),
                   meta_info=dataset.meta_info.to_dict())

    @classmethod
    def from_path(self, _path, name):
        path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
        if not (path.exists() and path.is_dir()):
            print(f'First time preprocessing {name}; may take some time...')
            dataset = self.from_ogb(name, root=_path)
            print(f'Saving processed data...')
            dataset.save(_path, name)
            return dataset
        else:
            return self.from_path_if_exists(_path, name)

    @classmethod
    def from_path_if_exists(self, _path, name):
        path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
        assert path.exists() and path.is_dir()
        data = {
            field: torch.load(path.joinpath(field + '.pt'))
            for field in self._fields
        }
        data['y'] = data['y'].long()
        data['x'] = data['x'].to(torch.float16)
        assert data['name'] == name
        return self(**data)

    def save(self, _path, name):
        path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
        # path.mkdir()
        for i, field in enumerate(self._fields):
            torch.save(self[i], path.joinpath(field + '.pt'))

    def adj_t(self):
        num_nodes = self.x.size(0)
        return SparseTensor(rowptr=self.rowptr, col=self.col,
                            sparse_sizes=(num_nodes, num_nodes),
                            is_sorted=True, trust_data=True)

    def share_memory_(self):
        self.x.share_memory_()
        self.y.share_memory_()
        self.rowptr.share_memory_()
        self.col.share_memory_()

        for v in self.split_idx.values():
            v.share_memory_()

    @property
    def num_features(self):
        return self.x.size(1)

    @property
    def num_classes(self):
        return int(self.meta_info['num classes'])
