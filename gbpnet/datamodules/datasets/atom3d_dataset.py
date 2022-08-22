import pandas as pd
import torch
import torch_cluster
import torch_geometric

from gbpnet.datamodules.datasets.helper import _normalize, _rbf

_element_mapping = lambda x: {
    'H': 0,
    'C': 1,
    'N': 2,
    'O': 3,
    'F': 4,
    'S': 5,
    'Cl': 6, 'CL': 6,
    'P': 7
}.get(x, 8)


def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1),
               D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num,
                         (edge_s, edge_v))

    return edge_s, edge_v


class BaseTransform:
    '''
    From https://github.com/drorlab/gvp-pytorch

    '''

    def __init__(self, edge_cutoff=4.5, num_rbf=16, max_num_neighbors=64, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
        self.max_num_neighbors = max_num_neighbors

    def __call__(self, df, edge_index=None):
        with torch.no_grad():
            coords = df[['x', 'y', 'z']].to_numpy()
            coords = torch.as_tensor(coords,
                                     dtype=torch.float32, device=self.device)
            atoms = torch.as_tensor(list(map(_element_mapping, df.element)),
                                    dtype=torch.long, device=self.device)

            edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff,
                                                    max_num_neighbors=self.max_num_neighbors)

            edge_s, edge_v = _edge_features(coords, edge_index,
                                            D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)

            return torch_geometric.data.Data(x=coords, atoms=atoms,
                                             edge_index=edge_index, edge_scalar=edge_s, edge_vector=edge_v)


########################################################################


class LBATransform(BaseTransform):
    '''
    From https://github.com/drorlab/gvp-pytorch
    '''

    def __call__(self, elem, index=-1):
        pocket, ligand = elem['atoms_pocket'], elem['atoms_ligand']
        df = pd.concat([pocket, ligand], ignore_index=True)

        data = super().__call__(df)
        with torch.no_grad():
            data.label = elem['scores']['neglog_aff']
            lig_flag = torch.zeros(df.shape[0], device=self.device, dtype=torch.bool)
            lig_flag[-len(ligand):] = 1
            data.lig_flag = lig_flag
        return data


class PSRTransform(BaseTransform):
    '''
    From https://github.com/drorlab/gvp-pytorch
    '''

    def __call__(self, elem, index=-1):
        df = elem['atoms']
        df = df[df.element != 'H'].reset_index(drop=True)
        data = super().__call__(df, elem.get('edge_index', None))
        data.label = elem['scores']['gdt_ts']
        data.id = eval(elem['id'])[0]
        return data
