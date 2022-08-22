from __future__ import print_function, absolute_import, division

import math

import numpy as np
import torch
import torch_cluster
import torch_geometric
from torch.nn import functional as F
from torch.utils import data as data

from gbpnet.configs.gbp_config import CPDFeatures
from gbpnet.datamodules.datasets.helper import _normalize, _rbf


class ProteinGraphDataset(data.Dataset):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, data_list, num_positional_embeddings=16, top_k=30, num_rbf=16, features: CPDFeatures = None,
                 device="cpu"):

        super(ProteinGraphDataset, self).__init__()

        if features is None:
            features = CPDFeatures()

        self.features = features
        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(e["seq"]) for e in data_list]

        self.letter_to_num = {
            "C": 4,
            "D": 3,
            "S": 15,
            "Q": 5,
            "K": 11,
            "I": 9,
            "P": 14,
            "T": 16,
            "F": 13,
            "A": 0,
            "G": 7,
            "H": 8,
            "E": 6,
            "L": 10,
            "R": 1,
            "W": 17,
            "V": 19,
            "N": 2,
            "Y": 18,
            "M": 12,
        }
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}
        self.num_to_letter_list = [None] * 20
        for k in self.letter_to_num:
            self.num_to_letter_list[self.letter_to_num[k]] = k

    @staticmethod
    def num_to_letter():
        letter_to_num = {
            "C": 4,
            "D": 3,
            "S": 15,
            "Q": 5,
            "K": 11,
            "I": 9,
            "P": 14,
            "T": 16,
            "F": 13,
            "A": 0,
            "G": 7,
            "H": 8,
            "E": 6,
            "L": 10,
            "R": 1,
            "W": 17,
            "V": 19,
            "N": 2,
            "Y": 18,
            "M": 12,
        }
        num_to_letter = {v: k for k, v in letter_to_num.items()}
        num_to_letter_list = [None] * 20
        for k in letter_to_num:
            num_to_letter_list[letter_to_num[k]] = k
        return num_to_letter_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self._featurize_as_graph(self.data_list[i])

    def _featurize_as_graph(self, protein):
        if "name" not in protein:
            name = protein["id"]
        else:
            name = protein["name"]
        with torch.no_grad():
            coords = torch.as_tensor(protein["coords"], device=self.device, dtype=torch.float32)
            seq = torch.as_tensor([self.letter_to_num[a] for a in protein["seq"]], device=self.device, dtype=torch.long)

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            dihedrals = self._dihedrals(coords)
            if not self.features.dihedral:
                dihedrals = torch.zeros_like(dihedrals, device=self.device)
            orientations = self._orientations(X_ca)
            if not self.features.orientations:
                orientations = torch.zeros_like(orientations, device=self.device)
            sidechains = self._sidechains(coords)
            if not self.features.sidechain:
                sidechains = torch.zeros_like(sidechains, device=self.device)

            if not self.features.relative_distance:
                rbf = torch.zeros_like(rbf, device=self.device)
            if not self.features.relative_position:
                pos_embeddings = torch.zeros_like(pos_embeddings, device=self.device)
            if not self.features.direction_unit:
                E_vectors = torch.zeros_like(E_vectors, device=self.device)

            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            # print(node_s.shape, node_v.shape)
            # print(edge_s.shape, edge_v.shape)
            # exit()

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))

        data = torch_geometric.data.Data(
            x=X_ca,
            seq=seq,
            name=name,
            node_scalar=node_s,
            node_vector=node_v,
            edge_scalar=edge_s,
            edge_vector=edge_v,
            edge_index=edge_index,
            mask=mask,
        )
        return data

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index, num_embeddings=None, period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec
