import functools
from copy import copy
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_scatter import scatter_add, scatter

from gbpnet.configs.gbp_config import GBPAtomwiseInteractionConfig
from gbpnet.configs.gbp_config import GBPConfig, GBPMessagePassingConfig


class ScalarVector(tuple):
    def __new__(cls, scalar, vector):
        return tuple.__new__(cls, (scalar, vector))

    def __getnewargs__(self):
        return self.scalar, self.vector

    @property
    def scalar(self):
        return self[0]

    @property
    def vector(self):
        return self[1]

    # Element-wise addition
    def __add__(self, other):
        if isinstance(other, tuple):
            scalar_other = other[0]
            vector_other = other[1]
        else:
            scalar_other = other.scalar
            vector_other = other.vector

        return ScalarVector(self.scalar + scalar_other, self.vector + vector_other)

    # Element-wise multiplication or scalar multiplication
    def __mul__(self, other):
        if isinstance(other, tuple):
            other = ScalarVector(other[0], other[1])

        if isinstance(other, ScalarVector):
            return ScalarVector(self.scalar * other.scalar, self.vector * other.vector)
        else:
            return ScalarVector(self.scalar * other, self.vector * other)

    def concat(self, others, dim=-1):
        dim %= len(self.scalar.shape)
        s_args, v_args = list(zip(*(self, *others)))
        return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

    def flatten(self):
        flat_vector = torch.reshape(self.vector, self.vector.shape[:-2] + (3 * self.vector.shape[-2],))
        return torch.cat([self.scalar, flat_vector], -1)

    @staticmethod
    def recover(x, vector_dim):
        v = torch.reshape(x[..., -3 * vector_dim:], x.shape[:-1] + (vector_dim, 3))
        s = x[..., : -3 * vector_dim]
        return ScalarVector(s, v)

    def vs(self):
        return self.scalar, self.vector

    def idx(self, idx):
        return ScalarVector(self.scalar[idx], self.vector[idx])

    def repeat(self, n, c=1, y=1):
        return ScalarVector(self.scalar.repeat(n, c), self.vector.repeat(n, y, c))

    def clone(self):
        return ScalarVector(self.scalar.clone(), self.vector.clone())

    def __setitem__(self, key, value):
        self.scalar[key] = value.scalar
        self.vector[key] = value.vector

    def __repr__(self):
        return f'ScalarVector({self.scalar}, {self.vector})'


def safe_norm(x, dim=-1, eps=1e-8, keepdim=False, sqrt=True):
    norm = torch.sum(x ** 2, dim=dim, keepdim=keepdim)
    if sqrt:
        norm = torch.sqrt(norm)
    return norm + eps


def nan_to_identity(activation):
    if activation is None:
        return nn.Identity()
    return activation


def is_identity(activation):
    return activation is None or isinstance(activation, nn.Identity)


class GBP(nn.Module):
    def __init__(
            self,
            input_dims,
            output_dims,
            activations: Tuple[Optional[Callable]] = (F.relu, torch.sigmoid),
            scalar_gate_act: Optional[Callable] = None,
            vector_gate: bool = False,
            scalar_gate: int = 0,
            bottleneck: int = 1,
            vector_residual=False,
    ):
        super(GBP, self).__init__()
        if activations is None:
            activations = (None, None)

        self.scalar_act, self.vector_act = nan_to_identity(activations[0]), nan_to_identity(activations[1])
        self.vector_residual = vector_residual
        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.vector_gate = vector_gate
        self.scalar_gate = scalar_gate
        self.scalar_gate_act = scalar_gate_act if scalar_gate_act else nn.Identity()

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert (
                    self.vector_input_dim % bottleneck == 0
            ), f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({bottleneck})"

            self.hidden_dim = self.vector_input_dim // bottleneck if bottleneck > 1 else max(self.vector_input_dim,
                                                                                             self.vector_output_dim)

            self.vector_down = Linear(self.vector_input_dim, self.hidden_dim, bias=False)
            self.scalar_out = Linear(self.hidden_dim + self.scalar_input_dim, self.scalar_output_dim)

            if self.vector_output_dim:
                self.vector_up = Linear(self.hidden_dim, self.vector_output_dim, bias=False)
                if self.vector_gate:
                    self.vector_out_scale = Linear(self.scalar_output_dim, self.vector_output_dim)
        else:
            self.scalar_out = Linear(self.scalar_input_dim, self.scalar_output_dim)

    def zero_vector(self, scalar_rep):
        return torch.zeros(scalar_rep.size(0), self.vector_output_dim, 3, device=scalar_rep.device)

    def vector_process(self, scalar_rep, v_pre, vector_hidden_rep):
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep += v_pre
        vector_rep = vector_rep.transpose(-1, -2)
        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_act(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_act):
            vector_rep = vector_rep * self.vector_act(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    def forward(self, s_maybev):
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybev
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat([scalar_rep, vector_norm], -1)
        else:
            merged = s_maybev

        scalar_rep = self.scalar_out(merged)

        if self.vector_input_dim and self.vector_output_dim:
            vector_rep = self.vector_process(scalar_rep, v_pre, vector_hidden_rep)

        scalar_rep = self.scalar_act(scalar_rep)
        vector_rep = self.zero_vector(
            scalar_rep) if self.vector_output_dim and not self.vector_input_dim else vector_rep
        return ScalarVector(scalar_rep, vector_rep) if self.vector_output_dim else scalar_rep


def GBPv2(input_dims, output_dims, gbp: GBPConfig = None, **kwargs):
    gbp_dict = copy(gbp.__dict__)
    gbp_dict["activations"] = gbp.activations
    del gbp_dict["scalar_act"]
    del gbp_dict["vector_act"]

    for key in kwargs:
        gbp_dict[key] = kwargs[key]

    return GBP(input_dims, output_dims, **gbp_dict)


class GBPEmbedding(nn.Module):
    def __init__(self, edge_input_dims, node_input_dims, edge_embed_dims, node_embed_dims, n_atoms=9,
                 gbp: GBPConfig = None, pre_norm=True):
        super(GBPEmbedding, self).__init__()
        if n_atoms > 0:
            self.atom_embedding = nn.Embedding(n_atoms, n_atoms)
        else:
            self.atom_embedding = None

        self.pre_norm = pre_norm
        if pre_norm:
            self.edge_norm = LayerNorm(edge_input_dims)
            self.node_norm = LayerNorm(node_input_dims)
        else:
            self.edge_norm = LayerNorm(edge_embed_dims)
            self.node_norm = LayerNorm(node_embed_dims)

        self.edge_embed = GBP(edge_input_dims, edge_embed_dims,
                              activations=(None, None), vector_gate=gbp.vector_gate, scalar_gate=gbp.scalar_gate)

        self.node_embed = GBP(node_input_dims, node_embed_dims,
                              activations=(None, None), vector_gate=gbp.vector_gate, scalar_gate=gbp.scalar_gate)

    def forward(self, batch):
        if self.atom_embedding is not None:
            node_rep = self.atom_embedding(batch.atoms)
        else:
            node_rep = ScalarVector(batch.node_scalar, batch.node_vector)

        edge_rep = ScalarVector(batch.edge_scalar, batch.edge_vector)
        if self.pre_norm:
            edge_rep = self.edge_norm(edge_rep)
            node_rep = self.node_norm(node_rep)

        node_rep = self.node_embed(node_rep)
        edge_rep = self.edge_embed(edge_rep)

        if not self.pre_norm:
            edge_rep = self.edge_norm(edge_rep)
            node_rep = self.node_norm(node_rep)

        return node_rep, edge_rep


class GBPMessagePassing(nn.Module):
    def __init__(
            self,
            input_dims,
            output_dims,
            edge_dims,
            aggr="mean",
            gbp: GBPConfig = None,
            gbp_mp: GBPMessagePassingConfig = None,
    ):
        super().__init__()
        self.aggr = aggr
        self.scalar_input_dim, self.vector_input_dum = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.edge_scalar_dim, self.edge_vector_dim = edge_dims
        self.self_message = gbp_mp.self_message

        in_scalar = 2 * self.scalar_input_dim + self.edge_scalar_dim
        in_vector = 2 * self.vector_input_dum + self.edge_vector_dim

        soft_gbp = gbp.duplicate(bottleneck=GBPConfig.bottleneck, vector_residual=GBPConfig.vector_residual)
        GBP_config1 = functools.partial(GBPv2, gbp=soft_gbp)
        GBP_config2 = functools.partial(GBPv2, gbp=gbp)

        self.gbp_conv = gbp_mp

        module_list = [
            GBP_config1(
                (in_scalar, in_vector),
                output_dims,
                activations=gbp.activations if gbp_mp.message_layers > 1 else None,
            )
        ]

        for i in range(gbp_mp.message_layers - 2):
            module_list.append(GBP_config2(output_dims, output_dims))

        if gbp_mp.message_layers > 1:
            module_list.append(GBP_config1(output_dims, output_dims, activations=(None, None)))

        self.atom_message_fusion = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        j, i = edge_index
        vector = x.vector.reshape(x.vector.shape[0], -1)
        x_reshaped = ScalarVector(x.scalar, vector)

        s_i, v_i = x_reshaped.idx(i)
        s_j, v_j = x_reshaped.idx(j)

        v_j = v_j.view(v_j.size(0), v_j.size(1) // 3, 3)
        v_i = v_i.view(v_i.size(0), v_i.size(1) // 3, 3)
        message = ScalarVector(s_j, v_j).concat((edge_attr, ScalarVector(s_i, v_i)))
        message = self.atom_message_fusion(message)
        message = message.flatten()

        message = scatter(message, i, dim=0, reduce=self.aggr, dim_size=x.scalar.size(0))

        return ScalarVector.recover(message, self.vector_output_dim)


class GBPAtomwiseInteraction(nn.Module):
    def __init__(
            self,
            node_dims,
            edge_dims,
            dropout=0.1,
            autoregressive=False,
            ff_activations=None,
            gbp: GBPConfig = None,
            gbp_int_layer: GBPAtomwiseInteractionConfig = None,
    ):

        super(GBPAtomwiseInteraction, self).__init__()

        message_function = GBPMessagePassing

        if ff_activations is None:
            ff_activations = gbp.activations

        self.pre_norm = gbp_int_layer.pre_norm
        reduce_function = "mean"
        if autoregressive:
            reduce_function = "add"
        self.interaction = message_function(
            node_dims,
            node_dims,
            edge_dims,
            aggr=reduce_function,
            gbp=gbp,
            gbp_mp=gbp_int_layer.gbp_mp,
        )

        ff_gbp = gbp.duplicate()
        ff_gbp.activations = ff_activations
        ff_without_res = gbp.duplicate(vector_residual=False)

        GBP_config_for_ff = functools.partial(GBPv2, gbp=ff_gbp)
        GBP_ff_without_res = functools.partial(GBPv2, gbp=ff_without_res)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.sv_dropout = nn.ModuleList([Dropout(dropout) for _ in range(2)])

        feedforward_network = []
        hidden_dims = 4 * node_dims.scalar, 2 * node_dims.vector

        if gbp_int_layer.feedforward_layers == 1:
            hidden_dims = node_dims

        feedforward_network.append(GBP_ff_without_res(node_dims, hidden_dims,
                                                      activations=None if gbp_int_layer.feedforward_layers == 1 else gbp.activations))

        inter_layers = [GBP_config_for_ff(hidden_dims, hidden_dims) for _ in
                        range(gbp_int_layer.feedforward_layers - 2)]
        feedforward_network.extend(inter_layers)

        if gbp_int_layer.feedforward_layers > 1:
            feedforward_network.append(GBP_ff_without_res(hidden_dims, node_dims, activations=(None, None)))
        self.feedforward_network = nn.Sequential(*feedforward_network)

    def auto_regressive(self, x, edge_index, edge_attr, autoregressive_x):
        autoregressive_x = ScalarVector(autoregressive_x[0], autoregressive_x[1])
        i, j = edge_index
        mask = i < j
        edge_index_forward = edge_index[:, mask]
        edge_index_backward = edge_index[:, ~mask]
        edge_attr_forward = edge_attr.idx(mask)
        edge_attr_backward = edge_attr.idx(~mask)

        autoregressive_mp = self.interaction(x, edge_index_forward, edge_attr_forward) \
                            + self.interaction(autoregressive_x, edge_index_backward, edge_attr_backward)

        count = scatter_add(torch.ones_like(j), j, dim_size=autoregressive_mp[0].size(0)).clamp(min=1).unsqueeze(-1)

        autoregressive_mp = ScalarVector(autoregressive_mp[0] / count, autoregressive_mp[1] / count.unsqueeze(-1))

        return autoregressive_mp

    def forward(self, sv, edge_index, edge_attr, sv_regressive=None, node_mask=None):

        sv = ScalarVector(sv[0], sv[1])
        edge_attr = ScalarVector(edge_attr[0], edge_attr[1])

        if self.pre_norm:
            sv = self.norm[0](sv)

        if sv_regressive is not None:
            residual_hidden = self.auto_regressive(sv, edge_index, edge_attr, sv_regressive)
        else:
            residual_hidden = self.interaction(sv, edge_index, edge_attr)

        if node_mask is not None:
            sv_res = sv
            sv, residual_hidden = sv.idx(node_mask), residual_hidden.idx(node_mask)

        sv = sv + self.sv_dropout[0](residual_hidden)

        if self.pre_norm:
            sv = self.norm[1](sv)
        else:
            sv = self.norm[0](sv)

        residual_hidden = self.feedforward_network(sv)

        sv = sv + self.sv_dropout[1](residual_hidden)

        if not self.pre_norm:
            sv = self.norm[1](sv)

        if node_mask is not None:
            sv_res[0][node_mask], sv_res[1][node_mask] = sv[0], sv[1]
            sv = sv_res
        return sv


class VectorDropout(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate):
        super(VectorDropout, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = x[0].device
        if not self.training:
            return x
        mask = torch.bernoulli((1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.scalar_dropout = nn.Dropout(drop_rate)
        self.vector_dropout = VectorDropout(drop_rate)

    def forward(self, x):
        if type(x) is torch.Tensor:
            return self.scalar_dropout(x)
        return ScalarVector(self.scalar_dropout(x[0]), self.vector_dropout(x[1]))


class LayerNorm(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.scalar_dims, self.vector_dims = dims
        self.scalar_norm = nn.LayerNorm(self.scalar_dims)

    def forward(self, x):
        if not self.vector_dims:
            return self.scalar_norm(x)
        s, v = x
        vector_norm = torch.clamp(torch.sum(torch.square(v), -1, True), min=1e-8)
        vector_norm = torch.sqrt(torch.mean(vector_norm, dim=-2, keepdim=True))
        return ScalarVector(self.scalar_norm(s), v / vector_norm)
