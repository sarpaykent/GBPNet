from functools import partial

import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
from torch_scatter import scatter

from gbpnet.configs.gbp_config import GBPConfig, ModelConfig, GBPAtomwiseInteractionConfig
from gbpnet.models.modules.gbp_release import GBP, GBPAtomwiseInteraction, ScalarVector, GBPEmbedding, LayerNorm


class GBPAtom3DModel(pl.LightningModule):
    def __init__(self,
                 task,
                 num_rbf=16, learning_rate=1e-4,
                 dropout=0.1, dense_dropout=0.1,
                 model: ModelConfig = None,
                 gbp: GBPConfig = None,
                 gbp_int_layer: GBPAtomwiseInteractionConfig = None,
                 **kwargs):

        super().__init__()

        self.task = task
        self.label_names = None
        self.loss_means = None
        self.loss_std = None

        self.loss_fn = nn.MSELoss()

        self.node_dims = ScalarVector(model.node_scalar, model.node_vector)
        self.edge_dims = ScalarVector(model.edge_scalar, model.edge_vector)
        edge_input_dims = ScalarVector(num_rbf, 1)
        node_input_dims = ScalarVector(9, 0)
        output_scale_factor = 2
        output_size = 1
        self.save_hyperparameters()

        metrics = [partial(torchmetrics.regression.mse.MeanSquaredError, squared=False)]
        for phase in ['train', 'valid', 'test']:
            for k, v in metrics:
                setattr(self, 'metric_' + k + '_' + phase, v())

        self.metrics = {phase: nn.ModuleDict(
            {k: getattr(self, 'metric_' + k + '_' + phase) for k, v in metrics}
        ) for phase in ['train', 'valid', 'test']}

        self.embedder = GBPEmbedding(edge_input_dims, node_input_dims, self.edge_dims, self.node_dims, n_atoms=9,
                                     gbp=gbp)

        self.interactions = nn.ModuleList(
            GBPAtomwiseInteraction(self.node_dims, self.edge_dims, dropout=dropout, gbp=gbp,
                                   gbp_int_layer=gbp_int_layer)
            for _ in range(model.encoder_layers))

        self.invariant_projection = nn.Sequential(
            LayerNorm(self.node_dims),
            GBP(self.node_dims, (self.node_dims.scalar, 0),
                activations=gbp.activations, vector_gate=gbp.vector_gate, scalar_gate=gbp.scalar_gate)
        )
        self.invariant_projection_dropout = nn.Dropout(self.hparams.dropout)
        self.dense = nn.Sequential(
            nn.Linear(self.node_dims.scalar, output_scale_factor * self.node_dims.scalar),
            nn.ReLU(inplace=True),
            nn.Dropout(model.dense_dropout),
            nn.Linear(output_scale_factor * self.node_dims.scalar, output_size)
        )

    @staticmethod
    def get_label(batch):
        if type(batch) in [list, tuple]: batch = batch[0]
        return batch.label

    @staticmethod
    def _get_num_graphs(batch):
        if type(batch) in [list, tuple]: batch = batch[0]
        return batch.num_graphs

    def forward(self, batch):
        node_rep, edge_rep = self.embedder(batch)

        for inter in self.interactions:
            node_rep = inter(node_rep, batch.edge_index, edge_rep)

        out = self.invariant_projection(node_rep)
        out = scatter(out, batch.batch, dim=0, reduce='mean')
        out = self.dense(out).squeeze(-1)
        return out

    def loop(self, batch, batch_idx, phase="train", dataloader_idx=0):
        label = self.get_label(batch)
        pred = self(batch)

        pred_metric = getattr(self, 'metric_pred_' + phase, None)
        label_metric = getattr(self, 'metric_label_' + phase, None)
        id_metric = getattr(self, 'metric_id_' + phase, None)
        if pred_metric:
            pred_metric.update(pred)
        if label_metric:
            label_metric.update(label)
        if id_metric:
            batch_ids = []
            for id in batch.id:
                batch_ids.append(float(id.replace('T', '-')))
            id_metric.update(batch_ids)

        loss = self.loss_fn(pred, label)

        for metric in self.metrics[phase].keys():
            pred = pred.detach()

            self.metrics[phase][metric](pred, label)
            self.log(
                f"{phase}/" + metric,
                self.metrics[phase][metric],
                metric_attribute=self.metrics[phase][metric],
                on_step=True,
                on_epoch=True,
                batch_size=self._get_num_graphs(batch),
            )

        self.log(f"{phase}/loss", loss, batch_size=self._get_num_graphs(batch))

        return loss, pred, label

    def test_step(self, batch, batch_idx):
        loss, pred, label = self.loop(batch, batch_idx, 'test')
        return loss
