import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningDataModule
from torch.distributions import Categorical

from gbpnet.configs.gbp_config import GBPConfig, ModelConfig, GBPAtomwiseInteractionConfig
from gbpnet.models.modules.gbp_release import GBP, GBPAtomwiseInteraction, GBPEmbedding, \
    ScalarVector


class GBPCPDModel(pl.LightningModule):

    def __init__(
            self,
            node_input_dims: int,
            edge_input_dims: int,
            dropout: float = 0.1,
            model: ModelConfig = None,
            gbp: GBPConfig = None,
            gbp_int_layer: GBPAtomwiseInteractionConfig = None,
            log_perplexity: bool = True,
            test_recovery: bool = True,
            **kwargs,
    ):
        super().__init__()

        self.log_perplexity = log_perplexity

        self.test_recovery = test_recovery
        self.recovery_metrics = []
        self.subsets = {}

        if model is None:
            raise ValueError("Model configuration not provided")

        if gbp is None:
            raise ValueError("GBP configuration not provided")

        if gbp_int_layer is None:
            raise ValueError("GBP-Conv configuration not provided")

        self.metrics = dict()

        self.save_hyperparameters()

        self.node_dims = (model.node_scalar, model.node_vector)
        self.edge_dims = (model.edge_scalar, model.edge_vector)

        self.loss_fn = nn.CrossEntropyLoss()

        self.atom_embedding = nn.Embedding(20, 20)
        self.embedder = GBPEmbedding(edge_input_dims, node_input_dims, self.edge_dims, self.node_dims, n_atoms=0,
                                     gbp=gbp, pre_norm=False)

        self.encoder_layers = nn.ModuleList(
            GBPAtomwiseInteraction(self.node_dims, self.edge_dims, dropout=dropout, gbp=gbp,
                                   gbp_int_layer=gbp_int_layer)
            for _ in range(model.encoder_layers))

        edge_hidden = (self.edge_dims[0] + 20, self.edge_dims[1])

        self.decoder_layers = nn.ModuleList(
            GBPAtomwiseInteraction(self.node_dims, edge_hidden,
                                   dropout=dropout, autoregressive=True,
                                   gbp=gbp, gbp_int_layer=gbp_int_layer)
            for _ in range(model.decoder_layers))

        self.invariant_projection = GBP(self.node_dims, (20, 0), activations=(None, None))

    def forward(self, batch):
        node_rep, edge_rep = self.embedder(batch)

        for layer in self.encoder_layers:
            node_rep = layer(node_rep, batch.edge_index, edge_rep)

        encoder_embeddings = node_rep

        sequence_embedding = self.atom_embedding(batch.seq)
        sequence_embedding = sequence_embedding[batch.edge_index[0]]
        sequence_embedding[batch.edge_index[0] >= batch.edge_index[1]] = 0
        edge_rep = (torch.cat([edge_rep.scalar, sequence_embedding], dim=-1), edge_rep.vector)

        for layer in self.decoder_layers:
            node_rep = layer(node_rep, batch.edge_index, edge_rep, sv_regressive=encoder_embeddings)

        output = self.invariant_projection(node_rep)

        return output

    def sample(self, node_rep, edge_index, edge_rep, n_samples, temperature=0.1):
        num_nodes = node_rep[0].size(0)
        with torch.no_grad():
            node_rep = self.embedder.node_embed(node_rep)
            node_rep = self.embedder.node_norm(node_rep)
            edge_rep = self.embedder.edge_embed(edge_rep)
            edge_rep = self.embedder.edge_norm(edge_rep)

            for layer in self.encoder_layers:
                node_rep = layer(node_rep, edge_index, edge_rep)

            node_rep = node_rep.repeat(n_samples, 1, 1)
            edge_rep = edge_rep.repeat(n_samples, 1, 1)

            edge_index = edge_index.expand(n_samples, -1, -1)
            offset = num_nodes * torch.arange(n_samples, device=self.device).view(-1, 1, 1)
            edge_index = torch.cat(tuple(edge_index + offset), dim=-1)

            residue_sequence = torch.zeros(n_samples * num_nodes, device=self.device, dtype=torch.int)
            sequence_embedding = torch.zeros(n_samples * num_nodes, 20, device=self.device)

            node_rep_cache = [node_rep.clone() for _ in self.decoder_layers]

            for i in range(num_nodes):

                sequence_embedding_ = sequence_embedding[edge_index[0]]
                sequence_embedding_[edge_index[0] >= edge_index[1]] = 0
                edge_rep_masked = ScalarVector(torch.cat([edge_rep[0], sequence_embedding_], dim=-1), edge_rep[1])

                edge_mask = edge_index[1] % num_nodes == i
                edge_index_ = edge_index[:, edge_mask]
                edge_rep_masked = edge_rep_masked.idx(edge_mask)
                node_mask = torch.zeros(n_samples * num_nodes, device=self.device, dtype=torch.bool)
                node_mask[i::num_nodes] = True

                for j, layer in enumerate(self.decoder_layers):
                    out = layer(node_rep_cache[j], edge_index_, edge_rep_masked,
                                sv_regressive=node_rep_cache[0], node_mask=node_mask)

                    out = out.idx(node_mask)

                    if j < len(self.decoder_layers) - 1:
                        node_rep_cache[j + 1].scalar[i::num_nodes] = out.scalar
                        node_rep_cache[j + 1].vector[i::num_nodes] = out.vector

                logits = self.invariant_projection(out)
                residue_sequence[i::num_nodes] = Categorical(logits=logits / temperature).sample()
                sequence_embedding[i::num_nodes] = self.atom_embedding(residue_sequence[i::num_nodes])

            return residue_sequence.view(n_samples, num_nodes)

    def recovery_from_protein(self, protein):
        node_rep = (protein.node_scalar, protein.node_vector)
        edge_rep = (protein.edge_scalar, protein.edge_vector)
        sample = self.sample(node_rep, protein.edge_index,
                             edge_rep, n_samples=100)
        recovery_ = sample.eq(protein.seq).float().mean()
        return recovery_

    def recovery_test_samples(self, samples):
        for protein in samples:
            recovery_ = self.recovery_from_protein(protein)
            self.recovery_metrics['all'].update(recovery_)
            for subset in self.subsets:
                if protein.name in self.subsets[subset]:
                    self.recovery_metrics[subset].update(recovery_)

    def load_splits(self):
        if getattr(self.trainer.datamodule, 'custom_splits', None) is not None:
            self.subsets = self.trainer.datamodule.custom_splits

    def on_test_start(self):
        self.load_splits()
        metric_keys = list(self.subsets.keys()) + ['all']
        self.recovery_metrics = nn.ModuleDict(
            {key: torchmetrics.CatMetric() for key in metric_keys}
        )

    def calculate_recovery_metrics(self):
        metric_keys = list(self.subsets.keys()) + ['all']
        output = {}
        for key in metric_keys:
            out = self.recovery_metrics[key].compute()
            recovery = torch.median(torch.tensor(out))
            output[key] = recovery
        return output

    def on_test_epoch_end(self):
        super().on_test_epoch_end()

        output = self.calculate_recovery_metrics()
        for key in output:
            self.log(f"test/recovery/" + key, output[key])

    def recovery(self, datamodule: LightningDataModule):
        self.on_test_start()
        self.eval()
        dataset = datamodule.testset
        self.recovery_test_samples(dataset)
        data = self.calculate_recovery_metrics()
        print(f'Test recovery: {data}')

    def loop(self, batch, batch_idx, phase="train", dataloader_idx=0):
        logits = self(batch)
        logits, label = logits[batch.mask], batch.seq[batch.mask]
        loss = self.loss_fn(logits, label)

        self.log(f"{phase}/loss", loss, batch_size=batch.num_graphs)
        if self.log_perplexity:
            self.log(f"{phase}/perplexity", torch.exp(loss), batch_size=batch.num_graphs)
        if phase == 'test' and dataloader_idx == 0 and self.test_recovery:
            self.recovery_test_samples(batch.to_data_list())

        for metric in self.metrics:
            self.metrics[metric](logits, label)
            self.log(
                f"{phase}/" + metric,
                self.metrics[metric],
                metric_attribute=self.metrics[metric],
                on_step=True,
                on_epoch=True,
                batch_size=batch.num_graphs,
            )

        return loss, logits, label

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, pred, label = self.loop(batch, batch_idx, "train", dataloader_idx=dataloader_idx)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, pred, label = self.loop(batch, batch_idx, "test", dataloader_idx=dataloader_idx)
        return loss
