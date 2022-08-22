from __future__ import print_function, absolute_import, division

from dataclasses import dataclass, field, MISSING
from typing import Optional, Tuple, Callable


class Config:
    """ """

    def __init__(self, **kwargs):

        self._from_kwargs(self, kwargs)

    @classmethod
    def from_kwargs(cls, kwargs):
        self = cls()

        cls._from_kwargs(self, kwargs)

        return self

    @staticmethod
    def _from_kwargs(instance, kwargs):
        keys = set(instance.__dataclass_fields__.keys())
        for k, v in kwargs.items():
            if k in keys:
                keys.remove(k)
                setattr(instance, k, v)

    @classmethod
    def new(cls, old, **kwargs):
        config = cls(**old.__dict__)

        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

        return config

    def duplicate(self, **kwargs):
        cls = type(self)
        config = cls(**self.__dict__)

        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

        return config

    def gets(self, keys):
        return {i: self.__dict__[i] for i in keys}


@dataclass(init=False)
class GBPConfig(Config):
    scalar_gate: int = 0
    vector_gate: bool = False
    vector_residual: bool = False

    scalar_act: Optional[Callable] = field(default="relu", metadata={"help": "activation function to use for scalars"})
    vector_act: Optional[Callable] = field(default="", metadata={"help": "activation function to use for vectors"})

    bottleneck: int = 1

    vector_linear: bool = True
    vector_identity: bool = True

    @property
    def activations(self) -> Tuple[Callable, Optional[Callable]]:
        return self.scalar_act, self.vector_act

    @activations.setter
    def activations(self, v: Optional[Callable]) -> None:
        self.scalar_act = v[0]
        self.vector_act = v[1]


@dataclass(init=False)
class CPDFeatures(Config):
    dihedral: bool = True
    orientations: bool = True
    sidechain: bool = True
    relative_distance: bool = True
    relative_position: bool = True
    direction_unit: bool = True


@dataclass(init=False)
class GBPMessagePassingConfig(Config):
    edge_encoder: bool = False
    edge_gate: bool = False
    message_layers: int = 3
    message_residual: int = 0
    message_ff_multiplier: int = 1
    self_message: bool = True


@dataclass(init=False)
class GBPAtomwiseInteractionConfig(Config):
    gbp_mp: GBPMessagePassingConfig = field(default=MISSING, init=True)
    pre_norm: bool = False
    feedforward_layers: int = 2
    drop_rate: float = 0.1


@dataclass(init=False)
class ModelConfig(Config):
    node_scalar: int = 100
    node_vector: int = 16
    edge_scalar: int = 32
    edge_vector: int = 1

    encoder_layers: int = 3
    decoder_layers: int = 3
    dense_dropout: float = 0.1
