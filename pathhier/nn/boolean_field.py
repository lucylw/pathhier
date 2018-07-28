from typing import Dict, TypeVar
import logging

import torch

from overrides import overrides
import numpy

from allennlp.data.fields.field import Field

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DataArray = TypeVar("DataArray", torch.Tensor, Dict[str, torch.Tensor])  # pylint: disable=invalid-name


class BooleanField(Field[numpy.ndarray]):
    """
    A ``BooleanField`` is a boolean label of some kind, where the labels are either True or False.
    Parameters
    ----------
    bool_label : ``Union[str, int]``.
    """
    def __init__(self,
                 label: bool):
        self.label = label

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self,
                  padding_lengths: Dict[str, int],
                  cuda_device=-1) -> DataArray:  # pylint: disable=unused-argument
        tensor = torch.Tensor([self.label])
        return tensor if cuda_device == -1 else tensor.cuda(cuda_device)

    @overrides
    def empty_field(self):
        return BooleanField(0)