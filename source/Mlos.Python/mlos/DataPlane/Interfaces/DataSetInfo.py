#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from abc import ABC, abstractmethod
from uuid import UUID

from mlos.Spaces import Hypergrid

class DataSetInfo(ABC):
    """Interface for all DataSetInfo subclasses."""

    @property
    @abstractmethod
    def data_set_id(self) -> UUID:
        raise NotImplementedError

    @property
    @abstractmethod
    def schema(self) -> Hypergrid:
        raise NotImplementedError
