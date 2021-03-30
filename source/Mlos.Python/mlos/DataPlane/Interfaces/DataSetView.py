#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from abc import ABC, abstractmethod
import pandas as pd

from mlos.DataPlane.Interfaces.DataSetInfo import DataSetInfo
from mlos.Spaces import Hypergrid


class DataSetView(ABC):
    """"""

    @property
    @abstractmethod
    def schema(self) -> Hypergrid:
        raise NotImplementedError

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_data_set_info(self) -> DataSetInfo:
        raise NotImplementedError
