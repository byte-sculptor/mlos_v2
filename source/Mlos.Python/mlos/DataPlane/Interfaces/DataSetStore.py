#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from abc import ABC, abstractmethod

import pandas as pd

from mlos.DataPlane.Interfaces.DataSetInfo import DataSetInfo
from mlos.DataPlane.Interfaces.DataSet import DataSet
from mlos.DataPlane.Interfaces.DataSetView import DataSetView

class DataSetStore(ABC):
    """An interface to be implemented by all DataStores.

    """
    @abstractmethod
    def create_data_set(self, data_set_info: DataSetInfo, df: pd.DataFrame) -> DataSet:
        raise NotImplementedError

    @abstractmethod
    def add_data_set(self, data_set: DataSet) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_data_set(self, data_set_info: DataSetInfo) -> DataSet:
        raise NotImplementedError

    @abstractmethod
    def get_data_set_view(self, data_set_info: DataSetInfo) -> DataSetView:
        raise NotImplementedError

    @abstractmethod
    def detach_data_set(self, data_set_info: DataSetInfo) -> None:
        raise NotImplementedError
