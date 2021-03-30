#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from abc import ABC, abstractmethod
import pandas as pd

from mlos.DataPlane.Interfaces.DataSetInfo import DataSetInfo
from mlos.DataPlane.Interfaces.DataSetView import DataSetView
from mlos.Spaces import Hypergrid

class DataSet(DataSetView):
    """"""

    @abstractmethod
    def set_dataframe(self, df: pd.DataFrame) -> None:
        raise NotImplementedError
