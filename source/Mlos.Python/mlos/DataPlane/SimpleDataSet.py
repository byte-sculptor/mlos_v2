#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import pandas as pd

from mlos.DataPlane.Interfaces.DataSet import DataSet
from mlos.Spaces import Hypergrid


class SimpleDataSet(DataSet):
    """Maintains a dataframe and a corresponding Hypergrid in memory.

    """

    def __init__(self, hypergrid: Hypergrid, df: pd.DataFrame):
        ...
