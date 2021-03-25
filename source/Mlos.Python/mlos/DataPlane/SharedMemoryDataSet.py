#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#

import pandas as pd
from mlos.DataPlane.DataSetInterface import DataSetInterface
from mlos.Spaces import Hypergrid


class SharedMemoryDataSet(DataSetInterface):
    """Maintains a dataframe and its associated metadata in shared memory.

    """

    def __init__(self, schema: Hypergrid,):
        self.schema = schema
        self._shared_memory_name = None
        self._shared_memory = None
        self._source_df: pd.DataFrame = None

    def set_dataframe(self, df: pd.DataFrame):
        assert df in self.schema
        self._source_df = df

        # Now let's put it in shared memory. We can attempt two ways:
        # 1. using the to_numpy call on the entire df - causes a few copies
        # 2. trying the to_numpy call on each column separately - not tested yet, but should result in fewer copies down the road
        #



