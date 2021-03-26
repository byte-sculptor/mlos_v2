#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import json
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd

from mlos.DataPlane.DataSetViewInterface import DataSetViewInterface
from mlos.DataPlane.SharedMemoryDataSetInfo import SharedMemoryDataSetInfo
from mlos.Spaces.HypergridsJsonEncoderDecoder import HypergridJsonDecoder


class SharedMemoryDataSetView(DataSetViewInterface):
    """Allows to view a dataframe stored in shared memory.

    """
    def __init__(self, data_set_info: SharedMemoryDataSetInfo):
        self.data_set_info = data_set_info
        self.schema = json.loads(data_set_info.schema_json_str, cls=HypergridJsonDecoder)
        self._shared_memory = None

    def detach(self):
        if self._shared_memory is not None:
            self._shared_memory.close()
            self._shared_memory = None

    def get_dataframe(self):
        self.detach()
        self._shared_memory = SharedMemory(name=self.data_set_info.shared_memory_name, create=False)
        shared_memory_np_records_array = np.recarray(
            shape=self.data_set_info.shared_memory_np_array_shape,
            dtype=self.data_set_info.shared_memory_np_array_dtype,
            buf=self._shared_memory.buf
        )
        df = pd.DataFrame.from_records(data=shared_memory_np_records_array, columns=self.schema.dimension_names, index='index')
        return df