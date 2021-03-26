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
    @classmethod
    def get_dataframe(cls, data_set_info: SharedMemoryDataSetInfo):
        schema = json.loads(data_set_info.schema_json_str, cls=HypergridJsonDecoder)
        shared_memory = SharedMemory(name=data_set_info.shared_memory_name, create=False)
        shared_memory_np_records_array = np.recarray(
            shape=data_set_info.shared_memory_np_array_shape,
            dtype=data_set_info.shared_memory_np_array_dtype,
            buf=shared_memory.buf
        )
        df = pd.DataFrame.from_records(data=shared_memory_np_records_array, columns=schema.dimension_names, index='index')
        shared_memory.close()
        return df
