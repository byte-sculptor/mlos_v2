#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from mlos.DataPlane.SharedMemoryDataSetInfo import SharedMemoryDataSetInfo

class PredictRequest:
    """Request for the host to produce predictions."""

    def __init__(self, model_id: str, data_set_info: SharedMemoryDataSetInfo):
        self.model_id = model_id
        self.data_set_info = data_set_info



