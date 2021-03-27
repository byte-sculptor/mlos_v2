#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from multiprocessing.shared_memory import SharedMemory
import pickle

from mlos.DataPlane.ModelHosting.SharedMemoryBackedModelInfo import SharedMemoryBackedModelInfo
from mlos.Optimizers.RegressionModels.RegressionModel import RegressionModel

class SharedMemoryBackedModelReader:
    """Provides functionality to retrieve an instance of a RegressionModel from shared memory.

    Steps:
        1. Connecting to the shared memory by its id.
        2. Deserializing the model - right now using pickle, but the models could either override the __getstate__, __setstate__ APIs
            or implement an analogous API.
    """

    def __init__(
        self,
        shared_memory_model_info: SharedMemoryBackedModelInfo
    ):
        self.shared_memory_model_info: SharedMemoryBackedModelInfo = shared_memory_model_info
        self._shared_memory: SharedMemory = None

    def detach(self):
        if self._shared_memory is not None:
            self._shared_memory.close()
            self._shared_memory = None

    def get_model(self):
        self.detach()
        self._shared_memory = SharedMemory(name=self.shared_memory_model_info.model_id, create=False)
        model = pickle.loads(self._shared_memory.buf)
        assert isinstance(model, RegressionModel)
        return model
