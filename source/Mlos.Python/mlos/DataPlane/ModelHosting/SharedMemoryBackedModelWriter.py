#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from multiprocessing.shared_memory import SharedMemory
import pickle

from mlos.DataPlane.ModelHosting.SharedMemoryBackedModelInfo import SharedMemoryBackedModelInfo
from mlos.Optimizers.RegressionModels.RegressionModel import RegressionModel


class SharedMemoryBackedModelWriter:
    """Provides functionality to store an instance of a RegressionModel in shared memory.

    Similarly to SharedMemoryDataSet this class wraps all the mechanics required to put the model in shared memory, that is:
        1. Serialize the model - right now we are using pickle, but we could allow the models to expose some minimal state required
            to reconstruct them.
        2. Create an appropriately sized shared memory block.
        3. Place the model in shared memory.

    As well as restoring the model from shared memory:
        1. Connecting to the shared memory by its id.
        2. Deserializing the model - rigth now using pickle, but the models could either override the __getstate__, __setstate__ APIs
            or implement an analogous API.

    We separate the shared memory model reader and shared memory model writer, because only the writer has the right to modify the model,
    and everyone else's access is read-only.

    TODO: we should also implement a common interface for model stores so that model can be backed by either memory, shared memory,
    files, or databases.
    """

    def __init__(self, model_id: str):
        self.model_id: str = model_id
        self._shared_memory = None

    def unlink(self):
        """Removes the shared memory everywhere."""
        if self._shared_memory is not None:
            self._shared_memory.unlink()
            self._shared_memory = None

    def set_model(self, model: RegressionModel):
        """Places the regression model in shared memory."""
        assert isinstance(model, RegressionModel)
        pickled_model = pickle.dumps(model)
        self._shared_memory = SharedMemory(name=self.model_id, create=True, size=len(pickled_model))
        self._shared_memory.buf[:] = pickled_model

    def get_model_info(self) -> SharedMemoryBackedModelInfo:
        return SharedMemoryBackedModelInfo(shared_memory_name=self.model_id)
