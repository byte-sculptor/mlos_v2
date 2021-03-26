#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from uuid import UUID, uuid4

from mlos.DataPlane.SharedMemoryDataSetInfo import SharedMemoryDataSetInfo
from mlos.Optimizers.RegressionModels.Prediction import Prediction

# TODO: convert all of this to gRPC so that models can be executed remotely and in any language.


class Request:
    """Base class for all requests."""
    def __init__(self, request_id: UUID = None):
        if request_id is None:
            request_id = uuid4()
        self.request_id = request_id


class Response:
    """Base class for all responses."""
    def __init__(self, request_id: UUID = None, success: bool = True, exception: Exception = None):
        assert (request_id is not None) and isinstance(request_id, UUID)
        self.request_id = request_id
        self.success = success
        self.exception = exception


class PredictRequest(Request):
    """Request for the host to produce predictions."""

    def __init__(self, model_id: str, data_set_info: SharedMemoryDataSetInfo, request_id: UUID = None):
        Request.__init__(self, request_id=request_id)
        self.model_id = model_id
        self.data_set_info = data_set_info


class PredictResponse(Response):
    """Response to PredictRequest.

    TODO: make all responses inherit from a common base class so that we can send back exceptions and warnings.

    For now the response contains the pickled prediction, which imposes an additional serialziation/deserialization overhead.
    TODO: put the prediction in shared memory
    """
    def __init__(self, prediction: Prediction, request_id: UUID):
        Response.__init__(self, request_id=request_id)
        self.prediction = prediction


class TrainRequest(Request):
    """Request for the model host to train a model.

    """
    def __init__(
        self,
        untrained_model_id: str,
        features_data_set_info: SharedMemoryDataSetInfo,
        objectives_data_set_info: SharedMemoryDataSetInfo,
        iteration_number: int,
        request_id: UUID = None
    ):
        Request.__init__(self, request_id=request_id)
        self.untrained_model_id: str = untrained_model_id
        self.features_data_set_info: SharedMemoryDataSetInfo = features_data_set_info
        self.objectives_data_set_info: SharedMemoryDataSetInfo = objectives_data_set_info
        self.iteration_number = iteration_number


class TrainResponse(Response):

    def __init__(self, trained_model_id: str, request_id: UUID):
        Response.__init__(self, request_id=request_id)
        self.trained_model_id: str = trained_model_id
