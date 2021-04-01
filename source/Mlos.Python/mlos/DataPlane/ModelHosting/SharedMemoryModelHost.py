#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from functools import wraps
from multiprocessing import connection, Event, Queue
import os
from queue import Empty
from typing import Dict
import sys
from uuid import UUID

from mlos.DataPlane.ModelHosting import Response, PredictRequest, PredictResponse, TrainRequest, TrainResponse, \
    SharedMemoryBackedModelReader, SharedMemoryBackedModelWriter
from mlos.DataPlane.SharedMemoryDataSets import SharedMemoryDataSet, SharedMemoryDataSetInfo, SharedMemoryDataSetStoreProxy, attached_data_set_view
from mlos.Logger import create_logger
from mlos.Optimizers.RegressionModels.RegressionModel import RegressionModel

def request_handler():
    def request_handler_decorator(wrapped_function):
        @wraps(wrapped_function)
        def wrapper(*args, **kwargs):
            request_id = None
            try:
                request = kwargs['request']
                request_id = request.request_id

                return wrapped_function(*args, **kwargs)
            except MemoryError as e:
                # Apparently memory error is unpicklable
                self = args[0]
                self.logger.error(f"Failed to process request:  {request_id}", exc_info=True)
                return Response(request_id=request_id, success=False, exception=Exception(f"MemoryError: {str(e)}"))
            except Exception as e:
                self = args[0]
                self.logger.error(f"Failed to process request:  {request_id}", exc_info=True)
                return Response(request_id=request_id, success=False, exception=e)

        return wrapper
    return request_handler_decorator


class SharedMemoryModelHost:
    """A worker responsible for scoring models held in shared memory.

    Objective:
        The goal is to parallelize model scoring (calls to model.predict(...)) so as to have many trees evaluating simultaneously
        in shared memory. This should provide a considerable speed boost for producing suggestions, tomograph interactions, and
        lastly model training.


    Strategy:
        Each instance of shared memory model executor will listen for requests on a queue. Once a request object is received,
        the executor will check its cache to see if it already has the requisite model deserialized, if it doesn't, it will locate
        the model in shared memory and deserialize it.

        It will then locate the features_df in shared memory and invoke the model's predict() method. Lastly, it will stash the
        prediction's dataframe in shared memory and send back the message informing the client how to find it.

        Finally, once all of the above works, we can extend the Host to also fit the models and place them in shared memory.


    Additional Notes:
        Tracing shows that currently, serial execution of models is one of the biggest bottlenecks in both registering an observation,
        and - more importantly - producing a suggestion. Parallelizing inference is the single biggest low-hanging fruit we have.

        Since Python is not big on multi-threading (GIL is here to stay), the preferred parallelization method is to use multiple
        processes. This creates the question of inter-process communication. We have a ton of options:
            * shared memory
            * pipes
            * sockets
            * ...
        But only shared memory lets us avoid unnecessary copying and serializing/deserializing of our (relatively large) dataframes
        and models. So we will leverage Python's multiprocessing.SharedMemory module to stash our data and models in shared memory.

    """

    def __init__(
        self,
        request_queue: Queue,
        response_queue: Queue,
        shutdown_event: Event,
        data_set_store_service_connection: connection,
        logger = None
    ):
        if logger is None:
            logger = create_logger(f"{self.__class__.__name__}_{os.getpid()}")
        self.logger = logger
        self.request_queue: Queue = request_queue
        self.response_queue: Queue = response_queue
        self.shutdown_event: Event = shutdown_event
        self._data_set_store_proxy: SharedMemoryDataSetStoreProxy = SharedMemoryDataSetStoreProxy(
            service_connection=data_set_store_service_connection,
            logger=logger
        )
        self._model_cache: Dict[str, RegressionModel] = dict()

        # We need to keep a reference to SharedMemory objects, or they will be garbage collected.
        #
        self._shared_memory_cache: Dict[str, SharedMemoryBackedModelWriter] = dict()

    def run(self):
        self.logger.info(f'{os.getpid()} running')
        timeout_s = 1
        while not self.shutdown_event.is_set():

            if self._data_set_store_proxy.total_leaked_bytes > 10**9:
                self.logger.info(f"Reached a memory-leak threshold. This process has leaked {self._data_set_store_proxy.total_leaked_bytes / (10**6)} MB. Exiting.")
                os._exit(-1)

            try:
                request = self.request_queue.get(block=True, timeout=timeout_s)
                request_id = request.request_id
                self.logger.info(f"{os.getpid()} Got request {request_id} of type: {type(request)}")
            except Empty:
                continue

            try:
                if isinstance(request, PredictRequest):
                    response = self._process_predict_request(request=request)
                elif isinstance(request, TrainRequest):
                    response = self._process_train_request(request=request)
                else:
                    response = Response(
                        request_id=request.request_id,
                        success=False,
                        exception=RuntimeError(f"Unknown request type: {type(request)}")
                    )
                self.response_queue.put(response)

            except:
                self.logger.error(f"Failed to process request {request_id}")

        self.logger.info(f"{os.getpid()} freeing up models memory")
        for name, model in self._shared_memory_cache.items():
            model.unlink()

        self.logger.info(f"{os.getpid()} shutting down")
        sys.exit(0)

    @request_handler()
    def _process_predict_request(self, request: PredictRequest):
        print(self._data_set_store_proxy._data_set_store.get_stats())
        if request.model_info.model_id in self._model_cache:
            self.logger.info(f"{os.getpid()} Model id: {request.model_info.model_id} found in cache.")
            model = self._model_cache[request.model_info.model_id]
        else:
            self.logger.info(f"{os.getpid()} Model id: {request.model_info.model_id} not found in cache. Deserializing from shared memory.")
            model_reader = SharedMemoryBackedModelReader(shared_memory_model_info=request.model_info)
            model = model_reader.get_model()
            self._model_cache[request.model_info.model_id] = model
            model_reader.detach()
            self.logger.info(f"{os.getpid()} Model id: {request.model_info.model_id} deserialized from shared memory and placed in the cache.")

        with attached_data_set_view(data_set_info=request.data_set_info) as features_data_set_view:
            features_df = features_data_set_view.get_dataframe()
            prediction = model.predict(feature_values_pandas_frame=features_df, include_only_valid_rows=True)


        prediction_df = prediction.get_dataframe()

        self.logger.info(f"column_names: {prediction.expected_column_names}")
        prediction_data_set = self._data_set_store_proxy.create_data_set(
            data_set_info=SharedMemoryDataSetInfo(
                schema=None,
                column_names=prediction.expected_column_names
            ),
            df=prediction_df
        )
        prediction_data_set_info = prediction_data_set.get_data_set_info()
        self._data_set_store_proxy.detach_data_set(data_set_info=prediction_data_set_info)


        prediction.clear_dataframe()

        # TODO: put predictions in shared memory and send the response.
        response = PredictResponse(
            request_id=request.request_id,
            prediction=prediction,
            prediction_data_set_info=prediction_data_set_info
        )

        self.logger.info(f"{os.getpid()} Produced a response with {len(prediction_df.index)} predictions of size: {prediction_data_set_info.shared_memory_np_array_nbytes / 2**20} MB")
        return response

    @request_handler()
    def _process_train_request(self, request: TrainRequest):
        self.logger.info(f"Deserializing model: {request.model_info.model_id} from shared memory.")

        untrained_model_reader = SharedMemoryBackedModelReader(shared_memory_model_info=request.model_info)
        model = untrained_model_reader.get_model()
        untrained_model_reader.detach()
        self.logger.info(f"Successlly deserialized model: {request.model_info.model_id} from shared memory.")

        with attached_data_set_view(data_set_info=request.features_data_set_info) as features_data_set_view, \
            attached_data_set_view(data_set_info=request.objectives_data_set_info) as objectives_data_set_view:

            features_df = features_data_set_view.get_dataframe()
            objectives_df = objectives_data_set_view.get_dataframe()

            model.fit(
                feature_values_pandas_frame=features_df,
                target_values_pandas_frame=objectives_df,
                iteration_number=request.iteration_number
            )

            trained_model_id = f"{request.model_info.model_id}_{request.iteration_number}"
            self.logger.info(
                f"{os.getpid()} Successfully trained the model {trained_model_id}. Placing it in local cache and in shared memory.")
            model_writer = SharedMemoryBackedModelWriter(model_id=trained_model_id)
            model_writer.set_model(model=model)

            self._model_cache[trained_model_id] = model

            # We must keep a reference to the shared memory. We need to implement a mechanism to clear this cache.
            #
            self._shared_memory_cache[trained_model_id] = model_writer

            response = TrainResponse(
                model_info=model_writer.get_model_info(),
                request_id=request.request_id
            )

            self.logger.info(f"{os.getpid()} Produced a response to request: {request.request_id}.")
            return response


def start_shared_memory_model_host(
    request_queue: Queue,
    response_queue: Queue,
    shutdown_event: Event,
    data_set_store_service_connection: connection
):
    model_host = SharedMemoryModelHost(
        request_queue=request_queue,
        response_queue=response_queue,
        shutdown_event=shutdown_event,
        data_set_store_service_connection=data_set_store_service_connection
    )
    model_host.run()
