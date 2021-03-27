#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Event, Process, Queue
import pickle
import time

import pandas as pd

from mlos.DataPlane.ModelHosting.SharedMemoryModelHost import start_shared_memory_model_host
from mlos.DataPlane.ModelHosting.ModelHostMessages import PredictRequest, PredictResponse, TrainRequest, TrainResponse
from mlos.DataPlane.SharedMemoryDataSet import SharedMemoryDataSet
from mlos.Optimizers.RegressionModels.DecisionTreeRegressionModel import DecisionTreeRegressionModel, decision_tree_config_store
from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import ObjectiveFunctionFactory, objective_function_config_store
from mlos.Spaces.HypergridAdapters import CategoricalToDiscreteHypergridAdapter

if __name__ == "__main__":

    request_queue = Queue()
    response_queue = Queue()
    shutdown_event = Event()

    model_host_processes = []

    try:
        for i in range(8):
            model_host_process = Process(
                target=start_shared_memory_model_host,
                kwargs=dict(
                    request_queue=request_queue,
                    response_queue=response_queue,
                    shutdown_event=shutdown_event
                )
            )
            model_host_process.start()
            model_host_processes.append(model_host_process)


        objective_function = ObjectiveFunctionFactory.create_objective_function(objective_function_config=objective_function_config_store.default)
        model_config = decision_tree_config_store.default
        parameter_space_adapter = CategoricalToDiscreteHypergridAdapter(adaptee=objective_function.parameter_space)
        model = DecisionTreeRegressionModel(
            model_config=model_config,
            input_space=parameter_space_adapter.target,
            output_space=objective_function.output_space
        )

        # Let's put the tree in the shared memory.
        #
        pickled_model = pickle.dumps(model)
        print(f"pickled model size: {len(pickled_model)}")
        untrained_model_name = 'untrained_tree'
        tree_shared_memory = SharedMemory(name=untrained_model_name, size=len(pickled_model), create=True)
        tree_shared_memory.buf[:] = pickled_model


        last_train_request_id = None
        num_requests = 10
        for i in range(num_requests):
            num_samples = (i + 1) * 1000
            # Let's fit the model remotely.
            #
            params_df = objective_function.parameter_space.random_dataframe(num_samples)
            objectives_df = objective_function.evaluate_dataframe(params_df)
            projected_params_df = parameter_space_adapter.project_dataframe(params_df, in_place=True)


            params_data_set = SharedMemoryDataSet(schema=parameter_space_adapter.target, shared_memory_name='params')
            params_data_set.set_dataframe(df=projected_params_df)

            objective_data_set = SharedMemoryDataSet(schema=objective_function.output_space, shared_memory_name='objectives')
            objective_data_set.set_dataframe(df=objectives_df)

            train_request = TrainRequest(
                untrained_model_id=untrained_model_name,
                features_data_set_info=params_data_set.get_data_set_info(),
                objectives_data_set_info=objective_data_set.get_data_set_info(),
                iteration_number=num_samples
            )


            last_train_request_id = train_request.request_id
            request_queue.put(train_request)

        last_train_response = None
        num_responses = 0
        timeout_s = 10
        while num_responses < num_requests:
            train_response: TrainResponse = response_queue.get(block=True, timeout=timeout_s)
            num_responses += 1
            if not train_response.success:
                raise train_response.exception

            if train_response.request_id == last_train_request_id:
                last_train_response = train_response


        shared_memory_data_set = SharedMemoryDataSet(schema=parameter_space_adapter.target, shared_memory_name="features_for_predict")
        shared_memory_data_set.set_dataframe(df=parameter_space_adapter.random_dataframe(100000))


        # Let's make the host produced the prediction.
        #
        desired_number_requests = 200000
        max_outstanding_requests = 1000
        num_outstanding_requests = 0
        num_complete_requests = 0

        while num_complete_requests < desired_number_requests:
            print(f"num_outstanding_requests: {num_outstanding_requests} / {max_outstanding_requests}, num_complete_requests: {num_complete_requests} / {desired_number_requests}")
            while num_outstanding_requests < max_outstanding_requests and (num_complete_requests + num_outstanding_requests) < desired_number_requests:
                predict_request = PredictRequest(model_id=last_train_response.trained_model_id, data_set_info=shared_memory_data_set.get_data_set_info())
                request_queue.put(predict_request)
                num_outstanding_requests += 1


            response_timeout_s = 30

            if num_outstanding_requests > 0:
                predict_response: PredictResponse = response_queue.get(block=True, timeout=response_timeout_s)
                num_outstanding_requests -= 1
                num_complete_requests += 1

                if not predict_response.success:
                    raise predict_response.exception

                assert len(predict_response.prediction.get_dataframe().index) == 100000

    finally:
        print("Setting the shutdown event")
        shutdown_event.set()
        print("Waiting for host to exit.")

        for model_host_process in model_host_processes:
            model_host_process.join()

        shared_memory_data_set.validate()
        shared_memory_data_set.unlink()


