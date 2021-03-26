#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Event, Process, Queue
import pickle
import time

from mlos.DataPlane.ModelHosting.SharedMemoryModelHost import start_shared_memory_model_host
from mlos.DataPlane.ModelHosting.ModelHostMessages import PredictRequest
from mlos.DataPlane.SharedMemoryDataSet import SharedMemoryDataSet
from mlos.Optimizers.RegressionModels.DecisionTreeRegressionModel import DecisionTreeRegressionModel, decision_tree_config_store
from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import ObjectiveFunctionFactory, objective_function_config_store
from mlos.Spaces.HypergridAdapters import CategoricalToDiscreteHypergridAdapter

if __name__ == "__main__":

    objective_function = ObjectiveFunctionFactory.create_objective_function(objective_function_config=objective_function_config_store.default)
    model_config = decision_tree_config_store.default
    parameter_space_adapter = CategoricalToDiscreteHypergridAdapter(adaptee=objective_function.parameter_space)
    model = DecisionTreeRegressionModel(
        model_config=model_config,
        input_space=parameter_space_adapter,
        output_space=objective_function.output_space
    )

    params_df = objective_function.parameter_space.random_dataframe(1000)
    projected_params_df = parameter_space_adapter.project_dataframe(params_df, in_place=True)
    objectives_df = objective_function.evaluate_dataframe(projected_params_df)
    model.fit(projected_params_df, objectives_df, 0)

    pickled_model = pickle.dumps(model)
    print(f"pickled model size: {len(pickled_model)}")

    # Let's put the tree in the shared memory.
    #
    tree_shared_memory = SharedMemory(name='pickled_tree', size=len(pickled_model), create=True)
    tree_shared_memory.buf[:] = pickled_model

    shared_memory_data_set = SharedMemoryDataSet(schema=objective_function.parameter_space, shared_memory_name="params2")
    shared_memory_data_set.set_dataframe(df=objective_function.parameter_space.random_dataframe(11))

    request_queue = Queue()
    response_queue = Queue()
    shutdown_event = Event()
    model_host_process = Process(
        target=start_shared_memory_model_host,
        kwargs=dict(
            request_queue=request_queue,
            response_queue=response_queue,
            shutdown_event=shutdown_event
        )
    )
    model_host_process.start()

    # Let's make the host produce the prediction.
    #
    predict_request = PredictRequest(model_id='pickled_tree', data_set_info=shared_memory_data_set.get_data_set_info())
    request_queue.put(predict_request)

    time.sleep(2)
    print("Setting the shutdown event")
    shutdown_event.set()
    print("Waiting for host to exit.")
    model_host_process.join()

    shared_memory_data_set.validate()
    shared_memory_data_set.unlink()


