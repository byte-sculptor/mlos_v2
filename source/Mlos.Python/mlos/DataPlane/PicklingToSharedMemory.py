#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from multiprocessing import cpu_count, Queue
import os
from queue import Empty

from mlos.DataPlane.ModelHosting.SharedMemoryModelHost import start_shared_memory_model_host
from mlos.DataPlane.ModelHosting import PredictRequest, PredictResponse, TrainRequest, TrainResponse, SharedMemoryBackedModelWriter
from mlos.DataPlane.ModelHosting.ModelHostProcessPool import ModelHostProcessPool
from mlos.DataPlane.SharedMemoryDataSets import SharedMemoryDataSet, SharedMemoryDataSetInfo, SharedMemoryDataSetService
from mlos.Logger import create_logger
from mlos.Optimizers.RegressionModels.DecisionTreeRegressionModel import DecisionTreeRegressionModel, decision_tree_config_store
from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import ObjectiveFunctionFactory, objective_function_config_store
from mlos.Spaces.HypergridAdapters import CategoricalToDiscreteHypergridAdapter


if __name__ == "__main__":
    logger = create_logger(f"main_{os.getpid()}")

    request_queue = Queue()
    response_queue = Queue()
    data_set_service = SharedMemoryDataSetService(logger=logger)
    data_set_service.launch()
    shared_memory_data_set_store = data_set_service.data_set_store

    host_process_pool = ModelHostProcessPool(
        shared_memory_data_set_service=data_set_service,
        request_queue=request_queue,
        response_queue=response_queue,
        max_num_processes=cpu_count(),
        logger=logger
    )


    try:
        host_process_pool.start()


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
        untrained_model_name = 'untrained_tree'
        model_writer = SharedMemoryBackedModelWriter(model_id=untrained_model_name)
        model_writer.set_model(model=model)

        last_train_request_id = None
        num_requests = 2

        params_data_sets = []
        objectives_data_sets = []

        for i in range(num_requests):
            num_samples = (i + 1) * 100000
            # Let's fit the model remotely.
            #
            params_df = objective_function.parameter_space.random_dataframe(num_samples)
            objectives_df = objective_function.evaluate_dataframe(params_df)
            projected_params_df = parameter_space_adapter.project_dataframe(params_df, in_place=True)

            params_data_set = shared_memory_data_set_store.create_data_set(
                data_set_info=SharedMemoryDataSetInfo(schema=parameter_space_adapter.target),
                df=projected_params_df
            )

            objective_data_set = shared_memory_data_set_store.create_data_set(
                data_set_info=SharedMemoryDataSetInfo(schema=objective_function.output_space),
                df=objectives_df
            )

            params_data_sets.append(params_data_set)
            objectives_data_sets.append(objective_data_set)

            train_request = TrainRequest(
                model_info=model_writer.get_model_info(),
                features_data_set_info=params_data_set.get_data_set_info(),
                objectives_data_set_info=objective_data_set.get_data_set_info(),
                iteration_number=num_samples
            )

            request_queue.put(train_request)

        num_responses = 0
        timeout_s = 10
        while num_responses < num_requests:
            train_response: TrainResponse = response_queue.get(block=True, timeout=timeout_s)
            num_responses += 1
            if not train_response.success:
                raise train_response.exception

            if train_response.request_id == last_train_request_id:
                last_train_response = train_response

        # The code above demonstrates that we can in principle use the remote model host to train models, but it's not yet
        # hardened. Let's train a model in this process, put it in shared memory and retain ownership of it.
        #
        params_df = objective_function.parameter_space.random_dataframe(10000)
        objectives_df = objective_function.evaluate_dataframe(params_df)
        projected_params_df = parameter_space_adapter.project_dataframe(params_df, in_place=True)
        model.fit(projected_params_df, objectives_df, iteration_number=10000)
        model_name = 'trained_tree'
        model_writer = SharedMemoryBackedModelWriter(model_id=model_name)
        model_writer.set_model(model=model)
        model_info = model_writer.get_model_info()


        # Now that we've trained the models, we can clean up the params_data_sets and objectives_data_sets.
        #
        data_sets_to_clean_up = params_data_sets + objectives_data_sets
        for data_set in data_sets_to_clean_up:
            shared_memory_data_set_store.unlink_data_set(data_set_info=data_set.get_data_set_info())

        num_predictions = 100000
        parameters_for_predictions = shared_memory_data_set_store.create_data_set(
            data_set_info=SharedMemoryDataSetInfo(schema=parameter_space_adapter.target),
            df=parameter_space_adapter.random_dataframe(num_predictions)
        )

        # Let's make the host produce the prediction.
        #
        desired_number_requests = 100000
        max_outstanding_requests = 100
        num_outstanding_requests = 0
        num_complete_requests = 0
        num_failed_requests = 0


        while num_complete_requests < desired_number_requests:

            logger.info(f"num_outstanding_requests: {num_outstanding_requests} / {max_outstanding_requests}, num_complete_requests: {num_complete_requests} / {desired_number_requests}, num_failed_requests: {num_failed_requests} / {num_complete_requests}")

            while num_outstanding_requests < max_outstanding_requests and (num_complete_requests + num_outstanding_requests) < desired_number_requests:
                predict_request = PredictRequest(model_info=model_info, data_set_info=parameters_for_predictions.get_data_set_info())
                request_queue.put(predict_request)
                num_outstanding_requests += 1


            response_timeout_s = 10

            if num_outstanding_requests > 0:
                try:
                    predict_response: PredictResponse = response_queue.get(block=True, timeout=response_timeout_s)
                    num_outstanding_requests -= 1
                    num_complete_requests += 1
                except Empty:
                    continue

                if not predict_response.success:
                    logger.info(f"Request {predict_response.request_id} failed with exception: {predict_response.exception}")
                    num_failed_requests += 1

                else:
                    try:
                        prediction_data_set = shared_memory_data_set_store.get_data_set(data_set_info=predict_response.prediction_data_set_info)
                        prediction = predict_response.prediction
                        prediction_df = prediction_data_set.get_dataframe()
                        logger.info(f"Response to request:{predict_response.request_id} received ")
                        prediction.set_dataframe(dataframe=prediction_df)
                        assert len(prediction.get_dataframe().index) == num_predictions

                        if num_complete_requests % 1 == 0:
                            parameters_for_predictions.validate()
                            stats_df = shared_memory_data_set_store.get_stats()
                            print(stats_df)
                    except:
                        logger.error("Failed to process the response.", exc_info=True)
                        num_failed_requests += 1
                    finally:
                        shared_memory_data_set_store.unlink_data_set(data_set_info=predict_response.prediction_data_set_info)





        parameters_for_predictions.validate()
        shared_memory_data_set_store.unlink_data_set(data_set_info=parameters_for_predictions.get_data_set_info())

    except Exception as e:
        logger.info("Exception: ", exc_info=True)

    finally:
        stats_df = shared_memory_data_set_store.get_stats()
        logger.info(stats_df)

        logger.info("Shutting down DataSetStoreService")
        data_set_service.stop()

        host_process_pool.stop()


        model_writer.unlink()


