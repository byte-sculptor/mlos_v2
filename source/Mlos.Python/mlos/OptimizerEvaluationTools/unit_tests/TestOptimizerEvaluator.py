#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import concurrent.futures
import os
import unittest

import pandas as pd

import mlos.global_values
from mlos.OptimizerEvaluationTools.OptimizerEvaluator import OptimizerEvaluator
from mlos.OptimizerEvaluationTools.OptimizerEvaluatorConfigStore import optimizer_evaluator_config_store
from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import objective_function_config_store
from mlos.Optimizers.BayesianOptimizerFactory import bayesian_optimizer_config_store
from mlos.Optimizers.RegressionModels.GoodnessOfFitMetrics import DataSetType
from mlos.Tracer import Tracer, traced



class TestOptimizerEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        mlos.global_values.declare_singletons()
        mlos.global_values.tracer = Tracer(actor_id=cls.__name__, thread_id=0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir = os.path.join(os.getcwd(), "temp")
        if not os.path.exists(cls.temp_dir):
            os.mkdir(cls.temp_dir)
        trace_output_path = os.path.join(cls.temp_dir, "TestOptimizerEvaluator.json")
        print(f"Dumping trace to {trace_output_path}")
        mlos.global_values.tracer.dump_trace_to_file(output_file_path=trace_output_path)

    def test_defaults(self):
        """Tests default optimizer configurations against default objective functions."""
        optimzier_evaluator_config = optimizer_evaluator_config_store.default
        optimizer_config = bayesian_optimizer_config_store.default
        objective_function_config = objective_function_config_store.default

        print(optimizer_config.to_json(indent=2))
        print(objective_function_config.to_json(indent=2))

        regression_model_fit_state, optima_over_time = OptimizerEvaluator.evaluate_optimizer(
            optimizer_evaluator_config=optimzier_evaluator_config,
            optimizer_config=optimizer_config,
            objective_function_config=objective_function_config
        )

        with pd.option_context('display.max_columns', 100):
            print(regression_model_fit_state.get_goodness_of_fit_dataframe(DataSetType.TRAIN).tail())
            for optimum_name, optimum_over_time in optima_over_time.items():
                print("#####################################################################################################")
                print(optimum_name)
                print(optimum_over_time.get_dataframe().tail(10))
                print("#####################################################################################################")


    def test_named_configs(self):
        """Tests all named optimizer configurations against all named objective functions."""
        optimizer_named_configs = bayesian_optimizer_config_store.list_named_configs()
        num_optimizer_configs = len(optimizer_named_configs)
        objective_function_named_configs = objective_function_config_store.list_named_configs()
        num_objective_function_configs = len(objective_function_named_configs)

        num_tests = 7

        with traced(scope_name="parallel_tests"), concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            outstanding_futures = set()

            for i in range(num_tests):
                named_optimizer_config = optimizer_named_configs[i % num_optimizer_configs]
                named_objective_function_config = objective_function_named_configs[i % num_objective_function_configs]

                print("#####################################################################################################")
                print(named_optimizer_config)
                print(named_objective_function_config)

                optimizer_evaluator_config = optimizer_evaluator_config_store.get_config_by_name(name="parallel_unit_tests_config")
                optimizer_config = named_optimizer_config.config_point
                objective_function_config = named_objective_function_config.config_point

                future = executor.submit(OptimizerEvaluator.evaluate_optimizer, optimizer_evaluator_config, optimizer_config, objective_function_config)
                outstanding_futures.add(future)

            done_futures, outstanding_futures = concurrent.futures.wait(outstanding_futures, return_when=concurrent.futures.ALL_COMPLETED)

            for future in done_futures:
                regression_model_fit_state, optima_over_time = future.result()
                with pd.option_context('display.max_columns', 100):
                    print(regression_model_fit_state.get_goodness_of_fit_dataframe(DataSetType.TRAIN).tail())
                    for optimum_name, optimum_over_time in optima_over_time.items():
                        print("#####################################################################################################")
                        print(optimum_name)
                        print(optimum_over_time.get_dataframe().tail(10))
                        print("#####################################################################################################")
