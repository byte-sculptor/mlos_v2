#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import unittest
import warnings

import pandas as pd

from mlos.OptimizerEvaluationTools.OptimizerEvaluator import OptimizerEvaluator
from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import ObjectiveFunctionFactory, objective_function_config_store
from mlos.Optimizers.BayesianOptimizerFactory import BayesianOptimizerFactory, bayesian_optimizer_config_store
from mlos.Optimizers.RegressionModels.GoodnessOfFitMetrics import DataSetType
import mlos.global_values



class TestOptimizerEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        mlos.global_values.declare_singletons()

    def test_defaults(self):
        """Tests default optimizer configurations against default objective functions."""
        optimizer_config = bayesian_optimizer_config_store.default
        objective_function_config = objective_function_config_store.default

        print(optimizer_config.to_json(indent=2))
        print(objective_function_config.to_json(indent=2))

        regression_model_fit_state, optima_over_time = OptimizerEvaluator.evaluate_optimizer(
            optimizer_config=optimizer_config,
            objective_function_config=objective_function_config,
            num_iterations=101,
            evaluation_frequency=10
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

        for named_optimizer_config in bayesian_optimizer_config_store.list_named_configs():
            for named_objective_function_config in objective_function_config_store.list_named_configs():
                print(named_optimizer_config)
                print(named_objective_function_config)

                optimizer_config = named_optimizer_config.config_point
                objective_function_config = named_objective_function_config.config_point

                print(optimizer_config.to_json(indent=2))
                print(objective_function_config.to_json(indent=2))

                regression_model_fit_state, optima_over_time = OptimizerEvaluator.evaluate_optimizer(
                    optimizer_config=optimizer_config,
                    objective_function_config=objective_function_config,
                    num_iterations=51,
                    evaluation_frequency=10
                )

                with pd.option_context('display.max_columns', 100):
                    print(regression_model_fit_state.get_goodness_of_fit_dataframe(DataSetType.TRAIN).tail())
                    for optimum_name, optimum_over_time in optima_over_time.items():
                        print("#####################################################################################################")
                        print(optimum_name)
                        print(optimum_over_time.get_dataframe().tail(10))
                        print("#####################################################################################################")
