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




    def test_random_configs(self):
        """Tests random optimizer configurations against random objective functions."""
        for _ in range(10):
            optimizer_config = bayesian_optimizer_config_store.parameter_space.random()

            # Let's make sure the optimizer config is not too time consuming.
            #
            random_forest_config = optimizer_config.homogeneous_random_forest_regression_model_config
            random_forest_config.n_estimators = min(random_forest_config.n_estimators, 20)

            decision_tree_config = random_forest_config.decision_tree_regression_model_config
            decision_tree_config.min_samples_to_fit = min(decision_tree_config.min_samples_to_fit, 20)
            decision_tree_config.n_new_samples_before_refit = min(decision_tree_config.n_new_samples_before_refit, 20)


            objective_function_config = objective_function_config_store.parameter_space.random()

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
