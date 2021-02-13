#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import math
import pandas as pd

from mlos.Optimizers.OptimizationProblem import OptimizationProblem
from mlos.Optimizers.ExperimentDesigner.UtilityFunctionOptimizers.UtilityFunctionOptimizer import UtilityFunctionOptimizer
from mlos.Optimizers.ExperimentDesigner.UtilityFunctions.UtilityFunction import UtilityFunction
from mlos.Spaces import SimpleHypergrid, DiscreteDimension, Point
from mlos.Spaces.Configs.ComponentConfigStore import ComponentConfigStore
from mlos.Tracer import trace


random_search_optimizer_config_store = ComponentConfigStore(
    parameter_space=SimpleHypergrid(
        name="random_search_optimizer_config",
        dimensions=[
            DiscreteDimension(name="num_samples_per_iteration", min=1, max=1000000)
        ]
    ),
    default=Point(
        num_samples_per_iteration=1000
    )
)


class RandomSearchOptimizer(UtilityFunctionOptimizer):
    """ Performs a random search over the search space.

    This is the simplest optimizer to implement and a good baseline for all other optimizers
    to beat.

    """

    def __init__(
            self,
            optimizer_config: Point,
            optimization_problem: OptimizationProblem,
            utility_function: UtilityFunction,
            logger=None
    ):
        UtilityFunctionOptimizer.__init__(self, optimizer_config, optimization_problem, utility_function, logger)

    @trace()
    def suggest(self, context_values_dataframe: pd.DataFrame = None):  # pylint: disable=unused-argument
        """ Returns the next best configuration to try.

        It does so by generating num_samples_per_iteration random configurations,
        passing them through the utility function and selecting the configuration with
        the highest utility value.

        TODO: make it capable of consuming the context values
        :return:
        """

        # Let's do batches of 10000 points max so that we minimize the risk of OOM.
        #
        max_num_samples_in_batch = 10000
        num_batches = math.ceil(self.optimizer_config.num_samples_per_iteration / max_num_samples_in_batch)

        max_value = None
        config_to_suggest = None

        for i in range(num_batches):
            parameter_values_dataframe = self.optimization_problem.parameter_space.random_dataframe(num_samples=max_num_samples_in_batch)
            feature_values_dataframe = self.optimization_problem.construct_feature_dataframe(
                parameter_values=parameter_values_dataframe,
                context_values=context_values_dataframe,
                product=True
            )
            utility_function_values = self.utility_function(feature_values_pandas_frame=feature_values_dataframe.copy(deep=False))
            num_utility_function_values = len(utility_function_values.index)
            if num_utility_function_values == 0:
                continue
                index_of_max_value = utility_function_values[['utility']].idxmax()['utility'] if num_utility_function_values > 0 else 0
                max_value_fort_this_batch = utility_function_values.loc[index_of_max_value, 'utility']
                print(f"Probability of improvement: {max_value_fort_this_batch}")

                if max_value is None or max_value_fort_this_batch > max_value:
                    max_value = max_value_fort_this_batch
                    argmax_point = Point.from_dataframe(feature_values_dataframe.loc[[index_of_max_value]])
                    config_to_suggest = argmax_point[self.optimization_problem.parameter_space.name]

        if config_to_suggest is None:
            config_to_suggest = self.optimization_problem.parameter_space.random()

        self.logger.debug(f"Suggesting: {str(config_to_suggest)}")
        return config_to_suggest
