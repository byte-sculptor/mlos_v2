import mlos.global_values
from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import ObjectiveFunctionFactory, objective_function_config_store
from mlos.Optimizers.BayesianOptimizerFactory import BayesianOptimizerFactory, bayesian_optimizer_config_store
from mlos.Optimizers.OptimizationProblem import OptimizationProblem, Objective
from mlos.Optimizers.OptimumDefinition import OptimumDefinition
from mlos.Spaces import Point

class OptimizerEvaluator:
    """Evaluates optimizers against objective functions.

    This class are responsible for:
        1. instantiating an optimizer
        2. instantiating an objective function
        3. launching an optimizer against that objective function
        4. keeping track of the goodness of fit, and of the various optima over time.
        5. Producing a report containing:
            1. Optimizer configuration
            1. Objective function configuration
            1. Goodness of fit over time
            1. Optima (best observation, best predicted value, best ucb, best lcb) over time

    """

    @staticmethod
    def evaluate_optimizer(
        optimizer_config: Point,
        objective_function_config: Point,
        num_iterations: int,
        evaluation_frequency: int,
    ):
        assert objective_function_config in objective_function_config_store.parameter_space
        assert optimizer_config in bayesian_optimizer_config_store.parameter_space
        assert num_iterations > 0
        assert evaluation_frequency > 0

        optimizer_factory = BayesianOptimizerFactory()
        objective_function = ObjectiveFunctionFactory.create_objective_function(objective_function_config)

        objective_name = objective_function.output_space.dimension_names[0]
        optimizer = optimizer_factory.create_local_optimizer(
            optimizer_config=optimizer_config,
            optimization_problem=OptimizationProblem(
                parameter_space=objective_function.parameter_space,
                objective_space=objective_function.output_space,
                objectives=[Objective(name=objective_name, minimize=True)]
            )
        )

        for i in range(num_iterations):
            parameters = optimizer.suggest()
            objectives = objective_function.evaluate_point(parameters)
            optimizer.register(parameters.to_dataframe(), objectives.to_dataframe())

            if i % evaluation_frequency == 0:
                best_observation_config, best_observation = optimizer.optimum(OptimumDefinition.BEST_OBSERVATION)
                best_predicted_value_config, best_predicted_value = optimizer.optimum(OptimumDefinition.PREDICTED_VALUE_FOR_OBSERVED_CONFIG)

                ucb_90_ci_config, ucb_90_ci_optimum = optimizer.optimum(OptimumDefinition.UPPER_CONFIDENCE_BOUND_FOR_OBSERVED_CONFIG, alpha=0.1)
                ucb_95_ci_config, ucb_95_ci_optimum = optimizer.optimum(OptimumDefinition.UPPER_CONFIDENCE_BOUND_FOR_OBSERVED_CONFIG, alpha=0.05)
                ucb_99_ci_config, ucb_99_ci_optimum = optimizer.optimum(OptimumDefinition.UPPER_CONFIDENCE_BOUND_FOR_OBSERVED_CONFIG, alpha=0.01)

                lcb_90_ci_config, lcb_90_ci_optimum = optimizer.optimum(OptimumDefinition.LOWER_CONFIDENCE_BOUND_FOR_OBSERVED_CONFIG, alpha=0.1)
                lcb_95_ci_config, lcb_95_ci_optimum = optimizer.optimum(OptimumDefinition.LOWER_CONFIDENCE_BOUND_FOR_OBSERVED_CONFIG, alpha=0.05)
                lcb_99_ci_config, lcb_99_ci_optimum = optimizer.optimum(OptimumDefinition.LOWER_CONFIDENCE_BOUND_FOR_OBSERVED_CONFIG, alpha=0.01)







