#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import pickle
import mlos.global_values
from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import ObjectiveFunctionFactory, objective_function_config_store
from mlos.OptimizerEvaluationTools.OptimizerEvaluationReport import OptimizerEvaluationReport
from mlos.OptimizerEvaluationTools.OptimizerEvaluatorConfigStore import optimizer_evaluator_config_store
from mlos.OptimizerEvaluationTools.OptimumOverTime import OptimumOverTime
from mlos.Optimizers.BayesianOptimizerFactory import BayesianOptimizerFactory, bayesian_optimizer_config_store
from mlos.Optimizers.OptimizationProblem import OptimizationProblem, Objective
from mlos.Optimizers.OptimumDefinition import OptimumDefinition
from mlos.Optimizers.RegressionModels.GoodnessOfFitMetrics import DataSetType
from mlos.Optimizers.RegressionModels.RegressionModelFitState import RegressionModelFitState
from mlos.Spaces import Point
from mlos.Tracer import trace, Tracer


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

    @classmethod
    @trace()
    def evaluate_optimizer(
        cls,
        optimizer_evaluator_config: Point,
        optimizer_config: Point,
        objective_function_config: Point
    ) -> OptimizerEvaluationReport:
        mlos.global_values.declare_singletons()

        assert optimizer_evaluator_config in optimizer_evaluator_config_store.parameter_space
        assert objective_function_config in objective_function_config_store.parameter_space
        assert optimizer_config in bayesian_optimizer_config_store.parameter_space

        if optimizer_evaluator_config.include_execution_trace_in_report:
            if mlos.global_values.tracer is None:
                mlos.global_values.tracer = Tracer()
            mlos.global_values.tracer.clear_events()

        optimizer_factory = BayesianOptimizerFactory()
        objective_function = ObjectiveFunctionFactory.create_objective_function(objective_function_config)

        objective_name = objective_function.output_space.dimension_names[0]
        optimization_problem = OptimizationProblem(
            parameter_space=objective_function.parameter_space,
            objective_space=objective_function.output_space,
            objectives=[Objective(name=objective_name, minimize=True)]
        )

        optimizer = optimizer_factory.create_local_optimizer(
            optimizer_config=optimizer_config,
            optimization_problem=optimization_problem
        )

        regression_model_fit_state = RegressionModelFitState()

        optima_over_time = {}
        optima_over_time["best_observation"] = OptimumOverTime(
            optimization_problem=optimization_problem,
            optimum_definition=OptimumDefinition.BEST_OBSERVATION
        )

        optima_over_time["best_predicted_value"] = OptimumOverTime(
            optimization_problem=optimization_problem,
            optimum_definition=OptimumDefinition.PREDICTED_VALUE_FOR_OBSERVED_CONFIG
        )

        optima_over_time["ucb_99"] = OptimumOverTime(
            optimization_problem=optimization_problem,
            optimum_definition=OptimumDefinition.UPPER_CONFIDENCE_BOUND_FOR_OBSERVED_CONFIG,
            alpha=0.01
        )

        optima_over_time["lcb_99"] = OptimumOverTime(
            optimization_problem=optimization_problem,
            optimum_definition=OptimumDefinition.LOWER_CONFIDENCE_BOUND_FOR_OBSERVED_CONFIG,
            alpha=0.01
        )

        #####################################################################################################

        for i in range(optimizer_evaluator_config.num_iterations):
            parameters = optimizer.suggest()
            objectives = objective_function.evaluate_point(parameters)
            optimizer.register(parameters.to_dataframe(), objectives.to_dataframe())

            if i % optimizer_evaluator_config.evaluation_frequency == 0:
                print(f"[{i+1}/{optimizer_evaluator_config.num_iterations}]")
                if optimizer.trained:
                    gof_metrics = optimizer.compute_surrogate_model_goodness_of_fit()
                    regression_model_fit_state.set_gof_metrics(data_set_type=DataSetType.TRAIN, gof_metrics=gof_metrics)

                for optimum_name, optimum_over_time in optima_over_time.items():
                    try:
                        optimum_config, optimum_value = optimizer.optimum(optimum_definition=optimum_over_time.optimum_definition,alpha=optimum_over_time.alpha)
                        optima_over_time[optimum_name].add_optimum_at_iteration(iteration=i, optimum_config=optimum_config, optimum_value=optimum_value)
                    except ValueError as e:
                        print(e)

        if optimizer.trained:
            gof_metrics = optimizer.compute_surrogate_model_goodness_of_fit()
            regression_model_fit_state.set_gof_metrics(data_set_type=DataSetType.TRAIN, gof_metrics=gof_metrics)

        for optimum_name, optimum_over_time in optima_over_time.items():
            try:
                optimum_config, optimum_value = optimizer.optimum(optimum_definition=optimum_over_time.optimum_definition, alpha=optimum_over_time.alpha)
                optima_over_time[optimum_name].add_optimum_at_iteration(iteration=optimizer_evaluator_config.num_iterations, optimum_config=optimum_config, optimum_value=optimum_value)
            except Exception as e:
                print(e)

        execution_trace = None
        if optimizer_evaluator_config.include_execution_trace_in_report:
            execution_trace = mlos.global_values.tracer.trace_events
            mlos.global_values.tracer.clear_events()

        pickled_optimizer = None
        if optimizer_evaluator_config.include_pickled_optimizer_in_report:
            pickled_optimizer = pickle.dumps(optimizer)

        pickled_objective_function = None
        if optimizer_evaluator_config.include_pickled_objective_function_in_report:
            pickled_objective_function = pickle.dumps(objective_function)

        return OptimizerEvaluationReport(
            optimizer_configuration=optimizer_config,
            objective_function_configuration=objective_function_config,
            pickled_optimizer=pickled_optimizer,
            pickled_objective_function=pickled_objective_function,
            num_optimization_iterations=optimizer_evaluator_config.num_iterations,
            evaluation_frequency=optimizer_evaluator_config.evaluation_frequency,
            regression_model_goodness_of_fit_state=regression_model_fit_state,
            optima_over_time=optima_over_time,
            execution_trace=execution_trace
        )

