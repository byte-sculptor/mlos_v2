#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import concurrent.futures

from mlos.Optimizers.BayesianOptimizerConfigStore import bayesian_optimizer_config_store
from mlos.Optimizers.BayesianOptimizerFactory import BayesianOptimizerFactory
from mlos.Optimizers.OptimizationProblem import OptimizationProblem, Objective
from mlos.Optimizers.OptimumDefinition import OptimumDefinition

from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import ObjectiveFunctionFactory
from mlos.OptimizerEvaluationTools.ObjectiveFunctionConfigStore import objective_function_config_store

from mlos.Spaces import Point

import mlos.global_values



def run_optimization(optimizer_config_json_string, objective_function_config_json_string, num_iterations):
    mlos.global_values.declare_singletons()
    optimizer_config = Point.from_json(optimizer_config_json_string)
    objective_function_config = Point.from_json(objective_function_config_json_string)

    objective_function = ObjectiveFunctionFactory.create_objective_function(objective_function_config)
    optimizer_factory = BayesianOptimizerFactory()
    optimizer = optimizer_factory.create_local_optimizer(
        optimizer_config=optimizer_config,
        optimization_problem=OptimizationProblem(
            parameter_space=objective_function.parameter_space,
            objective_space=objective_function.output_space,
            objectives=[Objective(name='y', minimize=True)]
        )
    )
    for i in range(num_iterations):
       parameters = optimizer.suggest()
       objectives = objective_function.evaluate_point(parameters)
       optimizer.register(parameters.to_dataframe(), objectives.to_dataframe())
    optimum_config, optimum_value = optimizer.optimum(OptimumDefinition.BEST_OBSERVATION)
    return optimizer_config_json_string, optimum_value.to_json()


if __name__ == "__main__":

    num_desired_runs = 20
    num_completed_runs = 0
    max_concurrent_jobs = 7

    objective_function_config = objective_function_config_store.get_config_by_name('three_level_quadratic')

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_jobs) as executor:
        outstanding_futures = set()
        while num_completed_runs < num_desired_runs:
            print(f"[{num_completed_runs}/{num_desired_runs}]")

            # Let's keep submitting new jobs to the pool until we have the desired number of executions.
            #
            num_remaining_jobs_to_schedule = min(num_desired_runs - num_completed_runs - len(outstanding_futures), max_concurrent_jobs)
            if num_remaining_jobs_to_schedule > 0:
                for _ in range(num_remaining_jobs_to_schedule):
                    inner_optimizer_config = bayesian_optimizer_config_store.default  # meta_optimizer.suggest()
                    future = executor.submit(
                        run_optimization,
                        inner_optimizer_config.to_json(),
                        objective_function_config.to_json(),
                        num_iterations=10
                    )
                    outstanding_futures.add(future)

            # Now let's wait for any future to complete.
            #
            done_futures, outstanding_futures = concurrent.futures.wait(outstanding_futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done_futures:
                inner_optimizer_config_json, best_observation_value_json = future.result()
                inner_optimizer_config = Point.from_json(inner_optimizer_config_json)
                best_observation_value = Point.from_json(best_observation_value_json)
                # meta_optimizer.register(
                #     feature_values_pandas_frame=inner_optimizer_config.to_dataframe(),
                #     target_values_pandas_frame=Point(optimum_value_after_100_iterations=best_observation_value.y).to_dataframe()
                # )
                num_completed_runs += 1


