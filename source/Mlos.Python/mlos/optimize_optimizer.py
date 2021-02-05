"""The goal of this notebook is to optimize the optimizer.

Specifically, this will be used to tune the parameters to the new multi-objective optimization capabilities, as well as a template
for all future optimizer optimizations.

We will first build a search-space for the target optimizer (a subset of the total configuration space), then configure the objective
function, then we will configure the OptimizerEvaluator, and lastly the meta-optimizer. We will let the meta-optimizer rip until it
runs out of budget, but it will also be spitting to disk all intermittent results so that we can monitor its progress in real time.


Some of the things I would like to tune in the target optimizer are:
 1. number of decision trees
 2. samples fraction per estimator
 3. bootstrap - does it help?
 4. decision trees: n_new_samples_before_refit
 5. fraction random parameters
 6. numeric optimizer implementation and related parameters
 7. num_monte_carlo samples in multi-objective poi

 And some of the things I care about are:
 1. pareto volume after 500, 1000, and 2000 iterations.
 2. execution duration for 2000 iterations

 We need a not too complicated function with three objectives. A 3D hypersphere could do the trick, though it might be too simple.
 Alternatively, we could use some higher degree polynomial, though with the way its implemented right now, it's always a lottery.

Let's start with a hypersphere since I understand it better.

"""
import concurrent.futures
import datetime
import math

import os

import pandas as pd


from mlos.OptimizerEvaluationTools.OptimizerEvaluator import OptimizerEvaluator
from mlos.OptimizerEvaluationTools.OptimizerEvaluatorConfigStore import optimizer_evaluator_config_store
from mlos.OptimizerEvaluationTools.SyntheticFunctions.EnvelopedWaves import EnvelopedWaves
from mlos.OptimizerEvaluationTools.SyntheticFunctions.MultiObjectiveEnvelopedWaves import MultiObjectiveEnvelopedWaves

from mlos.Optimizers.BayesianOptimizerConfigStore import bayesian_optimizer_config_store
from mlos.Optimizers.BayesianOptimizerFactory import BayesianOptimizerFactory
from mlos.Optimizers.ExperimentDesigner.UtilityFunctionOptimizers.GlowWormSwarmOptimizer import GlowWormSwarmOptimizer, glow_worm_swarm_optimizer_config_store
from mlos.Optimizers.ExperimentDesigner.UtilityFunctionOptimizers.RandomSearchOptimizer import RandomSearchOptimizer, random_search_optimizer_config_store
from mlos.Optimizers.OptimizationProblem import Objective, OptimizationProblem
from mlos.Spaces import SimpleHypergrid, Point, CategoricalDimension, ContinuousDimension, DiscreteDimension, OrdinalDimension


if __name__ == "__main__":

    num_inner_optimization_iterations = 2000
    reports_dir = r"M:\optimizing_optimizer\feb_03_2021_2"
    os.mkdir(reports_dir)

    target_parameter_space = SimpleHypergrid(
        name="optimizer_parameters",
        dimensions=[
            DiscreteDimension(name="num_decision_trees", min=5, max=20),
            ContinuousDimension(name="samples_fraction_per_tree", min=0.3, max=1),
            CategoricalDimension(name="bootstrap", values=[True, False]),
            DiscreteDimension(name="n_samples_before_refit", min=1, max=20),
            ContinuousDimension(name="fraction_random_suggestions", min=0.2, max=0.5),
            CategoricalDimension('numeric_optimizer_implementation', values=[RandomSearchOptimizer.__name__, GlowWormSwarmOptimizer.__name__]),
            DiscreteDimension(name="num_monte_carlo_samples", min=10, max=1000)
        ]
    ).join(
            subgrid=random_search_optimizer_config_store.parameter_space,
            on_external_dimension=CategoricalDimension('numeric_optimizer_implementation', values=[RandomSearchOptimizer.__name__])
    ).join(
        subgrid=glow_worm_swarm_optimizer_config_store.parameter_space,
        on_external_dimension=CategoricalDimension('numeric_optimizer_implementation', values=[GlowWormSwarmOptimizer.__name__])
    )

    objective_space = SimpleHypergrid(
        name="objective_space",
        dimensions=[
            ContinuousDimension(name="pareto_volume_after_1000_iterations", min=0, max=math.inf),
            ContinuousDimension(name="pareto_volume_after_2000_iterations", min=0, max=math.inf),
            ContinuousDimension(name="duration_s", min=0, max=10*60*60)
        ]
    )

    optimization_problem = OptimizationProblem(
        parameter_space=target_parameter_space,
        objective_space=objective_space,
        objectives=[
            Objective(name="pareto_volume_after_1000_iterations", minimize=False),
            Objective(name="pareto_volume_after_2000_iterations", minimize=False),
            Objective(name="duration_s", minimize=True)
        ]
    )

    meta_optimizer_config = bayesian_optimizer_config_store.get_config_by_name("default_multi_objective_optimizer_config")
    meta_optimizer_config.homogeneous_random_forest_regression_model_config.features_fraction_per_estimator = 1.0

    meta_optimizer = BayesianOptimizerFactory().create_local_optimizer(
        optimization_problem=optimization_problem,
        optimizer_config=meta_optimizer_config
    )

    params_file_path = os.path.join(reports_dir, "params.csv")
    objectives_file_path = os.path.join(reports_dir, "objectives.csv")

    if os.path.exists(params_file_path) and os.path.exists(objectives_file_path):
        try:
            params_df = pd.read_csv(params_file_path)
            objectives_df = pd.read_csv(objectives_file_path)
            meta_optimizer.register(
                parameter_values_pandas_frame=params_df,
                target_values_pandas_frame=objectives_df
            )
        except Exception as e:
            print(e)

    objective_function_config = Point(
        implementation=MultiObjectiveEnvelopedWaves.__name__,
        multi_objective_enveloped_waves_config=Point(
            num_objectives=3,
            phase_difference=1,
            period_change=1.12,
            single_objective_function=EnvelopedWaves.__name__,
            enveloped_waves_config=(Point(
                num_params=10,
                num_periods=100,
                vertical_shift=1,
                phase_shift=0,
                period=2 * math.pi,
                envelope_type="linear",
                sine_envelope_config=Point(
                    amplitude=2,
                    phase_shift=0,
                    period=12 * math.pi
                ),
                linear_envelope_config=Point(slope=30),
                quadratic_envelope_config=Point(a=1, p=20)
            ))
        )
    )

    optimizer_evaluator_config = optimizer_evaluator_config_store.default
    optimizer_evaluator_config.num_iterations = num_inner_optimization_iterations + 1
    optimizer_evaluator_config.evaluation_frequency = int(num_inner_optimization_iterations / 10)
    print(optimizer_evaluator_config)

    def get_suggestion(meta_optimizer):
        target_config = bayesian_optimizer_config_store.default
        suggested_config = meta_optimizer.suggest()

        target_config.homogeneous_random_forest_regression_model_config.n_estimators = suggested_config.num_decision_trees
        target_config.homogeneous_random_forest_regression_model_config.samples_fraction_per_estimator = suggested_config.samples_fraction_per_tree
        target_config.homogeneous_random_forest_regression_model_config.bootstrap = suggested_config.bootstrap
        target_config.homogeneous_random_forest_regression_model_config.decision_tree_regression_model_config.n_new_samples_before_refit = suggested_config.n_samples_before_refit
        target_config.experiment_designer_config.fraction_random_suggestions = suggested_config.fraction_random_suggestions
        target_config.experiment_designer_config.numeric_optimizer_implementation = suggested_config.numeric_optimizer_implementation

        if suggested_config.numeric_optimizer_implementation == GlowWormSwarmOptimizer.__name__:
            target_config.experiment_designer_config.glow_worm_swarm_optimizer_config = suggested_config.glow_worm_swarm_optimizer_config
        elif suggested_config.numeric_optimizer_implementation == RandomSearchOptimizer.__name__:
            target_config.experiment_designer_config.random_search_optimizer_config = suggested_config.random_search_optimizer_config

        assert target_config in bayesian_optimizer_config_store.parameter_space
        return suggested_config, target_config



    max_workers = 10
    max_concurrent_runs = max_workers
    total_runs = 10
    completed_runs = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        outstanding_futures = set()

        while completed_runs < total_runs:

            while len(outstanding_futures) < max_concurrent_runs:
                try:
                    suggested_config, target_config = get_suggestion(meta_optimizer)
                    optimizer_evaluator = OptimizerEvaluator(
                        optimizer_evaluator_config=optimizer_evaluator_config,
                        objective_function_config=objective_function_config,
                        optimizer_config=target_config,
                        suggestion=suggested_config
                    )
                    future = executor.submit(optimizer_evaluator.evaluate_optimizer)
                    meta_optimizer.add_pending_suggestion(suggestion=suggested_config)
                    outstanding_futures.add(future)
                except Exception as e:
                    print(e)

            done_futures, outstanding_futures = concurrent.futures.wait(outstanding_futures, return_when=concurrent.futures.FIRST_COMPLETED)

            for future in done_futures:
                try:
                    optimizer_evaluation_report = future.result()
                    objectives = Point(
                        pareto_volume_after_1000_iterations=sum(optimizer_evaluation_report.pareto_volume_over_time[int(num_inner_optimization_iterations/2)])/2,
                        pareto_volume_after_2000_iterations=sum(optimizer_evaluation_report.pareto_volume_over_time[num_inner_optimization_iterations])/2,
                        duration_s = (optimizer_evaluation_report.end_time - optimizer_evaluation_report.start_time).total_seconds()
                    )
                    parameters = optimizer_evaluation_report.suggestion
                    meta_optimizer.register(
                        parameter_values_pandas_frame=parameters.to_dataframe(),
                        target_values_pandas_frame=objectives.to_dataframe()
                    )
                    completed_runs += 1

                    now = datetime.datetime.utcnow()
                    now_str = now.strftime("%d.%m.%Y.%H.%M.%S.%f")
                    report_dir = os.path.join(reports_dir, now_str)
                    os.mkdir(report_dir)
                    optimizer_evaluation_report.write_to_disk(report_dir)
                except Exception as e:
                    print(e)

            try:
                params_df, objectives_df, _ = meta_optimizer.get_all_observations()
                params_df.to_csv(params_file_path, index=False)
                objectives_df.to_csv(objectives_file_path, index=False)
            except Exception as e:
                print(e)
