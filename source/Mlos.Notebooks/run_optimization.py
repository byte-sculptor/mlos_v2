import random
import pandas as pd


from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import ObjectiveFunctionFactory
from mlos.OptimizerEvaluationTools.SyntheticFunctions.Hypersphere import Hypersphere

from mlos.Optimizers.BayesianOptimizerFactory import BayesianOptimizerFactory, bayesian_optimizer_config_store
from mlos.Optimizers.OptimizationProblem import OptimizationProblem, Objective
from mlos.Optimizers.ParetoFrontier import ParetoFrontier

from mlos.Spaces import ContinuousDimension, DiscreteDimension, Point, SimpleHypergrid



def run_optimization(run_id, max_num_pending_suggestions, num_iterations, add_pending_suggestions=False):
    hypersphere_radius = 10
    objective_function_config = Point(
        implementation=Hypersphere.__name__,
        hypersphere_config=Point(
            num_objectives=5,
            minimize='none',
            radius=hypersphere_radius
        )
    )

    objective_function = ObjectiveFunctionFactory.create_objective_function(objective_function_config=objective_function_config)
    optimization_problem = objective_function.default_optimization_problem
    optimizer = BayesianOptimizerFactory().create_local_optimizer(
        optimization_problem=optimization_problem,
        optimizer_config=bayesian_optimizer_config_store.get_config_by_name("default_multi_objective_optimizer_config")
    )


    pending_suggestions = []
    
    pareto_volume_estimators_over_time = []
    
    num_issued_suggestions = 0
    num_registered_observations = 0
    
    for i in range(num_iterations):
        config = optimizer.suggest()
        num_issued_suggestions += 1
        
        pending_suggestions.append(config)
        
        if add_pending_suggestions:
            optimizer.add_pending_suggestion(config)

        if len(pending_suggestions) >= max_num_pending_suggestions:
            config = pending_suggestions.pop(random.randint(0, len(pending_suggestions) - 1))
            value = objective_function.evaluate_point(config)
            optimizer.register(config.to_dataframe(), value.to_dataframe())
            
            num_registered_observations += 1
            print(f"[{run_id}][{num_registered_observations}/{num_iterations}]")
            if num_registered_observations % 10 == 0:
                volume_estimator = optimizer.pareto_frontier.approximate_pareto_volume()
                pareto_volume_estimators_over_time.append(volume_estimator)
            


    for config in pending_suggestions:
        value = objective_function.evaluate_point(config)
        optimizer.register(config.to_dataframe(), value.to_dataframe())
        num_registered_observations += 1
        if num_registered_observations % 10 == 0:
            volume_estimator = optimizer.pareto_frontier.approximate_pareto_volume()
            pareto_volume_estimators_over_time.append(volume_estimator)

    return add_pending_suggestions, pareto_volume_estimators_over_time