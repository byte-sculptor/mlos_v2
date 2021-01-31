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

from mlos.Optimizers.BayesianOptimizerConfigStore import bayesian_optimizer_config_store
from mlos.Spaces import SimpleHypergrid, Point, CategoricalDimension, ContinuousDimension, DiscreteDimension, OrdinalDimension

from mlos.Optimizers.ExperimentDesigner.UtilityFunctionOptimizers.RandomSearchOptimizer import RandomSearchOptimizer, random_search_optimizer_config_store
from mlos.Optimizers.ExperimentDesigner.UtilityFunctionOptimizers.GlowWormSwarmOptimizer import GlowWormSwarmOptimizer, glow_worm_swarm_optimizer_config_store


print(bayesian_optimizer_config_store.default)

parameter_space = SimpleHypergrid(
    name="optimizer_parameters",
    dimensions=[
        DiscreteDimension(name="num_decision_trees", min=5, max=20),
        ContinuousDimension(name="samples_fraction_per_tree", min=0.3, max=1),
        CategoricalDimension(name="bootstrap", values=[True, False]),
        DiscreteDimension(name="n_samples_before_refit", min=1, max=20),
        ContinuousDimension(name="fraction_random_suggestions", min=0, max=1),
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

print(parameter_space)

parameter_space
