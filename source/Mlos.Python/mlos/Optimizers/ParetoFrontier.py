#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

from mlos.Optimizers.OptimizationProblem import OptimizationProblem
from mlos.Spaces import Hypergrid
from mlos.Tracer import trace
from mlos.Utils.KeyOrderedDict import KeyOrderedDict

class ParetoVolumeEstimator:
    """Contains all information required to compute a confidence interval on the pareto volume.

    Note that the dimensionality analysis for this volume is meaningless. Each objective carries
    its own units, and multiplying them together is meaningless.

    Pareto volume estimate can be used to monitor the growth of the pareto frontier over time.

    """

    def __init__(
            self,
            num_random_points: int,
            num_dominated_points: int,
            objective_space: Hypergrid
    ):
        assert 0 <= num_dominated_points <= num_random_points
        assert len(objective_space.dimensions) > 0
        self.num_random_points = num_random_points
        self.num_dominated_points = num_dominated_points
        self.objective_space: Hypergrid = objective_space
        self.sample_proportion_of_dominated_points = (1.0 * num_dominated_points) / num_random_points

    def get_two_sided_confidence_interval_on_pareto_volume(self, alpha=0.01):
        z_score = norm.ppf(1 - alpha / 2.0)
        p_hat = self.sample_proportion_of_dominated_points
        #use_wilson_score = False
        #if use_wilson_score:
        #    center = (self.num_dominated_points + 0.5 * z_score ** 2) / (self.num_random_points + z_score ** 2)
        #    num_nondominated_points = self.num_random_points - self.num_dominated_points
        #    ci_radius = z_score / (self.num_random_points + z_score * 2) * math.sqrt(
        #        self.num_dominated_points * num_nondominated_points / self.num_random_points + z_score ** 2 / 4)
        #    lower_bound_on_proportion = center - ci_radius
        #    upper_bound_on_proportion = center + ci_radius
        #else:
        ci_radius = z_score * math.sqrt(p_hat * (1 - p_hat) / self.num_random_points)
        lower_bound_on_proportion = p_hat - ci_radius
        upper_bound_on_proportion = p_hat + ci_radius


        total_volume_of_enclosing_parallelotope = 1.0
        for dimension in self.objective_space.dimensions:
            total_volume_of_enclosing_parallelotope *= (dimension.max - dimension.min)

        total_volume_of_enclosing_parallelotope = abs(total_volume_of_enclosing_parallelotope)

        lower_bound_on_pareto_volume = lower_bound_on_proportion * total_volume_of_enclosing_parallelotope
        upper_bound_on_pareto_volume = upper_bound_on_proportion * total_volume_of_enclosing_parallelotope
        return lower_bound_on_pareto_volume, upper_bound_on_pareto_volume





class ParetoFrontier:
    """Maintains a set of non-dominated configurations.

    This class will have several pieces of functionality:
        1. It will be able to construct and maintain a pareto frontier from a set of observations for one or more objectives.
        2. It will be able to update the frontier upon receiving a new observation.
        3. It will be able to decide whether any given point is dominated or not (needed for Monte Carlo utility functions).

    Each point will be characterized by:
        1. Configuration parameters
        2. Objective function values
        3. Possibly context values

    A point belongs to a pareto frontier if it is not dominated by any other point. So if two points have the exact same values
    for all objectives (but possibly different configurations), we will consider both of them to be pareto efficient.


    """

    def __init__(
            self,
            optimization_problem: OptimizationProblem,
            objectives_df: pd.DataFrame = None,
            parameters_df: pd.DataFrame = None
    ):

        self.optimization_problem: OptimizationProblem = optimization_problem
        self._objective_names = optimization_problem.objective_names
        self._pareto_df: pd.DataFrame = None

        # What parameters produced the pareto.
        #
        self._params_for_pareto_df: pd.DataFrame = None

        # Maintains a version of the pareto frontier, where all objectives are set to be maximized. So value for the objectives that were
        # originally meant to be minimized, are multiplied by -1.
        #
        self._pareto_df_maximize_all: pd.DataFrame = None

        if objectives_df is not None:
            assert parameters_df is not None and len(parameters_df.index) == len(objectives_df.index)
            self.update_pareto(objectives_df, parameters_df)

    @property
    def empty(self) -> bool:
        return (self._pareto_df is None) or self._pareto_df.empty

    @property
    def pareto_df(self) -> pd.DataFrame:
        return self._pareto_df.copy(deep=True)

    @property
    def params_for_pareto_df(self):
        if self._params_for_pareto_df is None:
            return None
        return self._params_for_pareto_df.copy(deep=True)

    def update_pareto(self, objectives_df: pd.DataFrame, parameters_df: pd.DataFrame):
        """Computes a pareto frontier for the given objectives_df (including weak-pareto-optimal points).

        We do this by consecutively removing points on the interior of the pareto frontier from objectives_df until none are left.

        We retain the points that fall onto the frontier line, for the following reasons:
            1. The code is more efficient.
            2. If they were jiggled only a little bit outwards they would be included.
            3. In real life we expect it to be an extremely rare occurrence.

        We retain duplicated points because they could be due to different configurations.

        :param optimization_problem:
        :param objectives_df:
        :return:
        """

        assert all(column in self.optimization_problem.objective_space.dimension_names for column in objectives_df.columns)

        # First let's discard any columns that we are not optimizing for.
        #
        pareto_df = objectives_df[self._objective_names]

        # Next, let's turn it into a maximization problem, by flipping the sign of all objectives that are to be minimized.
        #
        pareto_df = self._flip_sign_for_minimized_objectives(pareto_df)

        # By presorting we guarantee, that all dominated points are below the currently considered point.
        #
        pareto_df.sort_values(
            by=[objective.name for objective in self.optimization_problem.objectives],
            ascending=False, # We want the maxima up top.
            inplace=True,
            na_position='last', # TODO: figure out what to do with NaNs.
            ignore_index=False
        )

        current_row_index = 0
        while current_row_index < len(pareto_df.index):
            non_dominated = (pareto_df >= pareto_df.iloc[current_row_index]).any(axis=1)
            pareto_df = pareto_df[non_dominated]
            current_row_index += 1

        self._pareto_df_maximize_all = pareto_df

        # Let's unflip the signs
        #
        pareto_df = self._flip_sign_for_minimized_objectives(pareto_df)
        self._pareto_df = pareto_df
        self._params_for_pareto_df = parameters_df.iloc[self._pareto_df.index]

    @trace()
    def is_dominated(self, objectives_df, reject_equal=False) -> pd.Series:
        """For each row in objectives_df checks if the row is dominated by any of the rows in pareto_df.

        :param objectives_df:
        :param pareto_df:
        :return:
        """
        objectives_df = objectives_df[self._objective_names]
        objectives_df = self._flip_sign_for_minimized_objectives(objectives_df)
        is_dominated = pd.Series([False for i in range(len(objectives_df.index))], index=objectives_df.index)
        for _, pareto_row in self._pareto_df_maximize_all.iterrows():
            if reject_equal:
                is_dominated_by_this_pareto_point = (objectives_df <= pareto_row).all(axis=1)
            else:
                is_dominated_by_this_pareto_point = (objectives_df < pareto_row).all(axis=1)
            is_dominated = is_dominated | is_dominated_by_this_pareto_point
        return is_dominated

    def approximate_pareto_volume(self, num_samples=1000000) -> ParetoVolumeEstimator:
        """Approximates the volume of the pareto frontier.

        The idea here is that we can randomly sample from the objective space and observe the proportion of
        dominated points to all points. This proportion will allow us to compute a confidence interval on
        the proportion of dominated points and we can use it to estimate the ratio between the volume of
        the frontier and the volume from which we sampled.

        We can get arbitrarily precise simply by drawing more samples.
        """

        # First we need to find the extremes for each of the objective values.
        #
        #objective_minima = KeyOrderedDict(ordered_keys=list(self.pareto_df.columns), value_type=float)
        #objective_maxima = KeyOrderedDict(ordered_keys=list(self.pareto_df.columns), value_type=float)
        #objective_ranges = KeyOrderedDict(ordered_keys=list(self.pareto_df.columns), value_type=float)

        #for objective in self.optimization_problem.objectives:
        #    min_objective_value = self.pareto_df[objective.name].min()
        #    max_objective_value = self.pareto_df[objective.name].max()
        #    objective_minima[objective.name] = min_objective_value
        #    objective_maxima[objective.name] = max_objective_value
        #    objective_ranges[objective.name] = max_objective_value - min_objective_value


        #random_points_array = np.random.uniform(low=0.0, high=1.0, size=(len(objective_ranges), num_samples))
        #random_objectives_df = pd.DataFrame({
        #    objective_name: random_points_array[i] * objective_range + objective_minima[objective_name]
        #    for i, (objective_name, objective_range)
        #    in enumerate(objective_ranges)
        #})

        #for objective in self.optimization_problem.objectives:
        #    assert (random_objectives_df[objective.name] >= objective_minima[objective.name]).all()
        #    assert (random_objectives_df[objective.name] <= objective_maxima[objective.name]).all()

        random_objectives_df = self.optimization_problem.objective_space.random_dataframe(
            num_samples=num_samples
        )

        num_dominated_points = self.is_dominated(objectives_df=random_objectives_df).sum()
        return ParetoVolumeEstimator(
            num_random_points=num_samples,
            num_dominated_points=num_dominated_points,
            objective_space=self.optimization_problem.objective_space
        )

    def compute_pareto_volume(self) -> float:
        """Analytically computes the pareto volume."""
        n_axes = len(self.optimization_problem.objectives)
        pareto_df = self.pareto_df.copy(deep=True)
        pareto_df = pareto_df.drop_duplicates()

        pareto_df = self._flip_sign_for_minimized_objectives(pareto_df)

        # We want all points in Q1, so we need to subtract the dimension minimum from points.
        for objective in self.optimization_problem.objectives:
            objective_dimension = self.optimization_problem.objective_space[objective.name]
            if objective.minimize:
                pareto_df[objective.name] += objective_dimension.max
            else:
                pareto_df[objective.name] -= objective_dimension.min

        assert (pareto_df >= -1e-12).all().all()


        if n_axes == 2:
            # The 2D case is simple and faster to compute, so we special case it.
            diffs = pareto_df.iloc[:, 1].diff()
            diffs.iloc[0] = pareto_df.iloc[0, 1]
            area = (diffs * pareto_df.iloc[:, 0]).sum()
            return area

        # partition the n-d space by all occurring values for each axes
        # first, sort all the axes separately
        sorted_corners_df = pd.DataFrame(np.sort(pareto_df.values, axis=0), columns=pareto_df.columns)

        # Then, generate all permutations of indices to get all the points on the partition grid
        # The shape of this array is (n_points, n_axes) where n_points is the number of cells in the partition,
        # So if there's three objectives, and 10 points on the pareto frontier,
        # there will be 10 ** 3 cells, and this array will have shape (10 ** 3, 3)
        all_permutations = np.array(np.meshgrid(*[range(len(pareto_df))] * n_axes)).reshape(n_axes, -1).T
        # Compute the length of the edges in the partition
        edge_lengths_df = pd.DataFrame()
        for col in sorted_corners_df.columns:
            steps = sorted_corners_df[col].diff()
            # The first length is the difference to zero, which is just the original value.
            # Alternatively one could add [0, 0] to the pareto curve.
            steps.iloc[0] = sorted_corners_df[col].iloc[0]
            edge_lengths_df[col] = steps
        # we need numpy arrays for fancy indexing
        edge_length_array = np.array(edge_lengths_df)
        corners_array = np.array(sorted_corners_df)
        # now we materialize the corners of all the cells
        all_points = np.c_[[corners_array[all_permutations[:, i], i] for i in range(n_axes)]].T
        # we also compute the volume for each cell
        all_areas = np.c_[[edge_length_array[all_permutations[:, i], i] for i in range(n_axes)]].prod(axis=0)
        # Then we check whether for each cell whether it is on the inside of the Pareto frontier
        all_cell_corner_points_df = pd.DataFrame(all_points, columns=pareto_df.columns)

        for objective in self.optimization_problem.objectives:
            objective_dimension = self.optimization_problem.objective_space[objective.name]
            if objective.minimize:
                pareto_df[objective.name] -= objective_dimension.max
            else:
                pareto_df[objective.name] += objective_dimension.min

        all_cell_corner_points_df = self._flip_sign_for_minimized_objectives(all_cell_corner_points_df)

        is_inside = self.is_dominated(all_cell_corner_points_df, reject_equal=True)
        # And finally we sum up all the cells using the result as a boolean mask.
        return all_areas[is_inside].sum()

    def _flip_sign_for_minimized_objectives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Takes a data frame in objective space and multiplies all minimized objectives by -1.

        The point of this is to convert all problems (minimization and maximization) to maximization problems to simplify
        implementation on everything else.

        :param df:
        :return:
        """
        output_df = df.copy(deep=True)
        for objective in self.optimization_problem.objectives:
            if objective.minimize:
                output_df[objective.name] = -output_df[objective.name]
        return output_df
