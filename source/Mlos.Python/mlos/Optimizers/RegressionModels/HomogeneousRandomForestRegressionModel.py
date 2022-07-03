#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import math
import random

import pandas as pd

from mlos.Spaces import Hypergrid, Point
from mlos.Tracer import trace
from mlos.Logger import create_logger
from mlos.Optimizers.RegressionModels.Prediction import Prediction
from mlos.Optimizers.RegressionModels.DecisionTreeRegressionModel import DecisionTreeRegressionModel
from mlos.Optimizers.RegressionModels.EnsembleRegressionModelBase import EnsembleRegressionModelBase
from mlos.Optimizers.RegressionModels.HomogeneousRandomForestConfigStore import homogeneous_random_forest_config_store
from mlos.Optimizers.RegressionModels.HomogeneousRandomForestFitState import HomogeneousRandomForestFitState
from mlos.Optimizers.RegressionModels.RegressionModel import RegressionModel


class HomogeneousRandomForestRegressionModel(EnsembleRegressionModelBase):
    """ A RandomForest with homogeneously configured trees.

    This is the first implementation of a random forest regressor (and more generally of an ensemble model)
    that returns variance in addition to prediction. This should allow us to build a more robust bayesian
    optimizer.

    1. In this random forest, all decision trees are uniformly configured.
    2. Each decision tree receives a subset of features and a subset of rows.

    """

    @trace()
    def __init__(
            self,
            model_config: Point,
            input_space: Hypergrid,
            output_space: Hypergrid,
            logger=None
    ):
        if logger is None:
            logger = create_logger("HomogeneousRandomForestRegressionModel")
        self.logger = logger

        assert model_config in homogeneous_random_forest_config_store.parameter_space

        EnsembleRegressionModelBase.__init__(
            self,
            model_type=type(self),
            model_config=model_config,
            input_space=input_space,
            output_space=output_space,
            fit_state=HomogeneousRandomForestFitState()
        )

    @property
    def trained(self):
        return self._trained

    @trace()
    def _create_regressors(self):
        """ Create individual estimators.

        Each estimator is meant to have a different subset of features and a different subset of samples.

        In the long run, we can solve it by creating an DataSet or DataSetView class, then each
        estimator would have its own DataSetView object that would know which data points to fetch and
        how to do it.

        For now however, I'll do it here in-line to get it working.

        1. Selecting features - to select a subset of features for each estimator we will:
            1. Create a random valid point in the search space. Any such valid config will not contain mutually exclusive
                parameters (for example if we chose an LRU eviction strategy, we won't get any parameters for a Random eviction strategy)
            2. Select a subset of dimensions from such a valid point so that we comply with the 'features_fraction_per_estimator' value.
            3. Build an input hypergrid for each estimator. However dimensions from above could be deeply nested.
                For example: smart_cache_config.lru_cache_config.lru_cache_size. We don't need the individual hypergrids
                to be nested at all so we will 'flatten' the dimension name by replacing the '.' with another delimiter.
                Then for each observation we will flatten the names again to see if the observation belongs to our observation
                space. If it does, then we can use our observation selection filter to decide if we want to feed that observation
                to the model. I'll leave the observation selection filter implementation for some other day.

        :return:
        """
        # Now we get to create all the estimators, each with a different feature subset and a different
        # observation filter

        self.logger.info(f"Creating {self.model_config.n_estimators} estimators. "
                         f"Tree config: {self.model_config.decision_tree_regression_model_config}. "
                         f"Request id: {random.random()}")

        all_dimension_names = [dimension.name for dimension in self.input_space.dimensions]
        total_num_dimensions = len(all_dimension_names)
        features_per_estimator = max(1, math.ceil(total_num_dimensions * self.model_config.features_fraction_per_estimator))

        for i in range(self.model_config.n_estimators):
            estimator_input_space = self._create_random_flat_subspace(
                original_space=self.input_space,
                subspace_name=f"estimator_{i}_input_space",
                max_num_dimensions=features_per_estimator
            )
            self.logger.info(f"Creating DecisionTreeRegressionModel with the input_space: {estimator_input_space}")

            estimator = DecisionTreeRegressionModel(
                model_config=self.model_config.decision_tree_regression_model_config,
                input_space=estimator_input_space,
                output_space=self.output_space,
                logger=self.logger
            )

            # TODO: each one of them also needs a sample filter.
            self._regressors.append(estimator)
            self.fit_state.decision_trees_fit_states.append(estimator.fit_state)

    @trace()
    def predict(
        self,
        feature_values_pandas_frame: pd.DataFrame,
        include_only_valid_rows: bool = True
    ):
        """ Aggregate predictions from all estimators

        see: https://arxiv.org/pdf/1211.0906.pdf
        section: 4.3.2 for details about how it's done for RandomForest

        :param feature_values_pandas_frame:
        :include_only_valid_rows:
        :return: Prediction
        """
        self.logger.debug(f"Creating predictions for {len(feature_values_pandas_frame.index)} samples.")

        feature_values_pandas_frame = self._input_space_adapter.project_dataframe(feature_values_pandas_frame, in_place=False)

        # dataframe column shortcuts
        is_valid_input_col = Prediction.LegalColumnNames.IS_VALID_INPUT.value
        predicted_value_col = Prediction.LegalColumnNames.PREDICTED_VALUE.value
        predicted_value_var_col = Prediction.LegalColumnNames.PREDICTED_VALUE_VARIANCE.value
        sample_var_col = Prediction.LegalColumnNames.SAMPLE_VARIANCE.value
        sample_size_col = Prediction.LegalColumnNames.SAMPLE_SIZE.value
        dof_col = Prediction.LegalColumnNames.PREDICTED_VALUE_DEGREES_OF_FREEDOM.value

        # collect predictions from ensemble constituent models
        predictions_per_regressor = [
            regressor.predict(feature_values_pandas_frame=feature_values_pandas_frame, include_only_valid_rows=True)
            for regressor in self._regressors
        ]
        prediction_dfs_per_regressor = [prediction.get_dataframe() for prediction in predictions_per_regressor]
        num_prediction_dfs = len(prediction_dfs_per_regressor)

        # We will concatenate all these prediction dataframes together, but to avoid duplicate columns, we first rename them.
        #
        old_names = [predicted_value_col, predicted_value_var_col, sample_var_col, sample_size_col, dof_col]
        predicted_value_col_names_per_regressor = [f"{predicted_value_col}_{i}" for i in range(num_prediction_dfs)]
        mean_var_col_names_per_regressor = [f"{predicted_value_var_col}_{i}" for i in range(num_prediction_dfs)]
        sample_var_col_names_per_regressor = [f"{sample_var_col}_{i}" for i in range(num_prediction_dfs)]
        sample_size_col_names_per_regressor = [f"{sample_size_col}_{i}" for i in range(num_prediction_dfs)]

        for i in range(num_prediction_dfs):
            new_names = [f"{old_name}_{i}" for old_name in old_names]
            old_names_to_new_names_mapping = {old_name: new_name for old_name, new_name in zip(old_names, new_names)}
            prediction_dfs_per_regressor[i].drop(columns=[is_valid_input_col], inplace=True)
            # We can safely overwrite them in place since we are their sole owner by now.
            prediction_dfs_per_regressor[i].rename(columns=old_names_to_new_names_mapping, inplace=True)

        # This creates a 'wide' dataframe with unique column names.
        #
        all_predictions_df = pd.concat(prediction_dfs_per_regressor, axis=1)
        all_predictions_df[predicted_value_col] = all_predictions_df[predicted_value_col_names_per_regressor].apply('mean', axis=1)

        # To compute the pooled variance we will use the second to last form of the equation from the paper:
        #   paper: https://arxiv.org/pdf/1211.0906.pdf
        #   section: section: 4.3.2 for details
        all_predictions_df[predicted_value_var_col] = all_predictions_df[mean_var_col_names_per_regressor].mean(axis=1) \
                                                      + (all_predictions_df[predicted_value_col_names_per_regressor] ** 2).mean(
            axis=1) \
                                                      - all_predictions_df[
                                                          predicted_value_col] ** 2 + 0.0000001  # A little numerical instability correction
        all_predictions_df[sample_var_col] = all_predictions_df[sample_var_col_names_per_regressor].mean(axis=1) \
                                             + (all_predictions_df[predicted_value_col_names_per_regressor] ** 2).mean(axis=1) \
                                             - all_predictions_df[
                                                 predicted_value_col] ** 2 + 0.0000001  # A little numerical instability correction
        all_predictions_df[sample_size_col] = all_predictions_df[predicted_value_col_names_per_regressor].count(axis=1)
        all_predictions_df[dof_col] = all_predictions_df[sample_size_col_names_per_regressor].sum(axis=1) - all_predictions_df[
            sample_size_col]
        all_predictions_df[is_valid_input_col] = True

        aggregate_predictions = Prediction(
            objective_name=self.target_dimension_names[0],
            predictor_outputs=self._PREDICTOR_OUTPUT_COLUMNS,
            allow_extra_columns=True
        )

        aggregate_predictions_df = all_predictions_df[[column.value for column in self._PREDICTOR_OUTPUT_COLUMNS]]
        aggregate_predictions.set_dataframe(aggregate_predictions_df)
        if not include_only_valid_rows:
            aggregate_predictions.add_invalid_rows_at_missing_indices(desired_index=feature_values_pandas_frame.index)
        return aggregate_predictions
